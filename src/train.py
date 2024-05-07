import sys
import os

sys.path.append(os.path.split(__file__)[0])

import torch
from torch.optim import AdamW
import numpy as np
from FVMmodel.GNNSolver import GenFVGN

# import os
from dataset import Load_mesh
from utils import get_param, scheduler, noise
import time
from utils.get_param import get_hyperparam
from utils.Logger import Logger
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.data.batch import Batch
from math import ceil
import random
import datetime

# configurate parameters
params, git_info = get_param.params()

# git information
if git_info is not False:
    git_info = {
        "git_branch": params.git_branch,
        "git_commit_dates": params.git_commit_dates,
    }
else:
    git_info = {"git_branch": " ", "git_commit_dates": " "}

# for saving model
# torch.manual_seed(0)
# torch.set_num_threads(4)
# check cuda
random.seed(int(datetime.datetime.now().timestamp()))
torch.cuda.set_per_process_memory_fraction(0.99, params.on_gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize Logger and load model / optimizer if according parameters were given
logger = Logger(
    get_hyperparam(params),
    use_csv=False,
    use_tensorboard=params.log,
    params=params,
    git_info=git_info,
    copy_code=True,
)

# logging gradient
grad_count = 0


def print_grad(grad, name):
    global grad_count
    grad_count += 1
    if grad_count % 1000 == 0:
        logger.log(f"Gradient/{name}_mean", grad.mean(), grad_count)
        logger.log(f"Gradient/{name}_std", grad.std(), grad_count)
        logger.log(f"Gradient/{name}_max", grad.max(), grad_count)
        logger.log(f"Gradient/{name}_min", grad.min(), grad_count)


def register_hooks(module):
    for name, param in module.named_parameters():
        if "weight" in name:  # 为权重参数注册钩子
            param.register_hook(lambda grad: print_grad(grad, name))


# initialize Training Dataset
start = time.time()
datasets_factory = Load_mesh.DatasetFactory(
    params=params,
    is_training=True,
    device=device,
    state_save_dir=logger.saving_path,
    split="train",
    inflow_bc_type="parabolic_velocity_field",
    dataset_dir=params.dataset_dir,
)

# refresh dataset size
params.dataset_size = datasets_factory.dataset_size

# create dataset objetc
datasets, loader, sampler = datasets_factory.create_datasets(
    batch_size=params.batch_size, num_workers=0, pin_memory=False
)

end = time.time()
print("Training traj has been loaded time consuming:{0}".format(end - start))

# initialize fluid model
model = GenFVGN(
    message_passing_num=params.message_passing_num,
    node_input_size=params.node_input_size + params.node_one_hot,
    edge_input_size=params.edge_input_size + params.edge_one_hot,
    cell_input_size=params.cell_input_size + params.cell_one_hot,
    node_output_size=params.node_output_size,
    edge_output_size=params.edge_output_size,
    cell_output_size=params.cell_output_size,
    drop_out=params.drop_out,
    attention=params.attention,
    MultiHead=params.multihead,
    hidden_size=params.hidden_size,
    normlizer_steps=25 * ceil(params.dataset_size / params.batch_size),
    device=device,
)

fluid_model = model.to(device)
fluid_model.train()
optimizer = AdamW(fluid_model.parameters(), lr=params.lr)
two_step_scheduler = scheduler.ExpLR(
    optimizer, decay_steps=params.n_epochs - params.before_explr_decay_steps, gamma=1e-4
)
lr_scheduler = scheduler.GradualStepExplrScheduler(
    optimizer,
    multiplier=1.0,
    milestone=[6000],
    gamma=1,
    total_epoch=params.before_explr_decay_steps,
    after_scheduler=two_step_scheduler,
    expgamma=1e-2,
    decay_steps=params.n_epochs - params.before_explr_decay_steps,
    min_lr=1e-6,
)

if (
    params.load_latest
    or params.load_date_time is not None
    or params.load_index is not None
):
    logger.load_logger(datetime=params.load_date_time)
    # load_logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=params.log,params=params,git_info=git_info)
    if params.load_optimizer:
        params.load_date_time, params.load_index = logger.load_state(
            model=fluid_model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            datetime=params.load_date_time,
            index=params.load_index,
            device=device,
        )
    else:
        params.load_date_time, params.load_index = logger.load_state(
            model=fluid_model,
            optimizer=None,
            scheduler=None,
            datetime=params.load_date_time,
            index=params.load_index,
            device=device,
        )
    params.load_index = int(params.load_index)
    print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index
# model.apply(register_hooks)


# training loss function
def loss_function(x):
    return torch.pow(x, 2)


load_index = params.load_index
n_epochs = params.n_epochs
traj_length = params.traj_length
epoch_rounds = int(params.dataset_size / params.batch_size)

# logger.plot_unv(curent_sample=None,graph_list=[graph_node,graph_edge,graph_cell],plot_index=10)
payback = False
params.accumulated_flag = False
# training loop
for epoch in range(n_epochs):
    fluid_model.train()
    sampler.set_epoch(epoch)

    if epoch % ceil(params.average_sequence_length / (params.dataset_size)) == 0:
        if params.average_sequence_length / (params.dataset_size) < 1:
            rst_time = ceil(params.dataset_size / (params.average_sequence_length))
        else:
            rst_time = ceil(params.average_sequence_length / (params.dataset_size))
        datasets._set_reset_env_flag(flag=True, rst_time=rst_time)
        params.accumulated_flag = True

    if epoch % params.iner_step == 0:
        payback = True

    if epoch % 5 == 0:
        datasets._set_plot_flag(_plot=True)

    start = time.time()
    for batch_index, (
        graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_cell_x,
    ) in enumerate(loader):
        training_start_time = time.time()
        optimizer.zero_grad()

        (
            graph_node,
            graph_node_x,
            graph_edge,
            graph_cell,
            graph_cell_x,
        ) = datasets.datapreprocessing(
            graph_node=graph_node.cuda(),
            graph_node_x=graph_node_x.cuda(),
            graph_edge=graph_edge.cuda(),
            graph_cell=graph_cell.cuda(),
            graph_cell_x=graph_cell_x.cuda(),
            dimless=params.dimless,
        )

        (
            predicted_node_uvp,
            loss_cont,
            loss_momtentum_x,
            loss_momtentum_y,
            loss_pressure_outlet,
            projection_method,
        ) = fluid_model(
            graph_cell=graph_cell,
            graph_node_x=graph_node_x,
            graph_edge=graph_edge,
            graph_node=graph_node,
            graph_cell_x=graph_cell_x,
            params=params,
        )

        """ continuity equation loss and momentum equation loss"""
        loss_cont = global_mean_pool(
            loss_function(loss_cont), batch=graph_cell.batch.cuda()
        ).view(-1)

        loss_mom = global_mean_pool(
            loss_function(loss_momtentum_x), batch=graph_cell.batch.cuda()
        ).view(-1) + global_mean_pool(
            loss_function(loss_momtentum_y), batch=graph_cell.batch.cuda()
        ).view(
            -1
        )

        """ optional loss term"""
        try:
            loss_projection_method = global_mean_pool(
                loss_function(projection_method), batch=graph_cell.batch.cuda()
            ).view(-1)
        except:
            loss_projection_method = 0.0

        try:
            # retrive loss_pressure_outlet shape
            num_columns = loss_pressure_outlet.shape[-1]

            # if loss_pressure_outlet shapes at [100, 1]
            if num_columns == 1:
                loss_pressure_outlet = global_mean_pool(
                    loss_function(loss_pressure_outlet[:, 0:1]),
                    batch=graph_cell.batch.cuda()[graph_cell.face[0]],
                ).view(-1)

            # if loss_pressure_outlet shapes at [100, 2]
            elif num_columns == 2:
                loss_pressure_outlet = global_mean_pool(
                    loss_function(loss_pressure_outlet[:, 0:1]),
                    batch=graph_cell.batch.cuda()[graph_cell.face[0]],
                ).view(-1) + global_mean_pool(
                    loss_function(loss_pressure_outlet[:, 1:2]),
                    batch=graph_cell.batch.cuda()[graph_cell.face[0]],
                ).view(-1)

        except:
            # otherwise
            loss_pressure_outlet = 0

        loss = (
            params.loss_cont * loss_cont
            + params.loss_mom * loss_mom
            + params.pressure_open_bc * loss_pressure_outlet
        )

        loss = params.loss_multiplier * torch.mean(torch.log(loss))

        # compute gradients
        loss.backward()

        # perform optimization step
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        optimizer.step()

        if payback:
            # We have already set all predicted value into original dimension during forward process
            with_dim_predicted_node_uvp = predicted_node_uvp.detach()

            graph_node.x = with_dim_predicted_node_uvp
            list(map(datasets.payback_test, Batch.to_data_list(graph_node)))

    payback = False
    params.accumulated_flag = False

    if epoch % params.iner_step == 0:
        loss = loss.detach().cpu()
        loss_mom = loss_mom.detach().cpu()
        loss_cont = loss_cont.detach().cpu()

        # Using formatted string for better readability
        headers = ["Epoch", "Loss", "Momentum Loss", "Continuity Loss", "Learning Rate", "Epoch Time", "Git Commit Date", "Current Branch"]
        values = [
            epoch,
            f"{loss.mean():.4f}",
            f"{loss_mom.mean():.4e}",
            f"{loss_cont.mean():.4e}",
            f"{learning_rate:.2e}",
            f"{time.time() - start:.2f}s",
            params.git_commit_dates,
            params.git_branch
        ]

        # Determine the maximum width for each column
        column_widths = [max(len(str(x)), len(headers[i])) + 2 for i, x in enumerate(values)]

        # Create a format string for each row
        row_format = "".join(["{:<" + str(width) + "}" for width in column_widths])

        # Print the headers and values
        print(row_format.format(*headers))
        print(row_format.format(*values))

        # Update logger with formatted strings
        logger.log(f"loss_{params.loss}", loss.mean(), epoch)
        logger.log("loss_mom", loss_mom.mean(), epoch)
        logger.log("loss_cont", loss_cont.mean(), epoch)
        logger.log("learning_rate", learning_rate, epoch)

        start = time.time()

    lr_scheduler.step()

    # save state after every 20 epoch
    if (epoch % 20 == 0 and params.log) or (epoch == n_epochs - 1):
        model_saving_path = logger.save_state(
            model=fluid_model, optimizer=optimizer, scheduler=lr_scheduler, index=epoch % 3
        )
