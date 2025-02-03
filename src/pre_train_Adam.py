import sys
import os

cur_path = os.path.split(__file__)[0]
sys.path.append(cur_path)

import torch
from torch.optim import Adam
import numpy as np
from FVMmodel.importer import NNmodel as Simulator

# import os
from Load_mesh import Graph_loader
from Utils import get_param, scheduler
import time
from Utils.get_param import get_hyperparam
from Utils.Logger import Logger
from math import ceil
import random
import datetime

# import logging
# logging.basicConfig(level=logging.DEBUG)
# torch._dynamo.config.verbose = True
# torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')

# configurate parameters
params = get_param.params()
seed = int(datetime.datetime.now().timestamp())
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(0.99, params.on_gpu)
torch.set_num_threads(os.cpu_count() // 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# initialize Logger and load model / optimizer if according parameters were given
logger = Logger(
    get_hyperparam(params),
    use_csv=True,
    use_tensorboard=False,
    params=params,
    copy_code=True,
    seed=seed,
)

# initialize Training Dataset
start = time.time()
datasets_factory = Graph_loader.DatasetFactory(
    params=params,
    dataset_dir=params.dataset_dir,
    state_save_dir=logger.saving_path,
    device=device,
)

# refresh dataset size
params.dataset_size = datasets_factory.dataset_size

# create dataset objetc
datasets, loader, sampler = datasets_factory.create_datasets(
    batch_size=params.batch_size, num_workers=0, pin_memory=True
)

end = time.time()
print("Training traj has been loaded time consuming:{0}".format(end - start))

# initialize fluid model
model = Simulator(params)
# fluid_model = torch.compile(model.to(device))
fluid_model = model.to(device)
fluid_model.train()
optimizer = Adam(fluid_model.parameters(), lr=params.lr)

""" >>> lr scheduler settings >>> """
steplr_decay_steps = int(params.n_epochs * 0.1)
explr_decay_steps = int(params.n_epochs * 0.5)
lr_scheduler = scheduler.StepexpLRScheduler(
    optimizer=optimizer,
    startlr=params.lr,
    steplr_milestone=steplr_decay_steps,
    steplr_gamma=1,
    explr_milestone=explr_decay_steps,
    explr_gamma=1e-1,
    total_epoch=params.n_epochs,
    min_lr=1e-6,
)
""" <<< lr scheduler settings <<< """

if (
    params.load_date_time is not None
    or params.load_index is not None
):
    logger.load_logger(datetime=params.load_date_time)
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

train_steps=0
# training loop
for epoch in range(params.n_epochs):
    fluid_model.train()
    payback = False

    if epoch % ceil(params.average_sequence_length / (params.dataset_size)) == 0:
        rst_time = ceil(params.dataset_size / (params.average_sequence_length))
        datasets._set_reset_env_flag(flag=True, rst_time=rst_time)

    start = time.time()

    for i_iter in range(params.max_inner_steps):
        train_steps+=1
        sampler.set_epoch(train_steps)
        
        if i_iter == params.max_inner_steps - 1:
            payback=True
            
        for batch_index, (
            graph_node,
            graph_node_x,
            graph_edge,
            graph_cell,
            graph_Index,
        ) in enumerate(loader):

            global_idx = graph_node.global_idx # backup cpu tensor
            
            ''' >>> please check src/Load_mesh/Graph_loader.py/->update_x_attr >>> '''
            graph_node.norm_uvp=params.norm_uvp
            graph_node.norm_global=params.norm_global
            ''' <<< please check src/Load_mesh/Graph_loader.py/->update_x_attr <<< '''
            
            (
                graph_node,
                graph_node_x,
                graph_edge,
                graph_cell,
                graph_Index,
            ) = datasets.datapreprocessing(
                graph_node=graph_node.cuda(),
                graph_node_x=graph_node_x.cuda(),
                graph_edge=graph_edge.cuda(),
                graph_cell=graph_cell.cuda(),
                graph_Index=graph_Index.cuda(),
            )

            optimizer.zero_grad()

            (
                loss_cont,
                loss_mom_x,
                loss_mom_y,
                loss_press,
                uvp_node_new,
                _,
            ) = fluid_model(
                graph_node=graph_node,
                graph_node_x=graph_node_x,
                graph_edge=graph_edge,
                graph_cell=graph_cell,
                graph_Index=graph_Index,
                is_training=True,
            )

            ''' back up exclude LSFD'''
            loss_batch = (
                params.loss_press * loss_press
                + params.loss_cont * loss_cont
                + params.loss_mom * loss_mom_x
                + params.loss_mom * loss_mom_y
            )

            loss = torch.mean(torch.log(loss_batch))
            ''' back up exclude LSFD'''
            
            # compute gradients
            loss.backward()
            
            # perform optimization step
            optimizer.step()
            
            if payback:
                # put back to dataset pool
                datasets.payback(
                    uvp_new=uvp_node_new.detach().cpu(),
                    global_idx=global_idx,
                )

    # 输出此epoch的处理时间
    print(f"Epoch {epoch} completed in {time.time() - start:.2f} seconds")
    print(f"Epoch {epoch} Loss {loss.item():.5f}")
    lr_scheduler.step()

    # save state after every 2 epoch
    if (epoch % 50 == 0) or (epoch == params.n_epochs - 1):
        model_saving_path = logger.save_state(
            model=fluid_model,
            optimizer=None,
            scheduler=None,
            index=str(epoch % 3),
        )
