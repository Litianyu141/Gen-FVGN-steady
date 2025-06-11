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
from FVMmodel.FVdiscretization.FVgrad import weighted_lstsq
from torch_geometric.data.batch import Batch
import random
import datetime

''' >>> 单独设置参数 >>> '''
params = get_param.params()
params.batch_size=4
params.dataset_size=4
params.load_date_time=None # str
params.load_index=None # int
params.on_gpu=1
params.dataset_dir = "datasets/lid_driven_cavity/lid_driven_cavity_101x101"
params.n_epochs = 40000
params.max_inner_steps = 100
params.norm_global=False # 先默认为False,如果读取的Logger里面为True,则会自动改为True
logger_head = "Logger"
''' <<< 单独设置参数 <<< '''

# configurate parameters
seed = int(datetime.datetime.now().timestamp())
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.set_float32_matmul_precision('high')
torch.cuda.set_per_process_memory_fraction(0.99, params.on_gpu)
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
    batch_size=params.batch_size, num_workers=0, pin_memory=False
)
end = time.time()
print("Training traj has been loaded time consuming:{0}".format(end - start))

# initialize fluid model
model = Simulator(params)
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

""" >>> load state from old date >>> """
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
""" <<< load state from old date <<< """

''' >>> fetch data and move to GPU >>> '''
(graph_node,graph_cell_x,graph_edge,graph_cell,graph_Index) = next(iter(loader))
(
    graph_node,
    graph_cell_x,
    graph_edge,
    graph_cell,
    graph_Index,
) = datasets.datapreprocessing(
    graph_node=graph_node.cuda(),
    graph_cell_x=graph_cell_x.cuda(),
    graph_edge=graph_edge.cuda(),
    graph_cell=graph_cell.cuda(),
    graph_Index=graph_Index.cuda(),
)
''' <<< fetch data and move to GPU <<< '''
init_loss = torch.ones((graph_cell.num_graphs,4),device=device)

for epoch in range(params.n_epochs+1):
    fluid_model.train() 
    start = time.time()
    
    uvp_pde_theta_backup = graph_node.x.clone()

    # 内迭代时保证每次输入都一样
    for i_iter in range(params.max_inner_steps):
        
        ''' >>> please check src/Load_mesh/Graph_loader.py/->update_x_attr >>> '''
        graph_node.x = uvp_pde_theta_backup
        graph_node.norm_uvp=params.norm_uvp
        graph_node.norm_global=params.norm_global
        ''' <<< please check src/Load_mesh/Graph_loader.py/->update_x_attr <<< '''
        
        optimizer.zero_grad()
        
        (
            loss_cont,
            loss_mom_x,
            loss_mom_y,
            loss_press,
            uvp_node_new,
            uvp_cell_new,
        ) = fluid_model(
            graph_node=graph_node,
            graph_cell_x=graph_cell_x,
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

    graph_node.x[:,0:3] = uvp_node_new.detach()
    graph_cell.x[:,0:3] = uvp_cell_new.detach()  

    graph_cell_x_list = Batch.to_data_list(graph_cell_x)
    graph_cell_list = Batch.to_data_list(graph_cell)
    
    for plot_order in range(len(graph_cell_list)):
        i_graph_cell = graph_cell_list[plot_order].cpu()
        datasets.export_to_tecplot(datasets.meta_pool[i_graph_cell.graph_index], i_graph_cell.x[:,0:3].detach() , datalocation="cell")
        ''' <<< plot at cell-center <<< '''

    # 输出此epoch的处理时间
    print(f"Epoch {epoch} completed in {time.time() - start:.5f} seconds")
    print(f"Epoch {epoch} Loss {loss.item():.5f}")
    lr_scheduler.step()

    # 替换以下graph.x以实现时间推进
    graph_node.x = torch.cat((uvp_node_new.detach(),uvp_pde_theta_backup[:,params.node_phi_size:]),dim=1)
    
    # save state after every 2 epoch
    if (epoch % 2 == 0) or (epoch == params.n_epochs - 1):
        model_saving_path = logger.save_state(
            model=fluid_model,
            optimizer=None,
            scheduler=None,
            index=str(epoch % 3),
        )