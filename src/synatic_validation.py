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
import csv

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

# Scalar Eular solution
def Scalar_Eular_solution(mesh_pos, phi_0, phi_x, phi_y, phi_xy, alpha_x, alpha_y, alpha_xy, L):
    """
    使用PyTorch计算给定网格点上的φ值及其对x和y的一阶导数。
    
    参数:
    mesh_pos: 一个形状为(100, 2)的张量，表示100个网格点的x和y坐标。
    phi_0, phi_x, phi_y, phi_xy: 公式中的系数。
    alpha_x, alpha_y, alpha_xy: 公式中的alpha参数。
    L: 域的长度。

    返回:
    phi_values: 一个张量，包含每个网格点上的φ值。
    dphi_dx: φ对x的一阶导数。
    dphi_dy: φ对y的一阶导数。
    """
    # 检查输入参数类型
    if not isinstance(mesh_pos, torch.Tensor):
        raise TypeError("mesh_pos must be a torch.Tensor")
    if not mesh_pos.dtype == torch.float32:
        raise TypeError("mesh_pos must be of type torch.float32")

    # 确保mesh_pos需要计算梯度
    node_pos = mesh_pos.clone()
    node_pos.requires_grad_(True)

    # 提取x和y坐标
    x = node_pos[:, 0]
    y = node_pos[:, 1]

    # 计算φ值
    phi_values = x + y + phi_x * torch.sin(alpha_x * (np.pi * x) / L) + phi_y * torch.sin(alpha_y * (np.pi * y) / L) + \
                phi_xy * torch.cos(alpha_xy * (np.pi * x * y) / L**2)

    # 计算φ对x的一阶导数
    dphi_dx = torch.autograd.grad(phi_values, x, grad_outputs=torch.ones_like(phi_values), create_graph=True)[0]

    # 计算φ对y的一阶导数
    dphi_dy = torch.autograd.grad(phi_values, y, grad_outputs=torch.ones_like(phi_values), create_graph=True)[0]

    return phi_values.view(-1,1), dphi_dx.view(-1,1), dphi_dy.view(-1,1)


# CSV文件的路径
csv_file_path = f"{logger.saving_path}/L1_dphidx_Node.csv"

# 在循环开始前，打开CSV文件并写入表头
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 根据您的需求写入适当的表头
    writer.writerow(["Graph Index", "WLSQ-NX-WCX_cavity-tri"])

    params.accumulated_flag = False

    for i_graph in range(datasets_factory.dataset_size):
        
        datasets._set_plot_flag(_plot=True)
        (graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_cell_x,
        has_boundary,
        origin_mesh_path) = loader.get_specific_data([i_graph])

        integrator = fluid_model.integrator

        # calcualate phi node value
        phi_values_node, dphi_dx_node, dphi_dy_node = Scalar_Eular_solution(
            mesh_pos = graph_node.pos, 
            phi_0=1., 
            phi_x=1, 
            phi_y=1, 
            phi_xy=1, 
            alpha_x=1, 
            alpha_y=1, 
            alpha_xy=1, 
            L=15)
        
        phi_values_cell, dphi_dx_cell, dphi_dy_cell = Scalar_Eular_solution(
            mesh_pos = graph_cell.pos, 
            phi_0=1., 
            phi_x=1, 
            phi_y=1, 
            phi_xy=1, 
            alpha_x=1, 
            alpha_y=1, 
            alpha_xy=1, 
            L=15)
            
        # nabla_phi_node,nabla_phi_cell = integrator.compute_Green_Gauss_Gradient_node_based(
        #         phi_node=phi_values_node,
        #         graph_node=graph_node,
        #         graph_edge=graph_edge,
        #         graph_cell=graph_cell,
        #     )
    
        nabla_phi_node, nabla_phi_face, nabla_phi_cell = integrator.node_based_WLSQ(
            phi_node=phi_values_node,
            phi_cell=None,
            node_contribu=False,
            node_x_contribu=True,
            cell_to_node_contribu=False,
            graph_node_x=graph_node_x,
            graph_node=graph_node,
            graph_edge=graph_edge,
            graph_cell=graph_cell,
        )
        
        dphi_dnode_2nd, dphi_dnode_2nd_face, dphi_dnode_2nd_cell = integrator.node_based_WLSQ(
            phi_node=nabla_phi_node,
            phi_cell=None,
            node_contribu=False,
            node_x_contribu=True,
            cell_to_node_contribu=False,
            graph_node_x=graph_node_x,
            graph_node=graph_node,
            graph_edge=graph_edge,
            graph_cell=graph_cell,
        )
        
        nabla_phi_cell = integrator.interploate_node_to_cell_2nd_order(
                    node_phi=nabla_phi_node, 
                    nabla_node_phi=dphi_dnode_2nd, 
                    graph_node=graph_node, 
                    graph_cell=graph_cell
                )
        
        # phi_cell_1st = integrator.interploate_node_to_cell_2nd_order(
        #             node_phi=phi_values_node, 
        #             nabla_node_phi=nabla_phi_node, 
        #             graph_node=graph_node, 
        #             graph_cell=graph_cell
        #         )
        
        # nabla_phi_node, nabla_phi_face, nabla_phi_cell = integrator.node_based_WLSQ(
        #     phi_node=phi_values_node,
        #     phi_cell=phi_cell_1st,
        #     node_contribu=False,
        #     node_x_contribu=True,
        #     cell_to_node_contribu=True,
        #     graph_node_x=graph_node_x,
        #     graph_node=graph_node,
        #     graph_edge=graph_edge,
        #     graph_cell=graph_cell,
        # )
        
        # phi_cell = integrator.interploate_node_to_cell_2nd_order(
        #             node_phi=phi_values_node, 
        #             nabla_node_phi=nabla_phi_node, 
        #             graph_node=graph_node, 
        #             graph_cell=graph_cell
        #         )
        
        # # calcualate phi cell value
        # nabla_phi_node, _, nabla_phi_cell = integrator.cell_based_WLSQ(
        #         phi_node=phi_values_node,
        #         phi_cell=phi_values_cell,
        #         graph_node=graph_node,
        #         graph_edge=graph_edge,
        #         graph_cell=graph_cell,
        #         graph_cell_x=graph_cell_x,
        #         inteploting=True,
        #         pressure_only=False,
        #     )
        
        # L1_error_dphidx_node = torch.abs(nabla_phi_node[:,0:1] - dphi_dx_node).mean()
        L1_error_dphidx_cell = torch.abs(nabla_phi_cell[:,0:1] - dphi_dx_cell).mean()
        
        # write results to CSV file
        writer.writerow([graph_node.graph_index.item(), f"{L1_error_dphidx_cell.mean():.4e}"])
        
        collection_x = torch.cat((nabla_phi_node[:,0:1],dphi_dx_node,phi_values_node),dim=-1).detach()
        
        graph_node.x = collection_x
        
        list(map(datasets.payback_sp, Batch.to_data_list(graph_node)))

        # Using formatted string for better readability
        # headers = ["Graph Index", "Abs_error_dphidx_node", "Abs_error_on_phi_cell"]
        headers = ["Graph Index", "WLSQ-Nx-Ncx_cavity-quad"]
        values = [
            graph_node.graph_index.item(),
            f"{L1_error_dphidx_cell.mean():.4e}",
            # f"{Abs_error_on_phi_cell.mean():.4e}",
        ]

        # Determine the maximum width for each column
        column_widths = [max(len(str(x)), len(headers[i])) + 2 for i, x in enumerate(values)]

        # Create a format string for each row
        row_format = "".join(["{:<" + str(width) + "}" for width in column_widths])

        # Print the headers and values
        print(row_format.format(*headers))
        print(row_format.format(*values))
