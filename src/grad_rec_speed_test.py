import sys
import os

cur_path = os.path.split(__file__)[0]
sys.path.append(cur_path)

import torch
import numpy as np

# import os
from Load_mesh import Graph_loader
from Utils import get_param
import time
from Utils.get_param import get_hyperparam
from Utils.Logger import Logger
from Post_process.to_vtk import write_hybrid_mesh_to_vtu_2D
from Post_process.to_tecplot import write_tecplotzone
from torch_geometric.data.batch import Batch
from Utils.utilities import Scalar_Eular_solution
import random
import datetime
from FVMmodel.FVdiscretization.FVgrad import node_based_WLSQ,compute_normal_matrix,Moving_LSQ,node_based_WLSQ_2nd_order
from Extract_mesh.parse_to_h5 import seperate_domain,build_k_hop_edge_index
import pyvista as pv
from Utils.utilities import NodeType

# configurate parameters
params = get_param.params()
seed = int(datetime.datetime.now().timestamp())
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(0.8, params.on_gpu)
torch.set_float32_matmul_precision('high')

device = "cuda"
params.dataset_dir="datasets/Grad_test/lid_driven_cavity_101x101"
params.dataset_size=1
params.batch_size=1
params.order = "2nd" # 1st, 2nd, 3rd, 4th

# 编译 node_based_WLSQ 函数 - 只编译主要测试函数
compiled_node_based_WLSQ = torch.compile(node_based_WLSQ)
# 不编译 compute_normal_matrix，因为它包含可能导致图断裂的操作

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

''' >>> fetch data and move to GPU >>> '''
for batch_index, (
    graph_node,
    graph_node_x,
    graph_edge,
    graph_cell,
    graph_Index,
) in enumerate(loader):
    
    case_name = "".join(
        chr(code) for code in graph_node.case_name.tolist()
    )
    
    (
        graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_Index,
    ) = datasets.datapreprocessing(
        graph_node=graph_node.to(device),
        graph_node_x=graph_node_x.to(device),
        graph_edge=graph_edge.to(device),
        graph_cell=graph_cell.to(device),
        graph_Index=graph_Index.to(device),
    )
    ''' <<< fetch data and move to GPU <<< '''
        
    # calcualate phi node value
    phi_node_GT, nabla_phi_GT, hessian_phi_GT = Scalar_Eular_solution(
        mesh_pos=graph_node.pos,
        phi_0=1.0,
        phi_x=0.01,
        phi_y=0.01,
        phi_xy=0.01,
        alpha_x=5,
        alpha_y=5,
        alpha_xy=5,
        L=1.0,
        device=device,
    )

    with torch.no_grad():
        ''' >>> 预计算moments >>> '''
        (A_node_to_node, two_way_B_node_to_node, extra_B_node_to_node) = compute_normal_matrix(
            order=params.order,
            mesh_pos=graph_node.pos,
            edge_index=graph_node_x.face_node_x,
            extra_edge_index=graph_node_x.support_edge
        )
        single_way_B = torch.chunk(two_way_B_node_to_node, 2, dim=0)[0]
        
        # 先运行一次编译版本进行预热，并检查输出
        grad_phi_warmup = compiled_node_based_WLSQ(
            phi_node=phi_node_GT,
            edge_index=graph_node_x.face_node_x,
            extra_edge_index=graph_node_x.support_edge,
            mesh_pos=graph_node.pos,
            order=params.order,
            precompute_Moments=[A_node_to_node, single_way_B, extra_B_node_to_node],
            rt_cond=False,
        )
        print(f"Warmup gradient shape: {grad_phi_warmup.shape}")
        print(f"Warmup gradient contains NaN: {torch.isnan(grad_phi_warmup).any()}")
        ''' <<< 预计算moments <<< '''

        ''' >>> Perform gradient reconstruction 50000 times and calculate average time >>> '''
        total_time = 0.0
        num_runs = 50000
        for _ in range(num_runs):
            start_time = time.time()
            grad_phi = compiled_node_based_WLSQ(
                phi_node=phi_node_GT,
                edge_index=graph_node_x.face_node_x,
                extra_edge_index=graph_node_x.support_edge,
                mesh_pos=graph_node.pos,
                order=params.order,
                precompute_Moments=[A_node_to_node, single_way_B, extra_B_node_to_node],
                rt_cond=False,
            )  
            total_time += time.time() - start_time

        average_time = total_time / num_runs
        print(f"{case_name} Average Grad Rec. Time over {num_runs} runs: {average_time}")
        ''' <<< Perform gradient reconstruction 100 times and calculate average time <<< '''
    
# Calculate the relative L2 error
grad_relative_l2_error_1st = torch.norm(
    grad_phi[:, 0, 0:2] - nabla_phi_GT[:, 0:2], dim=0
) / torch.norm(nabla_phi_GT[:, 0:2], dim=0)
MSE = torch.mean((grad_phi[:, 0, 0:2] - nabla_phi_GT[:, 0:2]) ** 2)
print(f"Gradient Relative L2 error: {grad_relative_l2_error_1st}")
print(f"Gradient MSE: {MSE}")
