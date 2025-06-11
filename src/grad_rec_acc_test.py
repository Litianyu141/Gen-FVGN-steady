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
from FVMmodel.FVdiscretization.FVgrad import weighted_lstsq,compute_normal_matrix,Moving_LSQ,weighted_lstsq_2nd_order
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

device = "cuda"
params.dataset_dir="datasets/trainset/airfoil_L=1/farfield_RAE2822_with_quad_bc_L=1"
params.dataset_size=1
params.batch_size=1
params.order = "2nd" # 1st, 2nd, 3rd, 4th

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
(graph_node,graph_cell_x,graph_edge,graph_cell,graph_Index) = next(iter(loader))
(
    graph_node,
    graph_cell_x,
    graph_edge,
    graph_cell,
    graph_Index,
) = (graph_node.to(device),
    graph_cell_x.to(device),
    graph_edge.to(device),
    graph_cell.to(device),
    graph_Index.to(device),
)
''' <<< fetch data and move to GPU <<< '''

# calcualate phi node value
phi_cell_GT, nabla_phi_GT, hessian_phi_GT = Scalar_Eular_solution(
    mesh_pos=graph_cell.cpd_centroid,
    phi_0=1.0,
    phi_x=0.01,
    phi_y=0.01,
    phi_xy=0.01,
    alpha_x=0.1,
    alpha_y=0.1,
    alpha_xy=0.1,
    L=1.0,
    device=device,
)

''' 验证使用预计算Moments方法是否正确 '''
(A_cell_to_cell, two_way_B_cell_to_cell) = compute_normal_matrix(
    order=params.order,
    mesh_pos=graph_cell.cpd_centroid,
    edge_index=graph_cell_x.neighbor_cell_x, # 默认应该是仅包含1阶邻居点+构成共点的单元的所有点
)

singleway_B = torch.chunk(two_way_B_cell_to_cell, 2, dim=0)[0]

grad_phi,cond_A = weighted_lstsq(
    phi_node=phi_cell_GT,
    edge_index=graph_cell_x.neighbor_cell_x, # 输入的时候一定得是单向的
    mesh_pos=graph_cell.cpd_centroid,
    order=params.order,
    precompute_Moments=[A_cell_to_cell,singleway_B],
    # precompute_Moments=None,
    rt_cond=True,
)
hessian_phi = torch.cat((grad_phi[:,0,2:3],grad_phi[:,0,4:5]),dim=1) # [N,[uxx,uxy]]
''' 验证使用预计算Moments方法是否正确 '''

''' 尝试使用两次WLSQ来重构2阶导数 '''
# hessian_phi,_ = weighted_lstsq(
#     phi_node=grad_phi[:,0,0:2],
#     edge_index=graph_cell_x.neighbor_cell_x, # 输入的时候一定得是单向的
#     mesh_pos=graph_node.pos,
#     order=params.order,
#     precompute_Moments=None,
#     # rt_cond=True,
# )  
# hessian_phi = torch.cat((hessian_phi[:,0,0:1],hessian_phi[:,0,1:2]),dim=1) # [N,[uxx,uxy]]
''' 尝试使用两次WLSQ来重构2阶导数 '''

''' 直接进行梯度重构 '''
# start_time = time.time()
# grad_phi = weighted_lstsq(
#     phi_node=phi_cell_GT,
#     edge_index=graph_cell_x.neighbor_cell_x, # 输入的时候一定得是单向的
#     mesh_pos=graph_node.pos,
#     order=params.order,
#     precompute_Moments=None,
# )  
# print(f"Grad Rec. Time consuming: {time.time()-start_time}")
# hessian_phi = torch.cat((grad_phi[:,0,2:3],grad_phi[:,0,4:5]),dim=1) # [N,[uxx,uxy]]

# grad_phi = weighted_lstsq_2nd_order(
#     phi_node=phi_cell_GT,
#     edge_index=graph_cell_x.neighbor_cell_x, # 输入的时候一定得是单向的
#     mesh_pos=graph_node.pos,
#     order=params.order,
#     precompute_Moments=None,
# )  
# # hessian_phi = torch.cat((grad_phi[:,0,2:3],grad_phi[:,0,4:5]),dim=1) # [N,[uxx,uxy]]
''' 直接进行梯度重构 '''

# calculate the relative L2 error
grad_relative_l2_error_1st = torch.norm(
    grad_phi[:, 0, 0:2] - nabla_phi_GT[:, 0:2], dim=0
) / torch.norm(nabla_phi_GT[:, 0:2], dim=0)
MSE = torch.mean((grad_phi[:, 0, 0:2] - nabla_phi_GT[:, 0:2])**2)
print(f"Gradient Relative L2 error: {grad_relative_l2_error_1st}")
print(f"Gradient MSE: {MSE}")

# calculate the relative L2 error
grad_relative_l2_error_1st = torch.norm(
    hessian_phi - hessian_phi_GT[:, 0, :], dim=0
) / torch.norm(hessian_phi_GT[:, 0, :], dim=0)
MSE = torch.mean((hessian_phi - hessian_phi_GT[:, 0, :])**2)
print(f"Hessian relative L2 error: {grad_relative_l2_error_1st}")
print(f"Hessian MSE: {MSE}")

cell_type = graph_cell.cell_type
interior_cell_mask = (cell_type== NodeType.NORMAL)

''' 将梯度保存到vtu文件 '''
father_dir = os.path.dirname(logger.saving_path)
for _ in range(1):
    father_dir = os.path.dirname(father_dir)
case_name = "".join(
    chr(code) for code in graph_cell.case_name.cpu().tolist()
)
os.makedirs(f"{father_dir}/Grad_test", exist_ok=True)

cells_node = graph_node.face
cells_face = graph_edge.face
cells_index = graph_cell.cells_index
domain_list = seperate_domain(cells_node, cells_face, cells_index)

pv_cells_node=[]
pv_cells_type=[]
pv_cells_index=[]
for domain in domain_list:
    
    _ct, _cells_node, _, _cells_index, _ = domain
    pv_cells_index.append(_cells_index)
    
    _cells_node = _cells_node.reshape(-1,_ct)
    num_cells = _cells_node.shape[0]
    _cells_node = torch.cat(
        (torch.full((_cells_node.shape[0],1),_ct,device=device), _cells_node),
        dim=1,
    ).reshape(-1)
    pv_cells_node.append(_cells_node)
    
    # 根据顶点数设置单元类型（3=三角形, 4=四边形, >4=多边形）
    if _ct == 3:
        cells_types = torch.full((num_cells,),pv.CellType.TRIANGLE)
    elif _ct == 4:
        cells_types = torch.full((num_cells,),pv.CellType.QUAD)
    else:
        cells_types = torch.full((num_cells,),pv.CellType.POLYGON)
    pv_cells_type.append(cells_types)
    
pv_cells_node = torch.cat(pv_cells_node,dim=0) 
pv_cells_type = torch.cat(pv_cells_type,dim=0)
pv_cells_index = torch.cat(pv_cells_index,dim=0)

def unique_preserve_order(x: torch.Tensor) -> torch.Tensor:
    """
    等价于 NumPy 的 `np.unique(x, return_index=True)[0][np.argsort(idx)]`，
    但全部在 PyTorch 内部完成，支持 CUDA。
    """
    # 1) 先做一次 unique 拿到 inverse，下步用来找“第一次出现在哪儿”
    uniq, inv = torch.unique(x, return_inverse=True)          # uniq 仍然是升序的
    # 2) 对 inv 做 scatter-amin 得到每个唯一值第一次出现的下标
    first_occ = torch.full_like(uniq, x.numel(), dtype=torch.long)
    first_occ.scatter_reduce_(0, inv, torch.arange(x.numel(), device=x.device),
                              reduce="amin")                  # PyTorch ≥1.12
    # 3) 根据 first_occ 升序重新排列 uniq
    order = torch.argsort(first_occ, stable=True)             # stable=True 保持同下标稳定
    return uniq[order]

pv_cells_index = unique_preserve_order(pv_cells_index)

write_hybrid_mesh_to_vtu_2D(
    mesh_pos=graph_node.pos.cpu(), 
    data={
        "cell|phi_cell_GT":phi_cell_GT[interior_cell_mask][pv_cells_index].cpu().numpy(),
        "cell|grad_phi_GT":nabla_phi_GT[interior_cell_mask][pv_cells_index].cpu().numpy(),
        "cell|grad_phi":grad_phi[interior_cell_mask,0,0:2][pv_cells_index].cpu().numpy(),
        "cell|hessian_phi_GT":hessian_phi_GT[interior_cell_mask,0,:][pv_cells_index].cpu().numpy(),
        "cell|hessian_phi":hessian_phi[interior_cell_mask][pv_cells_index].cpu().numpy(),
        "cell|cond_A":cond_A[interior_cell_mask][pv_cells_index].squeeze().cpu().numpy(),
    }, 
    cells_node=pv_cells_node.cpu().numpy(),
    cells_type=pv_cells_type.cpu().numpy(), 
    filename=f"{father_dir}/Grad_test/grad_{case_name}_{params.order}.vtu"
)
''' 将梯度保存到vtu文件 '''

''' >>> test to tecplot >>> '''
face_type = graph_edge.face_type
BC_face_mask = ~(face_type== NodeType.NORMAL)
neighbor_cell = graph_cell.edge_index.clone()
neighbor_cell[1,BC_face_mask] = neighbor_cell[0,BC_face_mask]

interior_zone = {"name": "EularGradTest", "rho": 1, "mu": 1, "dt": 1}
interior_zone["node|X"] = graph_node.pos[:, 0:1].cpu().unsqueeze(0).numpy()
interior_zone["node|Y"] = graph_node.pos[:, 1:2].cpu().unsqueeze(0).numpy()
interior_zone["cell|phi_cell_GT"] = phi_cell_GT[None,interior_cell_mask,0:1].cpu().numpy()
interior_zone["cell|nabla_phi_GT"] = nabla_phi_GT[None,interior_cell_mask,0:1].cpu().numpy()
interior_zone["cell|grad_phi"] = grad_phi[None,interior_cell_mask,0,0:1].cpu().numpy()
interior_zone["cells_node"] = graph_node.face.unsqueeze(0).cpu().numpy()
interior_zone["cells_index"] = graph_cell.cells_index.unsqueeze(0).cpu().numpy()
interior_zone["face_node"] = graph_node.edge_index.cpu().transpose(0, 1).unsqueeze(0).numpy()
interior_zone["neighbor_cell"] = neighbor_cell.cpu().transpose(0, 1).unsqueeze(0).numpy()

write_zone = [interior_zone, None]

write_tecplotzone(
    f"{father_dir}/Grad_test/grad_{case_name}_{params.order}.dat",
    datasets=write_zone,
    time_step_length=1,
)
''' <<< test to tecplot <<< '''