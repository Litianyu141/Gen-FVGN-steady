import numpy as np
from torch_geometric.data import Data
import enum
import torch
from torch_scatter import scatter

class NodeType(enum.IntEnum):
    NORMAL = 0
    INFLOW = 1
    OUTFLOW = 2
    WALL_BOUNDARY = 3
    PRESS_POINT = 4
    IN_WALL = 5


def calc_cell_centered_with_node_attr(
    node_attr, cells_node, cells_index, reduce="mean", map=True
):
    if cells_node.shape != cells_index.shape:
        raise ValueError("wrong cells_node/cells_index dim")

    if len(cells_node.shape) > 1:
        cells_node = cells_node.view(-1)

    if len(cells_index.shape) > 1:
        cells_index = cells_index.view(-1)

    if map:
        mapped_node_attr = node_attr[cells_node]
    else:
        mapped_node_attr = node_attr

    cell_attr = scatter(src=mapped_node_attr, index=cells_index, dim=0, reduce=reduce)

    return cell_attr


def calc_node_centered_with_cell_attr(
    cell_attr, cells_node, cells_index, reduce="mean", map=True
):
    if cells_node.shape != cells_index.shape:
        raise ValueError(f"wrong cells_node/cells_index dim ")

    if len(cells_node.shape) > 1:
        cells_node = cells_node.view(-1)

    if len(cells_index.shape) > 1:
        cells_index = cells_index.view(-1)

    if map:
        maped_cell_attr = cell_attr[cells_index]
    else:
        maped_cell_attr = cell_attr

    cell_attr = scatter(src=maped_cell_attr, index=cells_node, dim=0, reduce=reduce)

    return cell_attr


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_and_trans_node_attr_to_cell_attr_graph(
    graph, has_changed_node_attr_to_cell_attr
):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, face, global_attr, mask_cell_interior = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    for key in graph.keys():
        if key == "x":
            x = graph.x  # avoid exception
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        elif key == "face":
            face = graph.face
        elif key == "mask_cell_interior":
            mask_cell_interior = graph.mask_cell_interior
        else:
            pass

    return (x, edge_index, edge_attr, face, global_attr, mask_cell_interior)


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph, has_changed_node_attr_to_cell_attr):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    (
        node_attr,
        edge_index,
        edge_attr,
        face,
        global_attr,
        mask_cell_interior,
    ) = decompose_and_trans_node_attr_to_cell_attr_graph(
        graph, has_changed_node_attr_to_cell_attr
    )

    ret = Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        face=face,
        mask_cell_interior=mask_cell_interior,
    )

    ret.global_attr = global_attr

    return ret


def shuffle_np(array):
    array_t = array.copy()
    np.random.shuffle(array_t)
    return array_t


def generate_boundary_zone(
    dataset=None, surf_mask=None, rho=None, mu=None, dt=None
):
    face_node = dataset["face|face_node"].long()
    if face_node.shape[0] > face_node.shape[1]:
        face_node = face_node.mT

    mesh_pos = dataset["node|pos"].to(torch.float32)
    
    surf_face_index,surf_face_mask = filter_adj(
        face_node.numpy(), 
        perm=torch.arange(mesh_pos.shape[0]).numpy(), 
        num_nodes=mesh_pos.shape[0]
    )
    surf_face_index = torch.from_numpy(surf_face_index)
    surf_face_mask = torch.from_numpy(surf_face_mask)
    
    boundary_zone = {"name": "OBSTACLE", "rho": rho, "mu": mu, "dt": dt}
    boundary_zone["zonename"] = "OBSTICALE_BOUNDARY"
    
    boundary_zone["node|surf_mask"] = surf_mask
    boundary_zone["face|surf_face_mask"] = surf_face_mask
    
    boundary_zone["face|face_node"] = surf_face_index
    boundary_zone["node|mesh_pos"] = mesh_pos[surf_mask]

    return boundary_zone


def filter_adj(edge_index, perm, num_nodes):
    # 初始化 mask 数组，默认值为 -1
    mask = np.full((num_nodes,), -1, dtype=int)
    
    # 根据 perm 更新 mask 的有效位置
    mask[perm] = np.arange(perm.size)
    
    # 提取 row 和 col
    row, col = edge_index[0], edge_index[1]
    
    # 更新 row 和 col 的索引，使用 mask 过滤无效节点
    row, col = mask[row], mask[col]
    
    # 创建一个布尔掩码来筛选有效边（row 和 col >= 0）
    valid_mask = (row >= 0) & (col >= 0)
    row, col = row[valid_mask], col[valid_mask]

    # 将 row 和 col 堆叠为一个 (2, N) 形状的数组
    return np.stack([row, col], axis=0), valid_mask

# Scalar Eular solution
def Scalar_Eular_solution(
    mesh_pos, phi_0, phi_x, phi_y, phi_xy, alpha_x, alpha_y, alpha_xy, L, device="cpu"
):
    """
    使用PyTorch计算给定网格点上的φ值及其一阶导数和二阶导数（Hessian矩阵）。

    参数:
    mesh_pos: 一个形状为(100, 2)的张量，表示100个网格点的x和y坐标。
    phi_0, phi_x, phi_y, phi_xy: 公式中的系数。
    alpha_x, alpha_y, alpha_xy: 公式中的alpha参数。
    L: 域的长度。

    返回:
    phi_values: 一个张量，包含每个网格点上的φ值，形状为(N, 1)。
    nabla_phi: 一个张量，包含每个网格点上的φ对x和y的一阶导数，形状为(N, 2)。
    hessian_phi: 一个张量，包含每个网格点上的φ对x和y的二阶导数（Hessian矩阵），形状为(N, 2, 2)。
    """
    # 检查输入参数类型
    if not isinstance(mesh_pos, torch.Tensor):
        raise TypeError("mesh_pos must be a torch.Tensor")
    if not mesh_pos.dtype == torch.float32:
        raise TypeError("mesh_pos must be of type torch.float32")

    # 确保 mesh_pos 需要计算梯度
    node_pos = mesh_pos.clone().to(device)
    node_pos.requires_grad_(True)

    # 提取 x 和 y 坐标
    x = node_pos[:, 0]
    y = node_pos[:, 1]

    # 计算 φ 值
    phi_values = (
        phi_0
        + phi_x * torch.sin(alpha_x * np.pi * x / L)
        + phi_y * torch.sin(alpha_y * np.pi * y / L)
        + phi_xy * torch.cos(alpha_xy * np.pi * x * y / L**2)
    )

    # 计算 φ 对 x 和 y 的一阶导数
    dphi_dx = torch.autograd.grad(
        phi_values, x, grad_outputs=torch.ones_like(phi_values), create_graph=True
    )[0]
    dphi_dy = torch.autograd.grad(
        phi_values, y, grad_outputs=torch.ones_like(phi_values), create_graph=True
    )[0]

    # 将一阶导数合并为 nabla_phi 张量
    nabla_phi = torch.stack([dphi_dx, dphi_dy], dim=1)

    # 计算 φ 对 x 的二阶导数（即 Hessian 矩阵的 [0, 0] 分量）
    d2phi_dx2 = torch.autograd.grad(
        dphi_dx, x, grad_outputs=torch.ones_like(dphi_dx), create_graph=True
    )[0]

    # 计算 φ 对 y 的二阶导数（即 Hessian 矩阵的 [1, 1] 分量）
    d2phi_dy2 = torch.autograd.grad(
        dphi_dy, y, grad_outputs=torch.ones_like(dphi_dy), create_graph=True
    )[0]

    # 计算 φ 对 x 和 y 的混合二阶导数（即 Hessian 矩阵的 [0, 1] 和 [1, 0] 分量）
    d2phi_dxdy = torch.autograd.grad(
        dphi_dx, y, grad_outputs=torch.ones_like(dphi_dx), create_graph=True
    )[0]

    d2phi_dydx = torch.autograd.grad(
        dphi_dy, x, grad_outputs=torch.ones_like(dphi_dy), create_graph=True
    )[0]

    # 组装 Hessian 矩阵
    hessian_phi = torch.stack(
        [
            torch.stack([d2phi_dx2, d2phi_dxdy], dim=1),
            torch.stack([d2phi_dydx, d2phi_dy2], dim=1),
        ],
        dim=1,
    )

    # 返回 φ, nabla_phi 和 hessian_phi
    return phi_values.view(-1, 1).detach(), nabla_phi.detach(), hessian_phi.detach()
