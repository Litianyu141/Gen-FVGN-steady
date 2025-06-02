# 2 -*- encoding: utf-8 -*-
"""
@File    :   parse_tfrecord.py
@Author  :   litianyu 
@Version :   2.0
@Contact :   lty1040808318@163.com
"""
import sys
import os

file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)
import numpy as np
import torch
from torch_scatter import scatter
from Utils.utilities import NodeType
from Post_process.to_vtk import write_point_cloud_to_vtk
from torch_geometric import utils as pyg_utils

def find_pos(mesh_point, mesh_pos_sp1):
    for k in range(mesh_pos_sp1.shape[0]):
        if (mesh_pos_sp1[k] == mesh_point).all():
            print("found{}".format(k))
            return k
    return False


def convert_to_tensors(input_dict):
    # 遍历字典中的所有键
    for key in input_dict.keys():
        # 检查值的类型
        value = input_dict[key]
        if isinstance(value, np.ndarray):
            # 如果值是一个Numpy数组，使用torch.from_numpy进行转换
            input_dict[key] = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            # 如果值不是一个PyTorch张量，使用torch.tensor进行转换
            input_dict[key] = torch.tensor(value)
        # 如果值已经是一个PyTorch张量，不进行任何操作

    # 返回已更新的字典
    return input_dict


def polygon_area(vertices):
    """
    使用shoelace formula（鞋带公式）来计算多边形的面积。
    :param vertices: 多边形的顶点坐标，一个二维numpy数组。
    :return: 多边形的面积。
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def sort_vertices_ccw(mesh_pos, face_center, cells_node, cells_face, cells_index, centroid):
    """
    使用中心点对顶点和面进行逆时针排序（完全向量化版本）
    
    参数:
    mesh_pos: 顶点坐标张量，维度=[N,2] (对于cells_node排序时使用)
    face_center: 面中心坐标张量，维度=[M,2] (对于cells_face排序时使用)
    cells_node: 构成多边形的顶点序号张量
    cells_face: 构成多边形的面序号张量
    cells_index: 每个顶点/面所属多边形的索引张量
    centroid: 多边形中心点坐标张量，维度=[C,2]
    
    返回:
    按逆时针排序后的cells_node, cells_face, cells_index
    """
    # 使用seperate_domain分离不同类型的单元
    domain_list = seperate_domain(cells_node, cells_face, cells_index)
    
    # 初始化
    new_cells_node = []
    new_cells_face = []
    re_cells_index = []
    
    # 对每种类型的单元分别处理
    for ct, sub_cells_node, sub_cells_face, sub_cells_index, original_indices in domain_list:
        if sub_cells_node.size(0) == 0:
            continue
        
        re_cells_index.append(sub_cells_index)
        
        # 将sub_cells_node和sub_cells_face reshape为[num_cells, ct]
        num_cells = sub_cells_node.size(0) // ct
        cells_node_2d = sub_cells_node.reshape(num_cells, ct)
        cells_face_2d = sub_cells_face.reshape(num_cells, ct)
        
        # 获取每个单元的质心 - 使用传入的centroid
        cell_centroids = centroid[sub_cells_index.reshape(num_cells, ct)[:, 0]]
        
        # 对cells_node进行排序：使用顶点坐标计算角度
        vertices_coords = mesh_pos[cells_node_2d]  # [num_cells, ct, 2]
        relative_coords_vertices = vertices_coords - cell_centroids.unsqueeze(1)
        angles_vertices = torch.atan2(relative_coords_vertices[:, :, 1], relative_coords_vertices[:, :, 0])
        sorted_indices_vertices = torch.argsort(angles_vertices, dim=1)
        sorted_cells_node_2d = torch.gather(cells_node_2d, 1, sorted_indices_vertices)
        
        # 对cells_face进行单独排序：使用面中心坐标计算角度
        face_coords = face_center[cells_face_2d]  # [num_cells, ct, 2]
        relative_coords_faces = face_coords - cell_centroids.unsqueeze(1)
        angles_faces = torch.atan2(relative_coords_faces[:, :, 1], relative_coords_faces[:, :, 0])
        sorted_indices_faces = torch.argsort(angles_faces, dim=1)
        sorted_cells_face_2d = torch.gather(cells_face_2d, 1, sorted_indices_faces)
        
        new_cells_node.append(sorted_cells_node_2d.flatten())
        new_cells_face.append(sorted_cells_face_2d.flatten())
        
    return torch.cat(new_cells_node, dim=0), torch.cat(new_cells_face, dim=0), torch.cat(re_cells_index, dim=0)

def find_max_distance(points):
    # 获取点的数量
    n_points = points.size(0)

    # 初始化最大距离为0
    max_distance = 0

    # 遍历每一对点
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # 计算两点之间的欧几里得距离
            distance = torch.norm(points[i] - points[j])

            # 更新最大距离
            max_distance = max(max_distance, distance)

    # 返回最大距离
    return max_distance


def compose_support_face_node_x(cells_type, cells_node):
    """
    Composes the unique connections between nodes that form the faces of each cell.

    Parameters:
    - cells_type (int): The number of nodes per cell (e.g., 3 for triangles, 4 for quadrilaterals).
    - cells_node (torch.Tensor): Tensor containing the indices of nodes for each cell, flattened.

    Returns:
    - torch.Tensor: A tensor of shape [2, num_faces], where each column represents a unique face defined by two node indices.
    """
    
    face_node_x=[]
    origin_cells_node = cells_node.clone()
    for _ in range(cells_type-1):
        cells_node = torch.roll(cells_node.reshape(-1,cells_type), shifts=1, dims=1).reshape(-1)
        face_node_x.append(torch.stack((origin_cells_node, cells_node), dim=0))
    face_node_x = torch.cat(face_node_x, dim=1)
    return torch.unique(face_node_x[:,~(face_node_x[0]==face_node_x[1])].sort(dim=0)[0],dim=1)

def compose_support_edge_to_node(cells_type, cells_face, cells_node, offset=None):
    """
    Constructs the mapping between faces and nodes, indicating which nodes belong to each face.

    Parameters:
    - cells_type (int): The number of nodes per cell.
    - cells_face (torch.Tensor): Tensor containing the indices of faces for each cell.
    - cells_node (torch.Tensor): Tensor containing the indices of nodes for each face.
    - offset (int, optional): An optional offset to be added to the face indices.

    Returns:
    - torch.Tensor: A tensor of shape [2, num_edges], representing the unique connections between faces and nodes.
    """
    if offset is not None:
        cells_face += offset
        
    support_edge_to_node=[]
    for _ in range(cells_type):
        support_edge_to_node.append(torch.stack((cells_face, cells_node), dim=0))
        cells_node = torch.roll(cells_node.reshape(-1,cells_type), shifts=1, dims=1).reshape(-1)
    return torch.unique(torch.cat(support_edge_to_node, dim=1).sort(dim=0)[0],dim=1)

def compose_support_cell_to_node(cells_type, cells_index, cells_node, offset=None):
    """
    Constructs the mapping between cells and nodes, indicating which nodes belong to each cell.

    Parameters:
    - cells_type (int): The number of nodes per cell.
    - cells_index (torch.Tensor): Tensor containing the indices of cells.
    - cells_node (torch.Tensor): Tensor containing the indices of nodes for each cell.
    - offset (int, optional): An optional offset to be added to the cell indices.

    Returns:
    - torch.Tensor: A tensor of shape [2, num_edges], representing the unique connections between cells and nodes.
    """
    if offset is not None:
        cells_index += offset
        
    support_cell_to_node=[]
    for _ in range(cells_type):
        support_cell_to_node.append(torch.stack((cells_index, cells_node), dim=0))
        cells_node = torch.roll(cells_node.reshape(-1,cells_type), shifts=1, dims=1).reshape(-1)
    return torch.unique(torch.cat(support_cell_to_node, dim=1).sort(dim=0)[0],dim=1)

def seperate_domain(cells_node, cells_face, cells_index):
    """
    Separates the domain into different regions based on cell types (e.g., triangular, quadrilateral and polygons).

    Parameters:
    - cells_node (torch.Tensor): Tensor containing the node indices for each cell.
    - cells_face (torch.Tensor): Tensor containing the face indices for each cell.
    - cells_index (torch.Tensor): Tensor containing the cell indices.

    Returns:
    - list: A list of tuples, each containing:
        - ct (int): The cell type (number of nodes per cell).
        - cells_node_sub (torch.Tensor): Subset of cells_node for the cell type.
        - cells_face_sub (torch.Tensor): Subset of cells_face for the cell type.
        - cells_index_sub (torch.Tensor): Subset of cells_index for the cell type.
        - original_indices (torch.Tensor): 原始位置索引，用于恢复顺序
    """
    cells_type_ex = scatter(src=torch.ones_like(cells_index), 
        index=cells_index, 
        dim=0, 
    )
    
    cells_type = torch.unique(cells_type_ex, dim=0)
    
    domain_list = []
    for ct in cells_type:
        mask = (cells_type_ex==ct)[cells_index]
        original_indices = torch.where(mask)[0]  # 记录原始位置
        domain_list.append((ct, cells_node[mask], cells_face[mask], cells_index[mask], original_indices))
        
    return domain_list

def build_k_hop_edge_index(edge_index, k):
    """
    用PyG的to_torch_coo_tensor和稀疏矩阵运算计算k跳邻居的连接关系。

    Parameters:
    - mesh_pos: [N, 2] 每个节点的坐标。
    - edge_index: [2, E] 原始的边索引，两个节点之间的连通关系。
    - num_nodes: 节点总数 N。
    - k: 跳数，表示距离多少跳的邻居。

    Returns:
    - new_edge_index: 新的边索引数组, 包含距离当前节点k跳以外的邻居的连通关系。
    """
    # 将edge_index转换为稀疏矩阵 (COO 格式)
    sparse_adj = pyg_utils.to_torch_coo_tensor(edge_index)

    # 初始化邻接矩阵为一跳邻居
    adj_k = sparse_adj

    # 进行k-1次邻接矩阵自乘，得到k跳邻居
    for _ in range(k - 1):
        adj_k = torch.sparse.mm(adj_k, sparse_adj)

    # 从稀疏矩阵中提取新的edge_index (两跳或k跳邻居)
    new_edge_index = adj_k.coalesce().indices()

    return new_edge_index


def extract_mesh_state(
    dataset,
    path=None,
):
    """
    face_center_pos, centroid, face_type, neighbour_cell, face_node_x
    """
    dataset = convert_to_tensors(dataset)

    """>>> prepare for converting >>>"""
    mesh_pos = dataset["node|pos"]
    node_type = dataset["node|node_type"]
    face_node = dataset["face|face_node"]
    cells_node = dataset["cells_node"]
    cells_index = dataset["cells_index"]
    cells_face = dataset["cells_face"]
    """<<< prepare for converting <<<"""

    """>>> compute centroid crds >>>"""
    centroid = scatter(
        src=mesh_pos[cells_node],
        index=cells_index,
        dim=0,
        reduce="mean",
    )
    dataset["cell|centroid"] = centroid
    """<<< compute centroid crds <<<"""

    """ >>> compute face_center_pos >>> """
    face_center_pos = (mesh_pos[face_node[0]] + mesh_pos[face_node[1]]) / 2.0
    dataset["face|face_center_pos"] = face_center_pos
    """ <<<   compute face_center_pos   <<< """

    """>>> ensure cells node and face counterclockwise >>>"""
    cells_node_ccw, cells_face_ccw, cells_index = sort_vertices_ccw(
        dataset["node|pos"],
        dataset["face|face_center_pos"],
        dataset["cells_node"],
        dataset["cells_face"],
        dataset["cells_index"],
        dataset["cell|centroid"],
    )
    dataset["cells_node"] = cells_node_ccw
    dataset["cells_face"] = cells_face_ccw
    dataset["cells_index"] = cells_index
    cells_node = dataset["cells_node"]
    cells_face = dataset["cells_face"]
    """<<< ensure cells node and face counterclockwise <<<"""    
    
    """ >>>   assign face type   >>>"""
    face_type = torch.full((face_node.shape[1],),NodeType.NORMAL).long()
    left_node_type,right_node_type = node_type[face_node[0]],node_type[face_node[1]]
    mask_inflow_face =  \
        (
            ((left_node_type==NodeType.INFLOW)|\
            (left_node_type==NodeType.WALL_BOUNDARY)|\
            (left_node_type==NodeType.OUTFLOW)|\
            (left_node_type==NodeType.PRESS_POINT)|\
            (left_node_type==NodeType.IN_WALL)) \
            & \
            (right_node_type==NodeType.INFLOW)
        )|\
        (
            ((right_node_type==NodeType.INFLOW)|\
            (right_node_type==NodeType.WALL_BOUNDARY)|\
            (right_node_type==NodeType.OUTFLOW)|\
            (right_node_type==NodeType.PRESS_POINT)|\
            (right_node_type==NodeType.IN_WALL)) \
            & \
            (left_node_type==NodeType.INFLOW)
        )
    face_type[mask_inflow_face]=NodeType.INFLOW
    
    mask_wall_face =  \
        (
            ((left_node_type==NodeType.WALL_BOUNDARY)|\
            (left_node_type==NodeType.INFLOW)|\
            (left_node_type==NodeType.OUTFLOW)|\
            (left_node_type==NodeType.PRESS_POINT)|\
            (left_node_type==NodeType.IN_WALL)) \
            & \
            (right_node_type==NodeType.WALL_BOUNDARY)
        )|\
        (
            ((right_node_type==NodeType.WALL_BOUNDARY)|\
            (right_node_type==NodeType.IN_WALL)|\
            (right_node_type==NodeType.OUTFLOW)|\
            (right_node_type==NodeType.PRESS_POINT)|\
            (right_node_type==NodeType.IN_WALL)) \
            & \
            (left_node_type==NodeType.WALL_BOUNDARY)
        )
    face_type[mask_wall_face]=NodeType.WALL_BOUNDARY
    
    mask_outflow_face =  \
        (
            ((left_node_type==NodeType.WALL_BOUNDARY)|\
            (left_node_type==NodeType.INFLOW)|\
            (left_node_type==NodeType.OUTFLOW)|\
            (left_node_type==NodeType.PRESS_POINT)|\
            (left_node_type==NodeType.IN_WALL)) \
            & \
            (right_node_type==NodeType.OUTFLOW)
        )|\
        (
            ((right_node_type==NodeType.WALL_BOUNDARY)|\
            (right_node_type==NodeType.IN_WALL)|\
            (right_node_type==NodeType.OUTFLOW)|\
            (right_node_type==NodeType.PRESS_POINT)|\
            (right_node_type==NodeType.IN_WALL)) \
            & \
            (left_node_type==NodeType.OUTFLOW)
        )
    face_type[mask_outflow_face]=NodeType.OUTFLOW
    dataset["face|face_type"] = face_type
    threeD_pos = torch.cat((face_center_pos,torch.zeros_like(face_center_pos[:,0:1])),dim=1)
    data_dict = {"node|pos":threeD_pos.numpy(),
                 "node|face_type":face_type.float().numpy(),}
    write_point_cloud_to_vtk(data_dict,f"{path['file_dir']}/face_type_in_scatter.vtu")
    """ <<<   assign face type   <<<"""

    """ >>>   compute face area   >>>"""
    face_area = torch.norm(
        (mesh_pos[face_node[0]] - mesh_pos[face_node[1]]), dim=1, keepdim=True
    )
    dataset["face|face_area"] = face_area
    """ <<<   compute face area   <<<"""

    """ >>> compute neighbor_cell >>> """
    # validation
    senders_cell = scatter(
        src = cells_index[:,None],
        index = cells_face,
        dim = 0,
        reduce = "max"
    ).squeeze(1)

    recivers_cell = scatter(
        src = cells_index[:,None],
        index = cells_face,
        dim = 0,
        reduce = "min"
    ).squeeze(1)
    
    neighbour_cell = torch.stack((recivers_cell, senders_cell), dim=0)
    dataset["face|neighbour_cell"] = neighbour_cell.to(torch.int64)
    """ <<< compute neighbor_cell <<< """

    """ >>> unit normal vector >>> """
    face_area = face_area
    senders_node, recivers_node = face_node[0], face_node[1]
    mesh_pos = mesh_pos
    pos_diff = mesh_pos[senders_node] - mesh_pos[recivers_node]
    unv = torch.cat((-pos_diff[:, 1:2], pos_diff[:, 0:1]), dim=1)
    unv = unv / (torch.norm(unv, dim=1, keepdim=True))

    if not torch.isfinite(unv).all():
        raise ValueError(f'unv Error mesh_file {path["file_dir"]}')

    face_to_centroid = (
        face_center_pos[cells_face.view(-1)] - centroid[cells_index.view(-1)]
    )
    cells_face_unv = unv[cells_face.view(-1)]
    unv_dir_mask = (
        torch.sum(face_to_centroid * cells_face_unv, dim=1, keepdim=True) > 0.0
    ).repeat(1, 2)
    cells_face_unv = torch.where(
        unv_dir_mask, cells_face_unv, (-1.0) * cells_face_unv
    )

    cells_face_area = face_area[cells_face.view(-1)]
    surface_vector = cells_face_unv * cells_face_area

    valid = scatter(
        src=surface_vector,
        index=cells_index,
        reduce="sum",
        dim=0,
    )
    
    if not torch.allclose(valid, torch.zeros_like(valid), rtol=1e-05, atol=1e-08, equal_nan=False):
        raise ValueError(f"wrong unv calculation {path['file_dir']}")
    dataset["unit_norm_v"] = cells_face_unv
    """ <<< unit normal vector <<< """

    # compute cells_area
    cells_face = cells_face
    face_area = face_area

    surface_vector = surface_vector
    full_synataic_function = 0.5 * face_center_pos[cells_face.view(-1)]

    cells_area = scatter(
        src=(full_synataic_function * surface_vector).sum(dim=1, keepdim=True),
        index=cells_index,
        reduce="sum",
        dim=0,
    ).squeeze(1)
    # use shoelace formula to validate

    test_cells_area = []
    for i in range(cells_index.max().numpy() + 1):
        test_cells_area.append(
            polygon_area(mesh_pos[cells_node[(cells_index == i).view(-1)].view(-1)])
        )
    test_cells_area = torch.from_numpy(np.asarray(test_cells_area))

    valid_cells_area = (cells_area - test_cells_area).sum()

    if not torch.allclose(cells_area, test_cells_area, rtol=1e-05, atol=1e-08, equal_nan=False):
        dataset["cell|cells_area"] = test_cells_area.unsqueeze(1)
        print(
            f"warning substitude cells area with shoelace formula(resdiual:{valid_cells_area.numpy()}) {path['file_dir']}"
        )
    else:
        dataset["cell|cells_area"] = cells_area

    ''' >>> compute face_node_x <<< '''
    domain_list = seperate_domain(
        cells_node=cells_node, 
        cells_face=cells_face, 
        cells_index=cells_index
    )
    
    face_node_x=[]
    for domain in domain_list:
        
        _ct, _cells_node, _cells_face, _cells_index, _ = domain
        
        face_node_x.append(
            compose_support_face_node_x(cells_type=_ct, cells_node=_cells_node)
        )
    face_node_x = torch.cat(face_node_x, dim=1)
    face_node_x = torch.unique(face_node_x[:,~(face_node_x[0]==face_node_x[1])],dim=1)
    dataset["face_node_x"] = face_node_x
    ''' >>> compute face_node_x <<< '''
    
    print(f"{path['case_name']}mesh has been extracted")

    return dataset
