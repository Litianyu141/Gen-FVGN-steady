import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import knn_graph
from Utils.utilities import NodeType
from FVMmodel.FVdiscretization.FVorder import moments_order


def calc_mirror_pos(pos_A, pos_B):
    """
    计算 pos_A 中每个点相对于 pos_B 中对应位置点的对称点坐标。

    参数:
        pos_A (torch.Tensor): 形状为 [N, 2] 的张量，表示 N 个二维点。
        pos_B (torch.Tensor): 形状为 [N, 2] 的张量，表示 N 个二维点。

    返回:
        torch.Tensor: 形状为 [N, 2] 的张量，表示对称点的坐标。
    """
    return 2 * pos_B - pos_A


def calc_replication_ghost(
    phi_node=None,
    mesh_pos=None,
    out_node_index=None,
    pivot_node_index=None,
    mask=None,
):
    if not mask.any():
        return None, None, None

    out_phi = phi_node[out_node_index][mask]
    pivot_phi = phi_node[pivot_node_index][mask]

    out_pos = mesh_pos[out_node_index][mask]
    pivot_pos = mesh_pos[pivot_node_index][mask]

    mirror_pos = calc_mirror_pos(out_pos, pivot_pos)
    mirror_phi = pivot_phi

    pos_diff = mirror_pos - pivot_pos
    phi_diff = mirror_phi - pivot_phi
    recivers_index = pivot_node_index[mask]

    return pos_diff, phi_diff, recivers_index


def calc_reflection_ghost(
    phi_node=None,
    mesh_pos=None,
    out_node_index=None,
    pivot_node_index=None,
    mask=None,
):
    if not mask.any():
        return None, None, None

    out_phi = phi_node[out_node_index][mask]
    pivot_phi = phi_node[pivot_node_index][mask]

    out_pos = mesh_pos[out_node_index][mask]
    pivot_pos = mesh_pos[pivot_node_index][mask]

    mirror_pos = calc_mirror_pos(out_pos, pivot_pos)
    mirror_phi = -out_phi

    pos_diff = mirror_pos - pivot_pos
    phi_diff = mirror_phi - pivot_phi
    recivers_index = pivot_node_index[mask]

    return pos_diff, phi_diff, recivers_index


def calc_ghost_point(
    phi_node=None,
    mesh_pos=None,
    outdegree_node_index=None,
    indegree_node_index=None,
    node_type=None,
):

    outer_node_type = node_type[outdegree_node_index]
    pivot_node_type = node_type[indegree_node_index]

    mask_wall_ghost_edge = (pivot_node_type == NodeType.WALL_BOUNDARY) & (
        outer_node_type == NodeType.NORMAL
    )
    mask_inflow_ghost_edge = (pivot_node_type == NodeType.INFLOW) & (
        outer_node_type == NodeType.NORMAL
    )
    mask_outflow_ghost_edge = (pivot_node_type == NodeType.OUTFLOW) & (
        outer_node_type == NodeType.NORMAL
    )

    pos_diff = []
    phi_diff = []
    recivers_index = []

    pos_diff_wall, phi_diff_wall, recivers_index_wall = calc_reflection_ghost(
        phi_node,
        mesh_pos,
        outdegree_node_index,
        indegree_node_index,
        mask_wall_ghost_edge,
    )
    if pos_diff_wall is not None:
        pos_diff.append(pos_diff_wall)
        phi_diff.append(phi_diff_wall)
        recivers_index.append(recivers_index_wall)

    pos_diff_inflow, phi_diff_inflow, recivers_index_inflow = calc_replication_ghost(
        phi_node,
        mesh_pos,
        outdegree_node_index,
        indegree_node_index,
        mask_inflow_ghost_edge,
    )

    if pos_diff_inflow is not None:
        pos_diff.append(pos_diff_inflow)
        phi_diff.append(phi_diff_inflow)
        recivers_index.append(recivers_index_inflow)

    pos_diff_outflow, phi_diff_outflow, recivers_index_outflow = calc_replication_ghost(
        phi_node,
        mesh_pos,
        outdegree_node_index,
        indegree_node_index,
        mask_outflow_ghost_edge,
    )

    if pos_diff_outflow is not None:
        pos_diff.append(pos_diff_outflow)
        phi_diff.append(phi_diff_outflow)
        recivers_index.append(recivers_index_outflow)

    pos_diff = torch.cat(pos_diff, dim=0)
    phi_diff = torch.cat(phi_diff, dim=0)
    recivers_index = torch.cat(recivers_index, dim=0)

    return pos_diff, phi_diff, recivers_index


def update_Green_Gauss_Gradient():

    pass


def compute_normal_matrix(
    order="1st",
    mesh_pos=None,
    outdegree=None,
    indegree=None,
    dual_edge=True, # 输入的in/outdegree是否是双向的
    periodic_idx=None,
):
    """
    Computes the normal matrices A and B for node-based weighted least squares (WLSQ)
    gradient reconstruction.

    Parameters:
    - order (str): The order of the reconstruction ('1st', '2nd', '3rd', or '4th').
    - mesh_pos (torch.Tensor): Tensor of shape [N, D] containing the positions of the mesh nodes.
    - outdegree (torch.Tensor): Tensor containing the indices of source nodes (outgoing edges).
    - indegree (torch.Tensor): Tensor containing the indices of target nodes (incoming edges).
    - dual_edge (bool): If True, the provided outdegree and indegree represent bidirectional edges.
                        If False, the function constructs bidirectional edges by concatenating
                        the input edges.

    Returns:
    - A_node_to_node (torch.Tensor): Normal matrix A for each node.
    - B_node_to_node (torch.Tensor): Matrix B for each node.
    """
    
    if dual_edge:
        outdegree_node_index, indegree_node_index = outdegree, indegree
    else:
        outdegree_node_index = torch.cat((outdegree, indegree), dim=0)
        indegree_node_index = torch.cat((indegree, outdegree), dim=0)
        
    mesh_pos_diff_on_edge = mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]

    (A_node_to_node, B_node_to_node) = moments_order(
        order=order,
        mesh_pos_diff_on_edge=mesh_pos_diff_on_edge,
        indegree_node_index=indegree_node_index,
    )

    # 由于周期边界条件的加入，那么需要将src的A传递给dst的A，B也同理，只不过不在预处理这里传递，在下面求解矩阵的计算中传递
    if periodic_idx is not None:
        valid_peridx = periodic_idx[periodic_idx[0]>=0] #  使用-1来屏蔽非周期边界的idx
        periodic_A = A_node_to_node[valid_peridx[0]] + A_node_to_node[valid_peridx[1]]
        A_node_to_node[valid_peridx[0]] = periodic_A
        A_node_to_node[valid_peridx[1]] = periodic_A
    
    return (A_node_to_node, B_node_to_node)

@torch.compile
def node_based_WLSQ(
    phi_node=None,
    edge_index=None,
    mesh_pos=None,
    dual_edge=True, # 输入的edge_index是否是双向的
    order=None,
    precompute_Moments: list = None,
    periodic_idx=None,
):
    '''
    B right-hand sides in precompute_Moments must be SINGLE-WAY
    on edge
    '''
    # edge_index = knn_graph(mesh_pos, k=9, loop=False)
    if (order is None) or (order not in ["1st", "2nd", "3rd", "4th"]):
        raise ValueError("order must be specified in [\"1st\", \"2nd\", \"3rd\", \"4th\"]")
    
    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1]
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

    if precompute_Moments is None:

        """node to node contribution"""
        (A_node_to_node, two_way_B_node_to_node) = compute_normal_matrix(
            order=order,
            mesh_pos=mesh_pos,
            outdegree=outdegree_node_index,
            indegree=indegree_node_index,
            dual_edge=False if dual_edge else True,
        )
        """node to node contribution"""

        phi_diff_on_edge = two_way_B_node_to_node * (
            (phi_node[outdegree_node_index] - phi_node[indegree_node_index]).unsqueeze(
                1
            )
        )

        B_phi_node_to_node = scatter_add(
            phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
        )

    else:
        """use precomputed moments"""
        A_node_to_node, Oneway_B_node_to_node = precompute_Moments

        half_dim = Oneway_B_node_to_node.shape[0]
        
        two_way_B_node_to_node = torch.cat(
            (Oneway_B_node_to_node, Oneway_B_node_to_node), dim=0
        )
        
        # 大于1阶的奇数阶项需要取负
        two_way_B_node_to_node[half_dim:,0:2]*= -1
        od = int(order[0])
        
        if od >=3 :
            two_way_B_node_to_node[half_dim:,5:9]*= -1
            
        phi_diff_on_edge = two_way_B_node_to_node * (
            (phi_node[outdegree_node_index] - phi_node[indegree_node_index]).unsqueeze(
                1
            )
        )

        B_phi_node_to_node = scatter_add(
            phi_diff_on_edge,
            indegree_node_index,
            dim=0,
            dim_size=mesh_pos.shape[0],
        )
        
    # 处理周期边界条件，假如A矩阵已经处理好了，这里来处理B矩阵
    if periodic_idx is not None:
        valid_peridx = periodic_idx[periodic_idx[0]>=0] #  使用-1来屏蔽非周期边界的idx
        periodic_B = B_phi_node_to_node[valid_peridx[0]] + B_phi_node_to_node[valid_peridx[1]]
        B_phi_node_to_node[valid_peridx[0]] = periodic_B
        B_phi_node_to_node[valid_peridx[1]] = periodic_B
    
    # 行归一化
    row_norms = torch.norm(A_node_to_node, p=2, dim=2, keepdim=True)
    A_normalized = A_node_to_node / (row_norms + 1e-8)
    B_normalized = B_phi_node_to_node / (row_norms + 1e-8)
    
    # lambda_reg = 1e-5  # 正则化参数
    # I = torch.eye(A_normalized.shape[-1], device=A_normalized.device)
    # A_normalized = A_normalized + lambda_reg * I
    
    # # 列归一化
    # col_norms = torch.norm(A_normalized, p=2, dim=1, keepdim=True)
    # A_normalized = A_normalized / (col_norms + 1e-8)
    # B_normalized = B_normalized * col_norms
    
    """ first method"""
    # nabla_phi_node_lst = torch.linalg.lstsq(
    #     A_normalized, B_normalized
    # ).solution.transpose(1, 2)

    """ second method"""
    # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

    """ third method"""
    nabla_phi_node_lst = torch.linalg.solve(
        A_normalized, B_normalized
    ).transpose(1, 2)

    """ fourth method"""
    # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)

    return nabla_phi_node_lst


def node_based_WLSQ_2nd_order(
    phi_node=None,
    edge_index=None,
    mesh_pos=None,
    dual_edge=True,
):
    # edge_index = knn_graph(mesh_pos, k=9, loop=False)

    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1]
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

    """node to node contribution"""
    mesh_pos_diff_on_edge = (
        mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
    )

    displacement = torch.cat(
        (
            mesh_pos_diff_on_edge,
            0.5 * (mesh_pos_diff_on_edge**2),
            mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2],
        ),
        dim=-1,
    ).unsqueeze(2)

    displacement_T = displacement.transpose(1, 2)

    r_d = torch.norm(mesh_pos_diff_on_edge, dim=1, keepdim=True) ** 3

    weight_node_to_node = 1 / r_d.unsqueeze(2)

    left_on_edge = torch.matmul(
        displacement * weight_node_to_node,
        displacement_T,
    )

    A_node_to_node = scatter_add(
        left_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )
    """node to node contribution"""

    phi_diff_on_edge = (
        weight_node_to_node
        * (
            (
                phi_node[outdegree_node_index] - phi_node[indegree_node_index]
            ).unsqueeze(1)
        )
        * displacement
    )

    B_phi_node_to_node = scatter_add(
        phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )

    """ first method"""
    # nabla_phi_node_lst = torch.linalg.lstsq(
    #     A_node_to_node_x, B_phi_node_to_node_x
    # ).solution

    """ second method"""
    # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

    """ third method"""
    nabla_phi_node_lst = torch.linalg.solve(
        A_node_to_node, B_phi_node_to_node
    ).transpose(1, 2)
    # [N,C,[ux, uy, uxx, uyy, uxy]]
    
    """ fourth method"""
    # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)
    
    return nabla_phi_node_lst


def node_based_WLSQ_3rd_order(
    phi_node=None,
    edge_index=None,
    mesh_pos=None,
    dual_edge=True,
):
    # edge_index = knn_graph(mesh_pos, k=9, loop=False)
    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1]
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

    """node to node contribution"""
    mesh_pos_diff_on_edge = (
        mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
    )
    displacement = torch.cat(
        (
            mesh_pos_diff_on_edge,
            0.5 * (mesh_pos_diff_on_edge**2),
            mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2],
            (1 / 6) * (mesh_pos_diff_on_edge**3),
            0.5 * (mesh_pos_diff_on_edge[:, 0:1] ** 2) * mesh_pos_diff_on_edge[:, 1:2],
            0.5 * (mesh_pos_diff_on_edge[:, 1:2] ** 2) * mesh_pos_diff_on_edge[:, 0:1],
        ),
        dim=-1,
    ).unsqueeze(2)

    displacement_T = displacement.transpose(1, 2)

    r_d = torch.norm(mesh_pos_diff_on_edge, dim=1, keepdim=True) ** 4
    # weight_node_to_node = 4 / (torch.pi * (1 - r_d**2)).unsqueeze(2)
    # weight_node_to_node = torch.sqrt(torch.tensor(4)/torch.pi) * ((1 - r_d**2)**4).unsqueeze(2)
    weight_node_to_node = 1 / r_d.unsqueeze(2)

    left_on_edge = torch.matmul(
        displacement * weight_node_to_node,
        displacement_T,
    )

    A_node_to_node = scatter_add(
        left_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )
    """node to node contribution"""

    phi_diff_on_edge = (
        weight_node_to_node
        * (
            (phi_node[outdegree_node_index] - phi_node[indegree_node_index]).unsqueeze(
                1
            )
        )
        * displacement
    )

    B_phi_node_to_node = scatter_add(
        phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )

    """ first method"""
    # nabla_phi_node_lst = torch.linalg.lstsq(A_node_to_node, B_phi_node_to_node).solution

    """ second method"""
    # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

    """ third method"""
    nabla_phi_node_lst = torch.linalg.solve(
        A_node_to_node, B_phi_node_to_node
    ).transpose(1, 2)
    # [N,C,[ux, uy, uxx, uyy, uxy, uxxx, uyyy, uxxy, uxyy]]
    
    """ fourth method"""
    # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)

    return nabla_phi_node_lst


def node_based_WLSQ_4th_order(
    phi_node=None,
    edge_index=None,
    mesh_pos=None,
    dual_edge=True,
):
    # edge_index = knn_graph(mesh_pos, k=9, loop=False)
    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1]
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

    """node to node contribution"""
    mesh_pos_diff_on_edge = (
        mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
    )
    displacement = torch.cat(
        (
            mesh_pos_diff_on_edge,
            0.5 * (mesh_pos_diff_on_edge**2),
            mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2],
            (1 / 6) * (mesh_pos_diff_on_edge**3),
            0.5 * (mesh_pos_diff_on_edge[:, 0:1] ** 2) * mesh_pos_diff_on_edge[:, 1:2],
            0.5 * (mesh_pos_diff_on_edge[:, 1:2] ** 2) * mesh_pos_diff_on_edge[:, 0:1],
            (1 / 24) * (mesh_pos_diff_on_edge[:, 0:1] ** 4),
            (1 / 6)
            * (mesh_pos_diff_on_edge[:, 0:1] ** 3)
            * mesh_pos_diff_on_edge[:, 1:2],
            (1 / 4)
            * (mesh_pos_diff_on_edge[:, 0:1] ** 2)
            * (mesh_pos_diff_on_edge[:, 1:2] ** 2),
            (1 / 6)
            * (mesh_pos_diff_on_edge[:, 0:1])
            * (mesh_pos_diff_on_edge[:, 1:2] ** 3),
            (1 / 24) * (mesh_pos_diff_on_edge[:, 1:2] ** 4),
        ),
        dim=-1,
    ).unsqueeze(2)

    displacement_T = displacement.transpose(1, 2)

    r_d = torch.norm(mesh_pos_diff_on_edge, dim=1, keepdim=True) ** 5
    # weight_node_to_node = torch.sqrt(torch.tensor(4)/torch.pi) * ((1 - r_d**2)**4).unsqueeze(2)
    weight_node_to_node = 1 / r_d.unsqueeze(2)

    left_on_edge = torch.matmul(
        displacement * weight_node_to_node,
        displacement_T,
    )

    A_node_to_node = scatter_add(
        left_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )
    """node to node contribution"""

    phi_diff_on_edge = (
        weight_node_to_node
        * (
            (phi_node[outdegree_node_index] - phi_node[indegree_node_index]).unsqueeze(
                1
            )
        )
        * displacement
    )

    B_phi_node_to_node = scatter_add(
        phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )

    """ first method"""
    # nabla_phi_node_lst = torch.linalg.lstsq(A_node_to_node, B_phi_node_to_node).solution

    """ second method"""
    # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

    """ third method"""
    nabla_phi_node_lst = torch.linalg.solve(
        A_node_to_node, B_phi_node_to_node
    ).transpose(1, 2)
    # [N,C,[ux, uy, uxx, uyy, uxy, uxxx, uyyy, uxxy, uxyy...]]
    
    """ fourth method"""
    # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)

    return nabla_phi_node_lst
    # u_{x}, u_{y}, u_{x x}, u_{y y}, u_{x y}, u_{x x x}, u_{y y y}, u_{x x y}, u_{x y y},
    # u_{x x x x}, u_{x x x y}, u_{x x y y}, u_{x y y y}, u_{y y y y}


def Moving_LSQ(
    phi_node=None,
    edge_index=None,
    mesh_pos=None,
    dual_edge=True,
    order=None,
    precompute_Moments: list = None,
    mask_boundary=None,
):
 
    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1]
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

    """node to node contribution"""
    mesh_pos_diff_on_edge = (
        mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
    )
    # In MLS, we need to first calculate the weight matrix
    radius = torch.norm(mesh_pos_diff_on_edge, dim=1, keepdim=True)
    max_node_radius = scatter_max(
        radius, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )[0]
    weight_node_to_node = torch.exp(-torch.pow(radius / max_node_radius[outdegree_node_index], 2))

    displacement = torch.cat(
        (
            weight_node_to_node,
            mesh_pos_diff_on_edge*weight_node_to_node,
            0.5 * (mesh_pos_diff_on_edge**2)*weight_node_to_node,
            mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2]*weight_node_to_node,
        ),
        dim=-1,
    ).unsqueeze(2)

    displacement_T = displacement.transpose(1, 2)

    left_on_edge = torch.matmul(
        displacement,
        displacement_T,
    )

    A_node_to_node = scatter_add(
        left_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )
    """node to node contribution"""

    phi_on_edge = (
        (
            (
                phi_node[outdegree_node_index]*weight_node_to_node
            ).unsqueeze(1)
        )
        * displacement
    )

    B_phi_node_to_node = scatter_add(
        phi_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )

    """ first method"""
    # nabla_phi_node_lst = torch.linalg.lstsq(
    #     A_node_to_node_x, B_phi_node_to_node_x
    # ).solution

    """ second method"""
    # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

    """ third method"""
    nabla_phi_node_lst = torch.linalg.solve(
        A_node_to_node, B_phi_node_to_node
    ).transpose(1, 2)
    # [N,C,[ux, uy, uxx, uyy, uxy]]
    
    """ fourth method"""
    # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)
    
    return nabla_phi_node_lst[:,:,1:]

