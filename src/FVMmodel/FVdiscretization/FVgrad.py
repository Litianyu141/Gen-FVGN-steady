import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import knn_graph
from Utils.utilities import NodeType
from FVMmodel.FVdiscretization.FVorder import moments_order


def calc_mirror_pos(pos_A, pos_B):
    """
    Calculate the mirror position of each point in pos_A with respect to pos_B.
    Args:
        pos_A (Tensor): [N, 2] coordinates of N points.
        pos_B (Tensor): [N, 2] coordinates of N points.
    Returns:
        Tensor: [N, 2] coordinates of the mirrored points.
    """
    return 2 * pos_B - pos_A


def calc_replication_ghost(
    phi_node=None,
    mesh_pos=None,
    out_node_index=None,
    pivot_node_index=None,
    mask=None,
):
    """
    Calculate ghost point values for replication boundary (inflow/outflow).
    Args:
        phi_node (Tensor): [N, C] Node values.
        mesh_pos (Tensor): [N, 2] Node positions.
        out_node_index (Tensor): [E] Outgoing node indices.
        pivot_node_index (Tensor): [E] Pivot node indices.
        mask (Tensor): [E] Boolean mask for valid edges.
    Returns:
        Tuple of (pos_diff, phi_diff, receivers_index) for ghost points.
    """
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
    """
    Calculate ghost point values for reflection boundary (wall).
    Args:
        phi_node (Tensor): [N, C] Node values.
        mesh_pos (Tensor): [N, 2] Node positions.
        out_node_index (Tensor): [E] Outgoing node indices.
        pivot_node_index (Tensor): [E] Pivot node indices.
        mask (Tensor): [E] Boolean mask for valid edges.
    Returns:
        Tuple of (pos_diff, phi_diff, receivers_index) for ghost points.
    """
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

    """
    Calculate ghost point differences for all boundary types (wall, inflow, outflow).
    Args:
        phi_node (Tensor): [N, C] Node values.
        mesh_pos (Tensor): [N, 2] Node positions.
        outdegree_node_index (Tensor): [E] Outgoing node indices.
        indegree_node_index (Tensor): [E] Incoming node indices.
        node_type (Tensor): [N] Node type.
    Returns:
        Tuple of (pos_diff, phi_diff, receivers_index) for all ghost points.
    """
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

    """
    Placeholder for Green-Gauss gradient update (not implemented).
    """
    pass


def compute_normal_matrix(
    order="1st",
    mesh_pos=None,
    edge_index=None, # 默认应该是仅包含1阶邻居点+构成共点的单元的所有点
    extra_edge_index=None, # 额外的模板。例如内部点指向边界点
    periodic_idx=None,
):
    """
    Compute the normal matrices A and B for node-based weighted least squares (WLSQ) gradient reconstruction.
    Args:
        order (str): Order of reconstruction ('1st', '2nd', '3rd', '4th').
        mesh_pos (Tensor): [N, D] Node positions.
        edge_index (Tensor): [2, E] Edge indices.
        extra_edge_index (Tensor, optional): [2, E_extra] Extra edge indices.
        periodic_idx (Tensor, optional): Periodic boundary indices.
    Returns:
        Tuple of (A_node_to_node, B_node_to_node[:split_index], B_node_to_node[split_index:]).
    """
    
    twoway_edge_index = torch.cat((edge_index,edge_index.flip(0)),dim=1)

    if extra_edge_index is not None:
        complete_edge_index = torch.cat((twoway_edge_index,extra_edge_index),dim=1)
    else:
        complete_edge_index = twoway_edge_index
        
    split_index = twoway_edge_index.shape[1] # 返回的B中要区分双向的部分和单向的部分
    
    outdegree_node_index, indegree_node_index = complete_edge_index[0], complete_edge_index[1]   
    
    mesh_pos_diff_on_edge = mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
    
    # L_local = scatter_mean(src=mesh_pos_diff_on_edge**2,index=indegree_node_index,dim=0).sqrt()
    
    # normaled_mesh_pos_diff_on_edge = mesh_pos_diff_on_edge/(L_local[indegree_node_index])
    
    (A_node_to_node, B_node_to_node) = moments_order(
        order=order,
        mesh_pos_diff_on_edge=mesh_pos_diff_on_edge,
        indegree_node_index=indegree_node_index,
    )

    # # 由于周期边界条件的加入，那么需要将src的A传递给dst的A，B也同理，只不过不在预处理这里传递，在下面求解矩阵的计算中传递
    # if periodic_idx is not None:
    #     valid_peridx = periodic_idx[periodic_idx[0]>=0] #  使用-1来屏蔽非周期边界的idx
    #     periodic_A = A_node_to_node[valid_peridx[0]] + A_node_to_node[valid_peridx[1]]
    #     A_node_to_node[valid_peridx[0]] = periodic_A
    #     A_node_to_node[valid_peridx[1]] = periodic_A
    
    return (A_node_to_node, B_node_to_node[:split_index], B_node_to_node[split_index:])

# @torch.compile
def node_based_WLSQ(
    phi_node=None,
    edge_index=None, # 输入的edge_index是否是双向的,注意对于edge_index一定是0-1
    extra_edge_index=None,
    mesh_pos=None,
    order=None,
    precompute_Moments: list = None, # 应一定包含3个元素，[A, 单向的B，和额外的B（即仅内部点指向边界点）]
    periodic_idx=None, 
    rt_cond=False,
):
    '''
    Node-based Weighted Least Squares (WLSQ) gradient reconstruction.
    Args:
        phi_node (Tensor): [N, C] Node values.
        edge_index (Tensor): [2, E] Edge indices.
        extra_edge_index (Tensor, optional): [2, E_extra] Extra edge indices.
        mesh_pos (Tensor): [N, D] Node positions.
        order (str): Order of reconstruction ('1st', '2nd', '3rd', '4th').
        precompute_Moments (list, optional): Precomputed [A, B, extra_B] moments.
        periodic_idx (Tensor, optional): Periodic boundary indices.
        rt_cond (bool): If True, also return condition number.
    Returns:
        nabla_phi_node_lst (Tensor): [N, C, ...] Node gradients (shape depends on order).
        If rt_cond is True, also returns condition number.
    '''
    # edge_index = knn_graph(mesh_pos, k=9, loop=False)
    if (order is None) or (order not in ["1st", "2nd", "3rd", "4th"]):
        raise ValueError("order must be specified in [\"1st\", \"2nd\", \"3rd\", \"4th\"]")
    
    twoway_edge_index = torch.cat((edge_index,edge_index.flip(0)),dim=1)

    if extra_edge_index is not None:
        complete_edge_index = torch.cat((twoway_edge_index,extra_edge_index),dim=1)
    else:
        complete_edge_index = twoway_edge_index

    outdegree_node_index, indegree_node_index = complete_edge_index[0], complete_edge_index[1]   

    if precompute_Moments is None:

        """node to node contribution"""
        (A_node_to_node, two_way_B_node_to_node,single_way_B_node_to_node) = compute_normal_matrix(
            order=order,
            mesh_pos=mesh_pos,
            edge_index=edge_index, # 默认应该是仅包含1阶邻居点+构成共点的单元的所有点
            extra_edge_index=extra_edge_index, # 额外的模板。例如内部点指向边界点
        )
        B_node_to_node = torch.cat((two_way_B_node_to_node,single_way_B_node_to_node),dim=0)
        """node to node contribution"""
        
        phi_diff_on_edge = B_node_to_node * (
            (phi_node[outdegree_node_index] - phi_node[indegree_node_index]).unsqueeze(
                1
            )
        )

        B_phi_node_to_node = scatter_add(
            phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
        )

    else:
        """use precomputed moments"""
        A_node_to_node, Oneway_B_node_to_node, extra_single_way_B_node_to_node = precompute_Moments

        half_dim = Oneway_B_node_to_node.shape[0]
        
        two_way_B_node_to_node = torch.cat(
            (Oneway_B_node_to_node, Oneway_B_node_to_node), dim=0
        )
        
        # 大于1阶的奇数阶项需要取负
        two_way_B_node_to_node[half_dim:,0:2]*= -1
        od = int(order[0])
        
        if od >=3 :
            two_way_B_node_to_node[half_dim:,5:9]*= -1
        
        B_node_to_node = torch.cat((two_way_B_node_to_node, extra_single_way_B_node_to_node),dim=0)
        
        phi_diff_on_edge = B_node_to_node * (
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
        
    # # 处理周期边界条件，假如A矩阵已经处理好了，这里来处理B矩阵
    # if periodic_idx is not None:
    #     valid_peridx = periodic_idx[periodic_idx[0]>=0] #  使用-1来屏蔽非周期边界的idx
    #     periodic_B = B_phi_node_to_node[valid_peridx[0]] + B_phi_node_to_node[valid_peridx[1]]
    #     B_phi_node_to_node[valid_peridx[0]] = periodic_B
    #     B_phi_node_to_node[valid_peridx[1]] = periodic_B
    
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

    if rt_cond:
        return nabla_phi_node_lst,torch.linalg.cond(A_normalized) 
    else:
        return nabla_phi_node_lst


def node_based_WLSQ_2nd_order(
    phi_node=None,
    edge_index=None,
    mesh_pos=None,
    dual_edge=True,
):
    """
    Node-based WLSQ gradient reconstruction (2nd order).
    Args:
        phi_node (Tensor): [N, C] Node values.
        edge_index (Tensor): [2, E] Edge indices.
        mesh_pos (Tensor): [N, 2] Node positions.
        dual_edge (bool): If True, use bidirectional edges.
    Returns:
        nabla_phi_node_lst (Tensor): [N, C, 5] Gradients and second derivatives.
    """
    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1] # edge_index must be 0-1
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

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
    """
    Node-based WLSQ gradient reconstruction (3rd order).
    Args:
        phi_node (Tensor): [N, C] Node values.
        edge_index (Tensor): [2, E] Edge indices.
        mesh_pos (Tensor): [N, 2] Node positions.
        dual_edge (bool): If True, use bidirectional edges.
    Returns:
        nabla_phi_node_lst (Tensor): [N, C, 9] Gradients and higher derivatives.
    """
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
    """
    Node-based WLSQ gradient reconstruction (4th order).
    Args:
        phi_node (Tensor): [N, C] Node values.
        edge_index (Tensor): [2, E] Edge indices.
        mesh_pos (Tensor): [N, 2] Node positions.
        dual_edge (bool): If True, use bidirectional edges.
    Returns:
        nabla_phi_node_lst (Tensor): [N, C, 14] Gradients and higher derivatives.
    """
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
 
    """
    Moving Least Squares (MLS) gradient reconstruction.
    Args:
        phi_node (Tensor): [N, C] Node values.
        edge_index (Tensor): [2, E] Edge indices.
        mesh_pos (Tensor): [N, 2] Node positions.
        dual_edge (bool): If True, use bidirectional edges.
        order (str): Order of reconstruction.
        precompute_Moments (list, optional): Precomputed moments.
        mask_boundary (Tensor, optional): Boundary mask.
    Returns:
        nabla_phi_node_lst (Tensor): [N, C, ...] Node gradients (shape depends on order).
    """
    if dual_edge:
        outdegree_node_index, indegree_node_index = edge_index[0], edge_index[1]
    else:
        outdegree_node_index = torch.cat((edge_index[0], edge_index[1]), dim=0)
        indegree_node_index = torch.cat((edge_index[1], edge_index[0]), dim=0)

    mesh_pos_diff_on_edge = (
        mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
    )
    # In MLS, first calculate the weight matrix
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

