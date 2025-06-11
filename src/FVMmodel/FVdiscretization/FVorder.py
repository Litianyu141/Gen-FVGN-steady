import torch
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import knn_graph
from Utils.utilities import NodeType


def moments_order(
    order="1nd",
    mesh_pos_diff_on_edge=None,
    indegree_node_index=None,
):
    """
    Compute moments for WLSQ reconstruction.
    Args:
        order (str): Order of reconstruction ('1st', '2nd', '3rd', '4th').
        mesh_pos_diff_on_edge (Tensor): [2*E, 2] Edge position differences.
        indegree_node_index (Tensor): [N] Node indices.
    Returns:
        Tuple of (A_cell_to_cell, B_cell_to_cell).
        A_cell_to_cell: [N, x, x], x depends on order.
        B_cell_to_cell: [2*E, x]
    """
    if order=="1st":
        od=1
        displacement = mesh_pos_diff_on_edge.unsqueeze(2)
    elif order=="2nd":
        od=1
        displacement = torch.cat(
            (
                mesh_pos_diff_on_edge,
                0.5 * (mesh_pos_diff_on_edge**2),
                mesh_pos_diff_on_edge[:, 0:1] * mesh_pos_diff_on_edge[:, 1:2],
            ),
            dim=-1,
        ).unsqueeze(2)
    elif order=="3rd":
        od=1
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
    elif order=="4th":
        od=1
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
    else:
        raise NotImplementedError(f"{order} Order not implemented")
    displacement_T = displacement.transpose(1, 2)
    weight_cell_to_cell = (1 / torch.norm(mesh_pos_diff_on_edge, dim=1, keepdim=True)**od).unsqueeze(2)
    left_on_edge = torch.matmul(
        displacement * weight_cell_to_cell,
        displacement_T,
    )
    A_cell_to_cell = scatter_add(
        left_on_edge, indegree_node_index, dim=0
    ) # [N, x, x], x is depend on order
    B_cell_to_cell = weight_cell_to_cell * displacement
    # [2*E, x]
    return A_cell_to_cell, B_cell_to_cell