import numpy as np
from torch_geometric.data import Data
import enum
import torch
from torch_scatter import scatter

flow_type_mapping = {
    "1": "pipe_flow",
    "2": "farfield-circular",
    "3": "cavity_flow",
    "4": "farfield-square",
    "5": "farfield-half-circular-square",
    "6": "circular-possion",
    "7": "cavity_wave",
}


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
    BOUNDARY_CELL = 10
    IN_WALL = 11
    OUT_WALL = 12
    GHOST_INFLOW = 13
    GHOST_OUTFLOW = 14
    GHOST_WALL = 15
    GHOST_AIRFOIL = 16


def calculate_diffusion_term(graph):
    pass


def caculate_advection_term(graph):
    pass


def caculate_velocity(graph):
    pass


def caculate_pressure(graph):
    pass


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


def extract_cylinder_boundary_mask(
    graph_node: Data, graph_edge: Data, graph_cell: Data
):
    face_node = graph_node.edge_index
    node_type = graph_node.node_type
    mesh_pos = graph_node.pos

    node_topwall = torch.max(mesh_pos[:, 1])
    node_bottomwall = torch.min(mesh_pos[:, 1])
    node_outlet = torch.max(mesh_pos[:, 0])
    node_inlet = torch.min(mesh_pos[:, 0])

    face_type = graph_edge.x[:, 0:1]
    left_face_node_pos = torch.index_select(mesh_pos, 0, face_node[0])
    right_face_node_pos = torch.index_select(mesh_pos, 0, face_node[1])

    left_face_node_type = torch.index_select(node_type, 0, face_node[0])
    right_face_node_type = torch.index_select(node_type, 0, face_node[1])

    face_center_pos = (left_face_node_pos + right_face_node_pos) / 2.0

    face_topwall = torch.max(face_center_pos[:, 1])
    face_bottomwall = torch.min(face_center_pos[:, 1])
    face_outlet = torch.max(face_center_pos[:, 0])
    face_inlet = torch.min(face_center_pos[:, 0])

    MasknodeT = torch.full((mesh_pos.shape[0], 1), True).cuda()
    MasknodeF = torch.logical_not(MasknodeT).cuda()

    MaskfaceT = torch.full((face_node.shape[1], 1), True).cuda()
    MaskfaceF = torch.logical_not(MaskfaceT).cuda()

    cylinder_node_mask = torch.where(
        (
            (node_type == NodeType.WALL_BOUNDARY)
            & (mesh_pos[:, 1:2] < node_topwall)
            & (mesh_pos[:, 1:2] > node_bottomwall)
            & (mesh_pos[:, 0:1] > node_inlet)
            & (mesh_pos[:, 0:1] < node_outlet)
        ),
        MasknodeT,
        MasknodeF,
    ).squeeze(1)

    cylinder_face_mask = torch.where(
        (
            (face_type == NodeType.WALL_BOUNDARY)
            & (face_center_pos[:, 1:2] < face_topwall)
            & (face_center_pos[:, 1:2] > face_bottomwall)
            & (face_center_pos[:, 0:1] > face_inlet)
            & (face_center_pos[:, 0:1] < face_outlet)
            & (left_face_node_pos[:, 1:2] < node_topwall)
            & (left_face_node_pos[:, 1:2] > node_bottomwall)
            & (left_face_node_pos[:, 0:1] > node_inlet)
            & (left_face_node_pos[:, 0:1] < node_outlet)
            & (right_face_node_pos[:, 1:2] < node_topwall)
            & (right_face_node_pos[:, 1:2] > node_bottomwall)
            & (right_face_node_pos[:, 0:1] > node_inlet)
            & (right_face_node_pos[:, 0:1] < node_outlet)
            & (left_face_node_type == NodeType.WALL_BOUNDARY)
            & (right_face_node_type == NodeType.WALL_BOUNDARY)
        ),
        MaskfaceT,
        MaskfaceF,
    ).squeeze(1)

    # plt.scatter(face_center_pos[cylinder_face_mask].cpu().numpy()[:,0],face_center_pos[cylinder_face_mask].cpu().numpy()[:,1],edgecolors='red')
    # plt.show()
    return cylinder_node_mask, cylinder_face_mask


def extract_cylinder_boundary(
    dataset,
    mask_node_boundary,
    mask_face_boundary,
    graph_node: Data,
    graph_edge: Data,
    graph_cell: Data,
    rho=None,
    mu=None,
    dt=None,
):
    write_zone = {"name": "OBSTACLE", "rho": rho, "mu": mu, "dt": dt}

    write_zone["node|X"] = (
        graph_node.pos[mask_node_boundary, 0:1].to("cpu").unsqueeze(0).numpy()
    )
    write_zone["node|Y"] = (
        graph_node.pos[mask_node_boundary, 1:2].to("cpu").unsqueeze(0).numpy()
    )
    write_zone["node|U"] = (
        dataset["predicted_node_uvp"][:, mask_node_boundary, 0:1].to("cpu").numpy()
    )
    write_zone["node|V"] = (
        dataset["predicted_node_uvp"][:, mask_node_boundary, 1:2].to("cpu").numpy()
    )
    write_zone["node|P"] = (
        dataset["predicted_node_uvp"][:, mask_node_boundary, 2:3].to("cpu").numpy()
    )
    # write_zone["face"] = graph_node.edge_index[:,mask_face_boundary].to('cpu').transpose(0,1).unsqueeze(0).numpy()

    origin_mesh_pos = graph_node.pos.to("cpu")

    boundary_mesh_pos = graph_node.pos[mask_node_boundary, :].to("cpu")

    index_mapping = {
        str(bondary_vertex_pos.view(-1).numpy()): new_index
        for new_index, bondary_vertex_pos in enumerate(boundary_mesh_pos)
    }

    boundary_face_unordered = (
        graph_node.edge_index[:, mask_face_boundary].to("cpu").view(-1, 1)
    )

    boundary_face_mapped = torch.tensor(
        [
            index_mapping[str(origin_mesh_pos[idx].view(-1).numpy())]
            for idx in boundary_face_unordered
        ],
        dtype=torch.long,
    )

    write_zone["face"] = (
        torch.stack(torch.chunk(boundary_face_mapped, 2), dim=1).unsqueeze(0).numpy()
    )

    return write_zone


def extract_cylinder_boundary_only_training(
    dataset=None, params=None, rho=None, mu=None, dt=None
):
    face_node = dataset["face"][0].long()
    if face_node.shape[0] > face_node.shape[1]:
        face_node = face_node.T

    node_type = dataset["node_type"][0]
    mesh_pos = dataset["mesh_pos"][0]

    # centroid = dataset["centroid"][0]
    cells_face = dataset["cells_face"][0]
    if cells_face.shape[0] > cells_face.shape[1]:
        cells_face = cells_face.T

    cells_node = dataset["cells_node"][0]
    if cells_node.shape[0] > cells_node.shape[1]:
        cells_node = cells_node.T

    node_topwall = torch.max(mesh_pos[:, 1])
    node_bottomwall = torch.min(mesh_pos[:, 1])
    node_outlet = torch.max(mesh_pos[:, 0])
    node_inlet = torch.min(mesh_pos[:, 0])

    face_type = dataset["face_type"][0]
    left_face_node_pos = torch.index_select(mesh_pos, 0, face_node[0])
    right_face_node_pos = torch.index_select(mesh_pos, 0, face_node[1])

    left_face_node_type = torch.index_select(node_type, 0, face_node[0])
    right_face_node_type = torch.index_select(node_type, 0, face_node[1])

    face_center_pos = (left_face_node_pos + right_face_node_pos) / 2.0

    face_topwall = torch.max(face_center_pos[:, 1])
    face_bottomwall = torch.min(face_center_pos[:, 1])
    face_outlet = torch.max(face_center_pos[:, 0])
    face_inlet = torch.min(face_center_pos[:, 0])

    MasknodeT = torch.full((mesh_pos.shape[0], 1), True)
    MasknodeF = torch.logical_not(MasknodeT)

    MaskfaceT = torch.full((face_node.shape[1], 1), True)
    MaskfaceF = torch.logical_not(MaskfaceT)

    mask_node_boundary = torch.where(
        (
            (node_type == NodeType.WALL_BOUNDARY)
            & (mesh_pos[:, 1:2] < node_topwall)
            & (mesh_pos[:, 1:2] > node_bottomwall)
            & (mesh_pos[:, 0:1] > node_inlet)
            & (mesh_pos[:, 0:1] < node_outlet)
        ),
        MasknodeT,
        MasknodeF,
    ).squeeze(1)

    mask_face_boundary = torch.where(
        (
            (face_type == NodeType.WALL_BOUNDARY)
            & (face_center_pos[:, 1:2] < face_topwall)
            & (face_center_pos[:, 1:2] > face_bottomwall)
            & (face_center_pos[:, 0:1] > face_inlet)
            & (face_center_pos[:, 0:1] < face_outlet)
            & (left_face_node_pos[:, 1:2] < node_topwall)
            & (left_face_node_pos[:, 1:2] > node_bottomwall)
            & (left_face_node_pos[:, 0:1] > node_inlet)
            & (left_face_node_pos[:, 0:1] < node_outlet)
            & (right_face_node_pos[:, 1:2] < node_topwall)
            & (right_face_node_pos[:, 1:2] > node_bottomwall)
            & (right_face_node_pos[:, 0:1] > node_inlet)
            & (right_face_node_pos[:, 0:1] < node_outlet)
            & (left_face_node_type == NodeType.WALL_BOUNDARY)
            & (right_face_node_type == NodeType.WALL_BOUNDARY)
        ),
        MaskfaceT,
        MaskfaceF,
    ).squeeze(1)

    boundary_zone = {"name": "OBSTACLE", "rho": rho, "mu": mu, "dt": dt}

    boundary_zone["mask_node_boundary"] = mask_node_boundary
    boundary_zone["mask_face_boundary"] = mask_face_boundary
    # boundary_zone["mask_cell_boundary"] = mask_cell_boundary

    origin_mesh_pos = mesh_pos

    boundary_mesh_pos = mesh_pos[mask_node_boundary, :]

    index_mapping = {
        str(bondary_vertex_pos.view(-1).numpy()): new_index
        for new_index, bondary_vertex_pos in enumerate(boundary_mesh_pos)
    }

    boundary_face_unordered = face_node[:, mask_face_boundary].view(-1, 1)

    boundary_face_mapped = torch.tensor(
        [
            index_mapping[str(origin_mesh_pos[idx].view(-1).numpy())]
            for idx in boundary_face_unordered
        ],
        dtype=torch.long,
    )

    boundary_zone["face"] = (
        torch.stack(torch.chunk(boundary_face_mapped, 2), dim=1).unsqueeze(0).numpy()
    )
    boundary_zone["mesh_pos"] = boundary_mesh_pos.unsqueeze(0).numpy()
    boundary_zone["zonename"] = "OBSTICALE_BOUNDARY"

    return boundary_zone
