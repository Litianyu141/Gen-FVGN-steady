import torch.nn as nn
import torch
from .blocks import EdgeBlock, NodeBlock
from utils.utilities import (
    decompose_and_trans_node_attr_to_cell_attr_graph,
    copy_geometric_data,
    NodeType,
    calc_cell_centered_with_node_attr,
    calc_node_centered_with_cell_attr,
)
from torch_geometric.data import Data
import numpy as np
from torch_scatter import scatter_add, scatter_mean
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from torch_geometric.data.batch import Batch


# from torch_geometric.nn import InstanceNorm
def build_mlp(
    in_size, hidden_size, out_size, drop_out=True, lay_norm=True, dropout_prob=0.2
):
    if drop_out:
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size),
        )
    else:
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size),
        )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


def build_mlp_test(
    in_size,
    hidden_size,
    out_size,
    drop_out=False,
    lay_norm=True,
    dropout_prob=0.2,
    specify_hidden_layer_num=2,
):
    layers = []
    layers.append(nn.Linear(in_size, hidden_size))
    if drop_out:
        layers.append(nn.Dropout(p=dropout_prob))
    layers.append(nn.SiLU())

    # Add specified number of hidden layers
    for i in range(specify_hidden_layer_num - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if drop_out:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.SiLU())

    layers.append(nn.Linear(hidden_size, out_size))

    if lay_norm:
        layers.append(nn.LayerNorm(normalized_shape=out_size))

    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(
        self,
        node_input_size=128,
        edge_input_size=128,
        cell_input_size=128,
        hidden_size=128,
        attention=False,
    ):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(
            edge_input_size, hidden_size, int(hidden_size), drop_out=False
        )
        self.nb_encoder = build_mlp(
            node_input_size, hidden_size, int(hidden_size), drop_out=False
        )
        self.attention = attention
        self.scale = torch.sqrt(torch.tensor(hidden_size))

    def forward(self, graph_node, graph_cell):
        (
            node_attr,
            edge_index,
            edge_attr,
            face,
            _,
            _,
        ) = decompose_and_trans_node_attr_to_cell_attr_graph(
            graph_node, has_changed_node_attr_to_cell_attr=False
        )

        # cell_ = self.cb_encoder(cell_attr)*mask_cell_interior.view(-1,1).long()
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        return (
            Data(x=node_, edge_attr=edge_, edge_index=edge_index, face=face),
            edge_,
            node_,
        )


class GnBlock(nn.Module):
    def __init__(self, hidden_size=128, drop_out=False, attention=True, MultiHead=1):
        super(GnBlock, self).__init__()

        eb_input_dim = int(3 * (hidden_size))
        nb_input_dim = int(hidden_size + (hidden_size / 2.0))
        # cb_input_dim = 2 * hidden_size
        # cb_custom_func = build_mlp(cb_input_dim, hidden_size, hidden_size,drop_out=False)
        # self.cb_module = CellBlock(hidden_size,hidden_size,attention=attention,MultiHead=MultiHead,custom_func=cb_custom_func)
        nb_custom_func = build_mlp(
            nb_input_dim, hidden_size, int(hidden_size), drop_out=False
        )
        self.nb_module = NodeBlock(
            hidden_size,
            hidden_size,
            attention=attention,
            MultiHead=MultiHead,
            custom_func=nb_custom_func,
        )
        eb_custom_func = build_mlp(
            eb_input_dim, hidden_size, int(hidden_size), drop_out=False
        )
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)

    def forward(self, graph_node, graph_edge, graph_cell):
        graph_node_last = copy_geometric_data(
            graph_node, has_changed_node_attr_to_cell_attr=True
        )
        # last_cell_attr = graph_last.x

        graph_node = self.eb_module(graph_node, graph_cell)

        graph_node = self.nb_module(graph_node, graph_cell)

        # resdiual connection
        x = graph_node.x + graph_node_last.x
        edge_attr = graph_node.edge_attr + graph_node_last.edge_attr

        return Data(
            x=x,
            edge_attr=edge_attr,
            edge_index=graph_node.edge_index,
            face=graph_node.face,
        )


class Decoder(nn.Module):
    def __init__(
        self,
        edge_hidden_size=128,
        cell_hidden_size=128,
        edge_output_size=3,
        cell_output_size=2,
        cell_input_size=2,
        node_output_size=2,
        attention=False,
    ):
        super(Decoder, self).__init__()

        # self.edge_transform_module = build_mlp_test(4*edge_hidden_size, edge_hidden_size, edge_hidden_size, drop_out=False,lay_norm=True,specify_hidden_layer_num=2)
        # self.edge_decode_module = build_mlp_test(edge_hidden_size, edge_hidden_size, edge_output_size, drop_out=False,lay_norm=False,specify_hidden_layer_num=2)
        # self.node_transform_module = build_mlp_test(3*cell_hidden_size, cell_hidden_size, cell_hidden_size, drop_out=False,lay_norm=True,specify_hidden_layer_num=2)
        self.node_decode_module = build_mlp_test(
            2 * int((cell_hidden_size)),
            cell_hidden_size,
            node_output_size,
            drop_out=False,
            lay_norm=False,
            specify_hidden_layer_num=2,
        )
        # self.attention=attention
        # self.scale=torch.sqrt(torch.tensor(cell_hidden_size))
        # self.cell_decode_module = build_mlp(cell_hidden_size, cell_hidden_size, node_output_size, drop_out=False,lay_norm=False)

    def forward(self, node_embedding=None, latent_graph_node=None):
        node_attr, _, _, _, _, _ = decompose_and_trans_node_attr_to_cell_attr_graph(
            latent_graph_node, has_changed_node_attr_to_cell_attr=True
        )

        node_decode_attr = self.node_decode_module(
            torch.cat((node_attr, node_embedding), dim=1)
        )

        return node_decode_attr


class EncoderProcesserDecoder(nn.Module):
    def __init__(
        self,
        message_passing_num,
        cell_input_size,
        edge_input_size,
        node_input_size,
        cell_output_size,
        edge_output_size,
        node_output_size,
        drop_out,
        hidden_size=128,
        attention=True,
        MultiHead=1,
    ):
        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            cell_input_size=cell_input_size,
            hidden_size=hidden_size,
            attention=attention,
        )
        self.message_passing_num = message_passing_num
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(
                GnBlock(
                    hidden_size=hidden_size,
                    drop_out=drop_out,
                    attention=attention,
                    MultiHead=MultiHead,
                )
            )
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(
            edge_hidden_size=hidden_size,
            cell_hidden_size=hidden_size,
            cell_output_size=cell_output_size,
            edge_output_size=edge_output_size,
            node_output_size=node_output_size,
            attention=attention,
        )

    def curl_a_2_velocity(self, a=None, mesh_pos=None, face_node=None):
        diff_a = a[face_node[0]] - a[face_node[1]]
        diff_edge = mesh_pos[face_node[0]] - mesh_pos[face_node[1]]
        length = torch.norm(diff_edge, dim=1, keepdim=True)
        nabla_a = (diff_a / length) * (diff_edge / length)
        velocity_field_edge = torch.cat((-nabla_a[:, 1:2], nabla_a[:, 0:1]), dim=1)

        return velocity_field_edge

    def inteplot_p_2_cell(self, p=None, face_node=None):
        return (p[face_node[0]] + p[face_node[1]]) / 2.0

    def supress_reverse_flow(self, node_decode_p=None, graph_node=None, p_ref=0.0):
        """TODO: still have some problem, need to be fixed"""
        oundary_outflow_mask = (graph_node.node_type == NodeType.OUTFLOW).squeeze(1)

        node_decode_p[oundary_outflow_mask] = torch.where(
            node_decode_p[oundary_outflow_mask] > p_ref,
            p_ref,
            node_decode_p[oundary_outflow_mask],
        )

        return node_decode_p

    def forward(self, graph_cell, graph_node, graph_edge, params=None):
        latent_graph_node, _, node_embedding = self.encoder(
            graph_node, graph_cell=graph_cell
        )
        # graph_embeded = copy_geometric_data(latent_graph_cell,has_changed_node_attr_to_cell_attr=True)
        # count = self.message_passing_num
        for model in self.processer_list:
            latent_graph_node = model(latent_graph_node, graph_edge, graph_cell)

        # decode latent graph to node attr a and p
        node_decode_attr = self.decoder(
            node_embedding=node_embedding, latent_graph_node=latent_graph_node
        )

        node_decode_uv, node_decode_p = 10 * torch.tanh(
            (node_decode_attr[:, 0:2]) / 10
        ), 10 * torch.tanh((node_decode_attr[:, 2:3]) / 10)

        # for Dirichlet boundary conditions
        boundary_fluid_mask = (
            (graph_node.node_type == NodeType.INFLOW)
            | (graph_node.node_type == NodeType.WALL_BOUNDARY)
            | (graph_node.node_type == NodeType.IN_WALL)
        ).squeeze(1)
        node_decode_uv[boundary_fluid_mask] = graph_node.y[boundary_fluid_mask, 0:2]

        if (not params.integrate_p) and (params.pressure_open_bc <= 0.0):
            boundary_outflow_mask = (graph_node.node_type == NodeType.OUTFLOW).squeeze(
                1
            )
            node_decode_p[boundary_outflow_mask] = 0.0

        # use artifacial wall method to supress reverse flow
        if params.prevent_reverse_flow:
            node_decode_p = self.supress_reverse_flow(
                node_decode_p=node_decode_p, graph_node=graph_node
            )

        # for cavity_flow pressure constraint point condition
        pressure_constraint_point = (graph_node.node_type == NodeType.IN_WALL).squeeze(
            1
        )
        node_decode_p[pressure_constraint_point] = 0.0

        node_decoded_uvp = (
            torch.cat((node_decode_uv, node_decode_p), dim=1)
            * graph_node.neural_network_output_mask
        )

        return node_decoded_uvp


class Intergrator(nn.Module):
    def __init__(self, edge_input_size=7, cell_input_size=2, cell_output_size=2):
        super(Intergrator, self).__init__()
        self.edge_input_size = edge_input_size
        self.cell_input_size = cell_input_size
        self.cell_output_size = cell_output_size
        self.count = 1
        self.plotted = False

    def enforce_boundary_ghost_gradient(
        self,
        nabala_phi_c=None,
        edge_neighbour_index=None,
        cells_type=None,
        face_type=None,
    ):
        # INFLOW
        mask_face_inflow = face_type == NodeType.INFLOW

        edge_neighbour_index_l = edge_neighbour_index[0]
        edge_neighbour_index_r = edge_neighbour_index[1]

        mask_inflow_ghost_cell = (
            cells_type[edge_neighbour_index_l[mask_face_inflow]]
            == NodeType.GHOST_INFLOW
        )
        # dim=0 stands for interior cell, dim=1 stands for ghost cell
        edge_neighbour_index_inflow = edge_neighbour_index[:, mask_face_inflow].clone()
        edge_neighbour_index_inflow[0, mask_inflow_ghost_cell] = edge_neighbour_index_r[
            mask_face_inflow
        ][mask_inflow_ghost_cell]
        edge_neighbour_index_inflow[1, mask_inflow_ghost_cell] = edge_neighbour_index_l[
            mask_face_inflow
        ][mask_inflow_ghost_cell]

        # ghost cell gradient equal zero at inflow
        nabala_phi_c[edge_neighbour_index_inflow[1]] = (
            nabala_phi_c[edge_neighbour_index_inflow[0]].clone().detach()
        )

        # WALL BOUNDARY
        mask_face_wall = face_type == NodeType.WALL_BOUNDARY
        mask_wall_ghost_cell = (
            cells_type[edge_neighbour_index_l[mask_face_wall]] == NodeType.GHOST_WALL
        )
        # dim=0 stands for interior cell, dim=1 stands for ghost cell
        edge_neighbour_index_wall = edge_neighbour_index[:, mask_face_wall].clone()
        edge_neighbour_index_wall[0, mask_wall_ghost_cell] = edge_neighbour_index_r[
            mask_face_wall
        ][mask_wall_ghost_cell]
        edge_neighbour_index_wall[1, mask_wall_ghost_cell] = edge_neighbour_index_l[
            mask_face_wall
        ][mask_wall_ghost_cell]

        # ghost cell gradient=interior cell gradient
        nabala_phi_c[edge_neighbour_index_wall[1]] = (
            nabala_phi_c[edge_neighbour_index_wall[0]].clone().detach()
        )

        # OUTFLOW
        mask_face_outflow = face_type == NodeType.OUTFLOW
        mask_outflow_ghost_cell = (
            cells_type[edge_neighbour_index_l[mask_face_outflow]]
            == NodeType.GHOST_OUTFLOW
        )
        # dim=0 stands for interior cell, dim=1 stands for ghost cell
        edge_neighbour_index_outflow = edge_neighbour_index[
            :, mask_face_outflow
        ].clone()
        edge_neighbour_index_outflow[
            0, mask_outflow_ghost_cell
        ] = edge_neighbour_index_r[mask_face_outflow][mask_outflow_ghost_cell]
        edge_neighbour_index_outflow[
            1, mask_outflow_ghost_cell
        ] = edge_neighbour_index_l[mask_face_outflow][mask_outflow_ghost_cell]

        # liner interpoltion
        nabala_phi_c[edge_neighbour_index_outflow[1]] = (
            nabala_phi_c[edge_neighbour_index_outflow[0]].clone().detach()
        )

        return nabala_phi_c

    def intergre_f2c_2d(
        self,
        phi_f,
        cells_face,
        edge_Euclidean_distance,
        cell_area,
        unv,
        edge_neighbour_index,
        cells_type,
        face_type,
        include_pressure=False,
    ):
        mask_cell_interior = (
            (cells_type.view(-1, 1) == NodeType.NORMAL)
            | (cells_type.view(-1, 1) == NodeType.INFLOW)
            | (cells_type.view(-1, 1) == NodeType.WALL_BOUNDARY)
            | (cells_type.view(-1, 1) == NodeType.OUTFLOW)
        ).long()
        if include_pressure:
            """nabala_phi_c should be 5 dim at dim1, phi_f dim=3"""
            nabala_phi_c = (
                (1.0 / cell_area)
                * (
                    self.chain_element_wise_vector_product_up_three(
                        torch.index_select(phi_f, 0, cells_face[0]),
                        unv[:, 0, :]
                        * torch.index_select(edge_Euclidean_distance, 0, cells_face[0]),
                    )
                    + self.chain_element_wise_vector_product_up_three(
                        torch.index_select(phi_f, 0, cells_face[1]),
                        unv[:, 1, :]
                        * torch.index_select(edge_Euclidean_distance, 0, cells_face[1]),
                    )
                    + self.chain_element_wise_vector_product_up_three(
                        torch.index_select(phi_f, 0, cells_face[2]),
                        unv[:, 2, :]
                        * torch.index_select(edge_Euclidean_distance, 0, cells_face[2]),
                    )
                )
            ) * mask_cell_interior

        else:
            """nabala_phi_c should be 4 dim at dim1"""
            nabala_phi_c = (
                (1.0 / cell_area)
                * (
                    self.chain_element_wise_vector_product_up(
                        torch.index_select(phi_f, 0, cells_face[0]),
                        unv[:, 0, :]
                        * torch.index_select(edge_Euclidean_distance, 0, cells_face[0]),
                    )
                    + self.chain_element_wise_vector_product_up(
                        torch.index_select(phi_f, 0, cells_face[1]),
                        unv[:, 1, :]
                        * torch.index_select(edge_Euclidean_distance, 0, cells_face[1]),
                    )
                    + self.chain_element_wise_vector_product_up(
                        torch.index_select(phi_f, 0, cells_face[2]),
                        unv[:, 2, :]
                        * torch.index_select(edge_Euclidean_distance, 0, cells_face[2]),
                    )
                )
            ) * mask_cell_interior

        nabala_phi_c = self.enforce_boundary_ghost_gradient(
            nabala_phi_c=nabala_phi_c,
            edge_neighbour_index=edge_neighbour_index,
            cells_type=cells_type,
            face_type=face_type,
        )

        return nabala_phi_c

    def intergre_f2c_1d(
        self,
        phi_f,
        cells_face,
        edge_Euclidean_distance,
        cell_area,
        unv,
        edge_neighbour_index,
        cells_type,
        face_type,
    ):
        """nabala_phi_c should be 4 dim at dim1"""
        nabala_phi_c = (1.0 / cell_area) * (
            phi_f[cells_face[0]] * unv[:, 0, :] * edge_Euclidean_distance[cells_face[0]]
            + phi_f[cells_face[1]]
            * unv[:, 1, :]
            * edge_Euclidean_distance[cells_face[1]]
            + phi_f[cells_face[2]]
            * unv[:, 2, :]
            * edge_Euclidean_distance[cells_face[1]]
        )

        # nabala_phi_c = self.enforce_boundary_ghost_gradient(nabala_phi_c=nabala_phi_c,
        #                                                     edge_neighbour_index=edge_neighbour_index,
        #                                                     cells_type=cells_type,
        #                                                     face_type=face_type)

        return nabala_phi_c

    def interpolating_gradients_to_faces(
        self,
        nabala_phi_c=None,
        phi_cell_convection_outer=None,
        phi_cell_convection_inner=None,
        out_centroid=None,
        in_centroid=None,
        edge_neighbour_index=None,
        edge_center_pos=None,
        cells_face=None,
    ):
        # edge_neighbour_index[0] is C,edge_neighbour_index[1] is F

        phi_cell_convection_outer_single = torch.chunk(
            phi_cell_convection_outer, 2, dim=0
        )[0]
        phi_cell_convection_inner_single = torch.chunk(
            phi_cell_convection_inner, 2, dim=0
        )[0]

        out_centroid_single = torch.chunk(out_centroid, 2, dim=0)[0]
        in_centroid_single = torch.chunk(in_centroid, 2, dim=0)[0]

        g_c = (
            torch.norm((out_centroid_single - edge_center_pos), dim=1)
            / torch.norm((in_centroid_single - out_centroid_single), dim=1)
        ).view(-1, 1)
        g_c = torch.where(torch.isfinite(g_c), g_c, torch.full_like(g_c, 0.5))
        g_f = 1.0 - g_c

        nabala_phi_f_hat = (
            torch.index_select(nabala_phi_c, 0, edge_neighbour_index[0]) * g_c
            + torch.index_select(nabala_phi_c, 0, edge_neighbour_index[1]) * g_f
        )

        # substitude zero vector to unit vector
        vector_CF = out_centroid_single - in_centroid_single

        d_CF = torch.norm(vector_CF, dim=1, keepdim=True)
        d_CF = torch.where(d_CF == 0, torch.full_like(d_CF, torch.mean(d_CF)), d_CF)

        e_CF = vector_CF / d_CF
        # e_CF = torch.where(torch.isnan(e_CF), torch.full_like(e_CF, 1), e_CF)
        e_CF = torch.where(
            torch.isfinite(e_CF),
            e_CF,
            torch.mean(d_CF, dim=1, keepdim=True).repeat(1, e_CF.shape[1]),
        )

        if nabala_phi_c.shape[-1] == 4:
            nabala_phi_f = nabala_phi_f_hat + self.chain_element_wise_vector_product_up(
                (
                    (
                        phi_cell_convection_outer_single
                        - phi_cell_convection_inner_single
                    )
                    / d_CF
                    - self.chain_vector_dot_product(nabala_phi_f_hat, e_CF)
                ),
                e_CF,
            )

        elif nabala_phi_c.shape[-1] == 2:
            nabala_phi_f = (
                nabala_phi_f_hat
                + (
                    (
                        phi_cell_convection_outer_single
                        - phi_cell_convection_inner_single
                    )
                    / d_CF
                    - self.chain_dot_product(nabala_phi_f_hat, e_CF)
                )
                * e_CF
            )

        return nabala_phi_f

    def correct_boundary_phic_2_faces(
        self,
        phi_f=None,
        phi_c=None,
        pressure_f=None,
        edge_neighbour_index=None,
        cells_type=None,
        face_type=None,
        edge_center_pos=None,
        centroid=None,
    ):
        # INFLOW
        mask_face_inflow = face_type == NodeType.INFLOW

        edge_neighbour_index_l = edge_neighbour_index[0]
        edge_neighbour_index_r = edge_neighbour_index[1]

        mask_inflow_ghost_cell = (
            cells_type[edge_neighbour_index_l[mask_face_inflow]]
            == NodeType.GHOST_INFLOW
        )
        # dim=0 stands for interior cell, dim=1 stands for ghost cell
        edge_neighbour_index_inflow = edge_neighbour_index[:, mask_face_inflow].clone()
        edge_neighbour_index_inflow[0, mask_inflow_ghost_cell] = edge_neighbour_index_r[
            mask_face_inflow
        ][mask_inflow_ghost_cell]
        edge_neighbour_index_inflow[1, mask_inflow_ghost_cell] = edge_neighbour_index_l[
            mask_face_inflow
        ][mask_inflow_ghost_cell]

        # constant padding at inflow boundary
        phi_f[mask_face_inflow, 0:2] = (2 - 0.999) * phi_c[
            edge_neighbour_index_inflow[0], 0:2
        ]
        pressure_f[mask_face_inflow, 0:1] = (2 - 0.999) * phi_c[
            edge_neighbour_index_inflow[0], 2:3
        ]
        ruler = torch.mean(
            torch.norm(
                edge_center_pos[mask_face_inflow, :]
                - centroid[edge_neighbour_index_inflow[0], :],
                dim=1,
            )
        )

        # WALL BOUNDARY
        mask_face_wall = face_type == NodeType.WALL_BOUNDARY
        mask_wall_ghost_cell = (
            cells_type[edge_neighbour_index_l[mask_face_wall]] == NodeType.GHOST_WALL
        )
        # dim=0 stands for interior cell, dim=1 stands for ghost cell
        edge_neighbour_index_wall = edge_neighbour_index[:, mask_face_wall].clone()
        edge_neighbour_index_wall[0, mask_wall_ghost_cell] = edge_neighbour_index_r[
            mask_face_wall
        ][mask_wall_ghost_cell]
        edge_neighbour_index_wall[1, mask_wall_ghost_cell] = edge_neighbour_index_l[
            mask_face_wall
        ][mask_wall_ghost_cell]

        # inverse interior flux to ghost cell at wall boundary
        scale_factor = (
            torch.norm(
                edge_center_pos[mask_face_wall, :]
                - centroid[edge_neighbour_index_wall[0], :],
                dim=1,
            )
            / ruler
        ).view(-1, 1)
        phi_f[mask_face_wall, 0:2] = (
            scale_factor * phi_c[edge_neighbour_index_wall[0], 0:2]
        )
        pressure_f[mask_face_wall, 0:1] = (2 - 0.999) * phi_c[
            edge_neighbour_index_wall[0], 2:3
        ]

        return torch.cat((phi_f, pressure_f), dim=1)

    def interpolating_phic_to_faces(
        self,
        phi_c=None,
        edge_neighbour_index=None,
        cells_face=None,
        edge_Euclidean_distance=None,
        cell_area=None,
        unv=None,
        cells_type=None,
        face_type=None,
        edge_center_pos=None,
        centroid=None,
        include_pressure=False,
    ):
        # mask_face_interior = ((face_type==NodeType.NORMAL)|(face_type==NodeType.OUTFLOW)|(face_type==NodeType.WALL_BOUNDARY)|(face_type==NodeType.INFLOW)).view(-1,1).long()

        total_area = (
            cell_area[edge_neighbour_index[0]] + cell_area[edge_neighbour_index[1]]
        )

        phi_f_hat = torch.index_select(phi_c, 0, edge_neighbour_index[0]) * (
            cell_area[edge_neighbour_index[0]] / total_area
        ) + torch.index_select(phi_c, 0, edge_neighbour_index[1]) * (
            cell_area[edge_neighbour_index[1]] / total_area
        )

        nabala_phi_c = self.intergre_f2c_2d(
            phi_f=phi_f_hat,
            cells_face=cells_face,
            edge_Euclidean_distance=edge_Euclidean_distance,
            cell_area=cell_area,
            unv=unv,
            edge_neighbour_index=edge_neighbour_index,
            cells_type=cells_type,
            face_type=face_type,
            include_pressure=True,
        )

        phi_f = phi_f_hat + 0.5 * self.chain_flux_dot_product_up_three(
            (
                nabala_phi_c[edge_neighbour_index[0]]
                + nabala_phi_c[edge_neighbour_index[1]]
            ),
            (
                edge_center_pos
                - 0.5
                * (
                    centroid[edge_neighbour_index[0]]
                    + centroid[edge_neighbour_index[1]]
                )
            ),
        )

        for _ in range(10):
            nabala_phi_c = self.intergre_f2c_2d(
                phi_f=phi_f,
                cells_face=cells_face,
                edge_Euclidean_distance=edge_Euclidean_distance,
                cell_area=cell_area,
                unv=unv,
                edge_neighbour_index=edge_neighbour_index,
                cells_type=cells_type,
                face_type=face_type,
                include_pressure=True,
            )

            phi_f = phi_f_hat + 0.5 * self.chain_flux_dot_product_up_three(
                (
                    nabala_phi_c[edge_neighbour_index[0]]
                    + nabala_phi_c[edge_neighbour_index[1]]
                ),
                (
                    edge_center_pos
                    - 0.5
                    * (
                        centroid[edge_neighbour_index[0]]
                        + centroid[edge_neighbour_index[1]]
                    )
                ),
            )

        return phi_f

    def interpolating_face_uv_to_cell(self, uvp_edge=None, cells_face=None):
        return (
            uvp_edge[cells_face[0]] + uvp_edge[cells_face[1]] + uvp_edge[cells_face[2]]
        ) / 3.0

    def interpolating_phic_to_faces_upwind(
        self,
        phi_c=None,
        phi_f_hat=None,
        unv=None,
        edge_neighbour_index=None,
        cells_face=None,
        centroid=None,
    ):
        cell_face_0 = edge_neighbour_index[:, cells_face[0]].T  # (2,cells_number)
        cell_face_1 = edge_neighbour_index[:, cells_face[1]].T
        cell_face_2 = edge_neighbour_index[:, cells_face[2]].T

        cell_face_0_U = phi_f_hat[cells_face[0]]
        cell_face_1_U = phi_f_hat[cells_face[1]]
        cell_face_2_U = phi_f_hat[cells_face[2]]

        # cell_three_face_0[:,1] is outside, cell_three_face_0[:,0] is inside
        cell_three_face_0 = torch.where(
            (
                self.chain_dot_product(
                    (centroid[cell_face_0[:, 1]] - centroid[cell_face_0[:, 0]]),
                    unv[:, 0, :],
                )
                > 0
            ).repeat(1, 2),
            torch.cat((cell_face_0[:, 0:1], cell_face_0[:, 1:2]), dim=1),
            torch.cat((cell_face_0[:, 1:2], cell_face_0[:, 0:1]), dim=1),
        )

        cell_three_face_1 = torch.where(
            (
                self.chain_dot_product(
                    (centroid[cell_face_1[:, 1]] - centroid[cell_face_1[:, 0]]),
                    unv[:, 0, :],
                )
                > 0
            ).repeat(1, 2),
            torch.cat((cell_face_1[:, 0:1], cell_face_1[:, 1:2]), dim=1),
            torch.cat((cell_face_1[:, 1:2], cell_face_1[:, 0:1]), dim=1),
        )

        cell_three_face_2 = torch.where(
            (
                self.chain_dot_product(
                    (centroid[cell_face_2[:, 1]] - centroid[cell_face_2[:, 0]]),
                    unv[:, 0, :],
                )
                > 0
            ).repeat(1, 2),
            torch.cat((cell_face_2[:, 0:1], cell_face_2[:, 1:2]), dim=1),
            torch.cat((cell_face_2[:, 1:2], cell_face_2[:, 0:1]), dim=1),
        )

        cell_phi_f_0_adv = torch.where(
            self.chain_dot_product(cell_face_0_U, unv[:, 0, :]) > 0,
            phi_c[cell_three_face_0[:, 0]],
            phi_c[cell_three_face_0[:, 1]],
        )
        cell_phi_f_1_adv = torch.where(
            self.chain_dot_product(cell_face_1_U, unv[:, 1, :]) > 0,
            phi_c[cell_three_face_1[:, 0]],
            phi_c[cell_three_face_1[:, 1]],
        )
        cell_phi_f_2_adv = torch.where(
            self.chain_dot_product(cell_face_2_U, unv[:, 2, :]) > 0,
            phi_c[cell_three_face_2[:, 0]],
            phi_c[cell_three_face_2[:, 1]],
        )

        return torch.stack(
            (cell_phi_f_0_adv, cell_phi_f_1_adv, cell_phi_f_2_adv), dim=1
        )

    def interpolating_phic_to_faces_upwind_test(
        self,
        uv_cell_hat=None,
        uv_face_hat=None,
        edge_neighbour_index=None,
        centroid=None,
    ):
        centroid_vec = (
            centroid[edge_neighbour_index[1]] - centroid[edge_neighbour_index[0]]
        )

        # x_axis = (torch.tensor([1.,0.]).to(centroid_vec.device)).unsqueeze(0)
        # y_axis = torch.tensor([0.,1.]).to(centroid_vec.device)
        mask = (
            torch.sum(uv_face_hat * centroid_vec, dim=1, keepdim=True) > 0.0
        ).repeat(1, 2)
        # mask_x = torch.sum(centroid_vec*x_axis,dim=1,keepdim=True)
        # mask_y = torch.sum(centroid_vec*y_axis,dim=1,keepdim=True)

        uv_edge_hat_upwind = torch.where(
            mask > 0,
            uv_cell_hat[edge_neighbour_index[0], :],
            uv_cell_hat[edge_neighbour_index[1], :],
        )
        # v_edge = torch.where(mask>0,uv_cell[edge_neighbour_index[0],1:2],uv_cell[edge_neighbour_index[1],1:2])

        return uv_edge_hat_upwind

    def interpolating_phic_to_faces_upwind_staggered(
        self, uv_cell=None, edge_neighbour_index=None, centroid=None
    ):
        centroid_vec = (
            centroid[edge_neighbour_index[1]] - centroid[edge_neighbour_index[0]]
        )

        x_axis = torch.tensor([1.0, 0.0]).to(centroid_vec.device)
        y_axis = torch.tensor([0.0, 1.0]).to(centroid_vec.device)

        mask_x = torch.sum(centroid_vec * x_axis, dim=1, keepdim=True)
        mask_y = torch.sum(centroid_vec * y_axis, dim=1, keepdim=True)

        u_edge = torch.where(
            mask_x > 0,
            uv_cell[edge_neighbour_index[0], 0:1],
            uv_cell[edge_neighbour_index[1], 0:1],
        )
        v_edge = torch.where(
            mask_y > 0,
            uv_cell[edge_neighbour_index[0], 1:2],
            uv_cell[edge_neighbour_index[1], 1:2],
        )

        return torch.cat((u_edge, v_edge), dim=1)

    def rhie_chow_interpolation(
        self,
        uv_edge=None,
        p_cell=None,
        nabla_p_cell=None,
        unv=None,
        edge_neighbour_index=None,
        centroid=None,
        edge_center_pos=None,
    ):
        g_c = (
            torch.norm(
                torch.index_select(centroid, 0, edge_neighbour_index[1])
                - edge_center_pos,
                dim=1,
            )
            / torch.norm(
                torch.index_select(centroid, 0, edge_neighbour_index[1])
                - torch.index_select(centroid, 0, edge_neighbour_index[0]),
                dim=1,
            )
        ).view(-1, 1)
        g_c = torch.where(torch.isnan(g_c), torch.full_like(g_c, 0.5), g_c)
        g_c = torch.where(torch.isinf(g_c), torch.full_like(g_c, 0.5), g_c)

        # g_f =  (torch.norm(torch.index_select(centroid,0,edge_neighbour_index[0])-edge_center_pos,dim=1)/torch.norm(torch.index_select(centroid,0,edge_neighbour_index[0])-torch.index_select(centroid,0,edge_neighbour_index[1]),dim=1)).view(-1,1)

        # g_f = torch.where(torch.isnan(g_f), torch.full_like(g_f, 0.5), g_f)
        # g_f = torch.where(torch.isinf(g_f), torch.full_like(g_f, 0.5), g_f)

        g_f = 1.0 - g_c

        nabala_p_f_hat = (
            torch.index_select(nabla_p_cell, 0, edge_neighbour_index[0]) * g_c
            + torch.index_select(nabla_p_cell, 0, edge_neighbour_index[1]) * g_f
        )

        # substitude zero vector to unit vector
        vector_CF = torch.index_select(
            centroid, 0, edge_neighbour_index[1]
        ) - torch.index_select(centroid, 0, edge_neighbour_index[0])

        d_CF = torch.norm(vector_CF, dim=1, keepdim=True)
        d_CF = torch.where(d_CF == 0, torch.full_like(d_CF, 1), d_CF)

        e_CF = vector_CF / d_CF
        e_CF = torch.where(torch.isnan(e_CF), torch.full_like(e_CF, 1), e_CF)
        # e_CF = torch.where(torch.isfinite(e_CF), e_CF, torch.full_like(e_CF, 1))

        uv_edge = (
            uv_edge
            + (
                (
                    torch.index_select(p_cell, 0, edge_neighbour_index[1])
                    - torch.index_select(p_cell, 0, edge_neighbour_index[0])
                )
                / d_CF
                - self.chain_dot_product(nabala_p_f_hat, e_CF)
            )
            * e_CF
        )

        return uv_edge

    def update_Green_Gauss_Gradient(
        self,
        phi_f=None,
        phi_c=None,
        edge_neighbour_index=None,
        cells_face=None,
        cell_area=None,
        unv=None,
        edge_Euclidean_distance=None,
        edge_center_pos=None,
        centroid=None,
        mask_cell_interior=None,
        bc=None,
        cells_type=None,
        face_type=None,
        graph_cell=None,
        graph_node=None,
    ):
        # nabala_phi_c = self.intergre_f2c_2d(phi_f=phi_f,
        #                                 cells_face=cells_face,
        #                                 edge_Euclidean_distance=edge_Euclidean_distance,
        #                                 cell_area=cell_area,
        #                                 unv=unv,
        #                                 edge_neighbour_index=edge_neighbour_index,
        #                                 cells_type=cells_type,
        #                                 face_type=face_type,
        #                                 include_pressure=False)

        nabala_cell_u = self.intergre_f2c_1d(
            phi_f=phi_f[:, 0:1],
            cells_face=cells_face,
            edge_Euclidean_distance=edge_Euclidean_distance,
            cell_area=cell_area,
            unv=unv,
            edge_neighbour_index=edge_neighbour_index,
            cells_type=cells_type,
            face_type=face_type,
        )

        nabala_cell_v = self.intergre_f2c_1d(
            phi_f=phi_f[:, 1:2],
            cells_face=cells_face,
            edge_Euclidean_distance=edge_Euclidean_distance,
            cell_area=cell_area,
            unv=unv,
            edge_neighbour_index=edge_neighbour_index,
            cells_type=cells_type,
            face_type=face_type,
        )

        nabala_phi_c = torch.cat((nabala_cell_u, nabala_cell_v), dim=1)

        nabala_phi_f = self.interpolating_gradients_to_faces(
            nabala_phi_c=nabala_phi_c,
            phi_c=phi_c,
            unv=unv,
            centroid=centroid,
            edge_neighbour_index=edge_neighbour_index,
            edge_center_pos=edge_center_pos,
            cells_face=cells_face,
        )

        # nabala_phi_f = self.interpolating_gradients_to_faces_test(nabala_phi_c=torch.cat((nabala_cell_u,nabala_cell_v),dim=1),
        #                                                     cells_node=graph_node.face,
        #                                                     num_nodes=graph_node.num_nodes,
        #                                                     face_node=graph_node.edge_index,
        #                                                     centroid=graph_cell.pos,
        #                                                     mesh_pos=graph_node.pos)

        return nabala_phi_f, nabala_phi_c
    
    def compute_Green_Gauss_Gradient_node_based(
        self,
        phi_node=None,
        graph_node=None,
        graph_edge=None,
        graph_cell=None,
    ):
        
        cells_node = cells_node = graph_node.face[0]
        cells_node_surface_vector = graph_cell.cells_node_surface_vector
        cells_index = graph_cell.face[0]
        cell_area = graph_cell.cell_area
        
        node_flux = phi_node[cells_node, :]*cells_node_surface_vector

        nabla_phi_cell = calc_cell_centered_with_node_attr(
            node_attr=node_flux,
            cells_node=cells_node,
            cells_index=cells_index,
            reduce="sum",
            map=False,
        )/cell_area.view(-1,1)

        nabla_phi_node = self.interpolating_phi_c_to_node(
            phi_c=nabla_phi_cell,
            cells_node=cells_node,
            centroid=graph_cell.pos,
            mesh_pos=graph_node.pos,
            cells_index=cells_index,
        )
        
        return nabla_phi_node,nabla_phi_cell

    def compute_cell_based_gradient_least_squares(
        self,
        phi_u,
        phi_v,
        centroid,
        edge_neighbour_index,
        cells_face,
        unv,
        mask_cell_interior,
        cells_type,
        face_type,
    ):
        cell_face_0 = (edge_neighbour_index[:, cells_face[0]]).T  # (2,cells_number)
        cell_face_1 = (edge_neighbour_index[:, cells_face[1]]).T
        cell_face_2 = (edge_neighbour_index[:, cells_face[2]]).T

        # cell_three_face_0[:,1] is outside, cell_three_face_0[:,0] is inside
        cell_three_face_0 = torch.where(
            (
                self.chain_dot_product(
                    (centroid[cell_face_0[:, 1]] - centroid[cell_face_0[:, 0]]),
                    unv[:, 0, :],
                )
                > 0
            ).repeat(1, 2),
            torch.cat((cell_face_0[:, 0:1], cell_face_0[:, 1:2]), dim=1),
            torch.cat((cell_face_0[:, 1:2], cell_face_0[:, 0:1]), dim=1),
        )

        cell_three_face_1 = torch.where(
            (
                self.chain_dot_product(
                    (centroid[cell_face_1[:, 1]] - centroid[cell_face_1[:, 0]]),
                    unv[:, 0, :],
                )
                > 0
            ).repeat(1, 2),
            torch.cat((cell_face_1[:, 0:1], cell_face_1[:, 1:2]), dim=1),
            torch.cat((cell_face_1[:, 1:2], cell_face_1[:, 0:1]), dim=1),
        )

        cell_three_face_2 = torch.where(
            (
                self.chain_dot_product(
                    (centroid[cell_face_2[:, 1]] - centroid[cell_face_2[:, 0]]),
                    unv[:, 0, :],
                )
                > 0
            ).repeat(1, 2),
            torch.cat((cell_face_2[:, 0:1], cell_face_2[:, 1:2]), dim=1),
            torch.cat((cell_face_2[:, 1:2], cell_face_2[:, 0:1]), dim=1),
        )

        cell_three_face_neighbour = torch.stack(
            (cell_three_face_0, cell_three_face_1, cell_three_face_2), dim=1
        )

        # n_cells = mesh_cells.T.shape[0]

        # a = torch.tensor([[1,2],[2,3],[4,5]])
        # calc centroid distance between cells
        d_center_points_diff = (
            centroid[cell_three_face_neighbour[:, :, 1]]
            - centroid[cell_three_face_neighbour[:, :, 0]]
        )
        dist_cell = torch.norm(d_center_points_diff, dim=2, keepdim=True)
        dist_cell_new = torch.where(
            (dist_cell != 0),
            dist_cell,
            torch.full_like(dist_cell, torch.mean(dist_cell)),
        )
        # dist_cell_new = torch.where((dist_cell!=0),dist_cell,torch.full(dist_cell.shape,torch.mean(dist_cell)).cuda())
        # dist_cell = torch.where(dist_cell==0,dist_cell,torch.full(dist_cell.shape,torch.mean(dist_cell)).cuda())
        # cell_volumes = cell_area

        # 构建线性系统 Ax = b
        d_center_points_diff = d_center_points_diff.unsqueeze(3)
        d_center_points_diff_T = d_center_points_diff.transpose(2, 3)

        weight = 1.0 / dist_cell_new

        A = torch.sum(
            torch.matmul(d_center_points_diff, d_center_points_diff_T)
            * (weight.unsqueeze(3)),
            dim=1,
        )

        B_u = torch.sum(
            weight
            * d_center_points_diff.squeeze(3)
            * (
                phi_u[cell_three_face_neighbour[:, :, 1]]
                - phi_u[cell_three_face_neighbour[:, :, 0]]
            ),
            dim=1,
        )

        B_v = torch.sum(
            weight
            * d_center_points_diff.squeeze(3)
            * (
                phi_v[cell_three_face_neighbour[:, :, 1]]
                - phi_v[cell_three_face_neighbour[:, :, 0]]
            ),
            dim=1,
        )

        solution_u = torch.zeros_like(A)[:, 0, :]

        solution_v = torch.zeros_like(A)[:, 0, :]
        # 使用最小二乘法求解线性系统
        # nabla_phi = torch.linalg.lstsq(A[mask_cell_interior.view(-1)], b.unsqueeze(2)[mask_cell_interior.view(-1)])
        # nabla_phi = torch.linalg.pinv(A[mask_cell_interior.view(-1)]) @ b.unsqueeze(2)[mask_cell_interior.view(-1)]
        # solution[mask_cell_interior.view(-1)] = nabla_phi[:,:,0]
        solution_u[mask_cell_interior.view(-1), :] = torch.linalg.lstsq(
            A[mask_cell_interior.view(-1)],
            B_u.unsqueeze(2)[mask_cell_interior.view(-1)],
        ).solution[:, :, 0]

        solution_v[mask_cell_interior.view(-1), :] = torch.linalg.lstsq(
            A[mask_cell_interior.view(-1)],
            B_v.unsqueeze(2)[mask_cell_interior.view(-1)],
        ).solution[:, :, 0]

        # nabla_phi = torch.linalg.pinv(A) @ b.unsqueeze(2)
        nabla_phi = torch.cat((solution_u, solution_v), dim=1)

        # mask_inflow_ghost_cell = cells_type==NodeType.GHOST_INFLOW
        # mask_wall_ghost_cell = cells_type==NodeType.GHOST_WALL
        # mask_outflow_ghost_cell = cells_type==NodeType.GHOST_OUTFLOW
        nabala_phi_c = self.enforce_boundary_ghost_gradient(
            nabala_phi_c=nabla_phi,
            edge_neighbour_index=edge_neighbour_index,
            cells_type=cells_type,
            face_type=face_type,
        )

        return nabala_phi_c

    def calc_symmetric_ghost_pos(
        self,
        boundary_edge_pos_left=None,
        boundary_edge_pos_right=None,
        interior_centroid=None,
    ):
        """boundary_edge_pos:[num_edges,2,2]"""

        # Unpack the edge positions for clarity
        x1, y1 = boundary_edge_pos_left[:, 0], boundary_edge_pos_left[:, 1]
        x2, y2 = boundary_edge_pos_right[:, 0], boundary_edge_pos_right[:, 1]

        # Unpack the vertex positions for clarity
        x3, y3 = interior_centroid[:, 0], interior_centroid[:, 1]

        # Calculate vectors AB and AC
        AB = torch.stack((x2 - x1, y2 - y1), dim=1)
        AC = torch.stack((x3 - x1, y3 - y1), dim=1)

        # Calculate the projection of AC onto AB
        proj_len = (AC * AB).sum(dim=1, keepdim=True) / (AB * AB).sum(
            dim=1, keepdim=True
        )
        AP = AB * proj_len

        # Calculate the coordinates of the projection point P
        P = torch.stack((x1, y1), dim=1) + AP

        # Calculate the coordinates of the symmetric point D
        D = 2 * P - torch.stack((x3, y3), dim=1)

        return D

    def enfoce_boundary_conditions(
        self,
        phi_cell=None,
        indegree_cell=None,
        outdegree_cell=None,
        face_type=None,
        graph_edge=None,
        is_pressure=False,
        dir=None,
    ):
        face_type = face_type.view(-1)

        phi_inner = phi_cell[indegree_cell]

        phi_outer = phi_cell[outdegree_cell]

        phi_diff_on_edge = phi_outer - phi_inner

        return phi_diff_on_edge, phi_outer, phi_inner

    def cell_based_WLSQ(
        self,
        phi_node=None,
        phi_cell=None,
        graph_node=None,
        graph_edge=None,
        graph_cell=None,
        graph_cell_x=None,
        inteploting=False,
        pressure_only=False,
    ):
        cells_node = graph_node.face[0]
        cells_index = graph_cell.face[0]

        outdegree_cell = torch.cat(
            (graph_cell.edge_index[0], graph_cell.edge_index[1]), dim=0
        )
        indegree_cell = torch.cat(
            (graph_cell.edge_index[1], graph_cell.edge_index[0]), dim=0
        )

        outdegree_cell_x = torch.cat(
            (graph_cell_x.edge_index[0], graph_cell_x.edge_index[1]), dim=0
        )
        indegree_cell_x = torch.cat(
            (graph_cell_x.edge_index[1], graph_cell_x.edge_index[0]), dim=0
        )

        twoway_face_node = torch.cat(
            (graph_node.edge_index, graph_node.edge_index.flip(0)), dim=-1
        )

        mask_x = (indegree_cell_x == outdegree_cell_x).view(-1)
        mask_interior = torch.logical_not(mask_x)

        # if not "B_cell_to_cell_x" in graph_cell_x.keys():
        if True:
            out_centroid_x = graph_cell.pos[outdegree_cell_x].clone()
            in_centroid_x = graph_cell.pos[indegree_cell_x].clone()

            """>>> boundary face centroid >>>"""

            # out_centroid_x[mask_x]=graph_edge.pos[mask_x]
            # out_centroid[mask]=symmetry_pos
            """<<< boundary face centroid <<<"""

            """>>>      cell to cell contributation       >>>"""
            # A matrix
            centroid_diff_on_edge = ((out_centroid_x - in_centroid_x)[mask_interior]).unsqueeze(2)

            centroid_diff_on_edge_T = centroid_diff_on_edge.transpose(1, 2)

            weight_cell_to_cell_x = 1.0 / torch.norm(
                centroid_diff_on_edge, dim=1, keepdim=True
            )

            left_on_edge_cell_to_cell = (
                torch.matmul(
                    centroid_diff_on_edge* weight_cell_to_cell_x, 
                    centroid_diff_on_edge_T* weight_cell_to_cell_x)
            )

            A_cell_to_cell_x = scatter_add(
                left_on_edge_cell_to_cell,
                indegree_cell_x[mask_interior],
                dim=0,
                dim_size=graph_cell.pos.shape[0],
            )

            # B matrix
            (
                phi_diff_cell_to_cell,
                phi_cell_outer,
                phi_cell_inner,
            ) = self.enfoce_boundary_conditions(
                phi_cell=phi_cell,
                indegree_cell=indegree_cell_x,
                outdegree_cell=outdegree_cell_x,
                face_type=graph_edge.x[:, 0:1],
                is_pressure=pressure_only,
                graph_edge=graph_edge,
            )

            phi_diff_cell_to_cell_edge = (
                weight_cell_to_cell_x**2
                * phi_diff_cell_to_cell[mask_interior].unsqueeze(1)
                * centroid_diff_on_edge
            )

            B_phi_cell_to_cell = scatter_add(
                phi_diff_cell_to_cell_edge,
                indegree_cell_x[mask_interior],
                dim=0,
                dim_size=graph_cell.pos.shape[0],
            )
            """<<<      cell to cell contributation       <<<"""

            """>>>      node to cell contributation       >>>"""
            # A matrix
            node_to_centroid_pos_diff = (
                graph_node.pos[cells_node] - graph_cell.pos[cells_index]
            ).unsqueeze(2)
            node_to_centroid_pos_diff_T = node_to_centroid_pos_diff.transpose(1, 2)

            weight_node_to_centroid = 1.0 / torch.norm(
                node_to_centroid_pos_diff, dim=1, keepdim=True
            )

            left_on_edge_node_to_centroid = (
                torch.matmul(
                    node_to_centroid_pos_diff* weight_node_to_centroid, 
                    node_to_centroid_pos_diff_T* weight_node_to_centroid
                    )
            )

            A_node_to_cell = calc_cell_centered_with_node_attr(
                node_attr=left_on_edge_node_to_centroid,
                cells_node=cells_node,
                cells_index=cells_index,
                reduce="sum",
                map=False,
            )

            # B matrix
            phi_diff_node_to_cell = (
                weight_node_to_centroid**2
                * (phi_node[cells_node] - phi_cell[cells_index]).unsqueeze(1)
                * node_to_centroid_pos_diff
            )

            B_phi_node_to_cell = calc_cell_centered_with_node_attr(
                node_attr=phi_diff_node_to_cell,
                cells_node=cells_node,
                cells_index=cells_index,
                reduce="sum",
                map=False,
            )
            """<<<      node to cell contributation       <<<"""
            # A_cell_to_cell_x += A_node_to_cell
            A_cell_to_cell_x=A_cell_to_cell_x
            # A_cell_to_cell_x = A_node_to_cell
            # A_inv_cell_to_cell_x = torch.linalg.inv(A_cell_to_cell_x)

            # B_u = B_phi_cell_to_cell + B_phi_node_to_cell
            B_u = B_phi_cell_to_cell
            # B_u = B_phi_node_to_cell
        else:
            (
                phi_diff_cell_to_cell,
                phi_cell_outer,
                phi_cell_inner,
            ) = self.enfoce_boundary_conditions(
                phi_cell=phi_cell,
                indegree_cell=indegree_cell_x,
                outdegree_cell=outdegree_cell_x,
                face_type=graph_edge.x[:, 0:1],
                is_pressure=pressure_only,
                graph_edge=graph_edge,
            )

            twoway_B_cell_to_cell_x = torch.cat(
                (graph_cell_x.B_cell_to_cell_x, -graph_cell_x.B_cell_to_cell_x), dim=0
            )

            phi_diff_cell_to_cell_edge = (
                twoway_B_cell_to_cell_x * phi_diff_cell_to_cell.unsqueeze(1)
            )[mask_interior]

            B_phi_cell_to_cell = scatter_add(
                phi_diff_cell_to_cell_edge,
                indegree_cell_x[mask_interior],
                dim=0,
                dim_size=graph_cell.pos.shape[0],
            )

            phi_diff_node_to_cell = graph_cell_x.B_node_to_cell * (
                phi_node[cells_node] - phi_cell[cells_index]
            ).unsqueeze(1)

            B_phi_node_to_cell = calc_cell_centered_with_node_attr(
                node_attr=phi_diff_node_to_cell,
                cells_node=cells_node,
                cells_index=cells_index,
                reduce="sum",
                map=False,
            )
            B_u = B_phi_cell_to_cell + B_phi_node_to_cell

            A_inv_cell_to_cell_x = graph_cell_x.A_inv_cell_to_cell_x

        """ first method"""
        # nabla_phi_cell_lst = torch.linalg.lstsq(A_cell_to_cell_x, B_u).solution

        """ second method"""
        # nabla_phi_cell_lst = torch.matmul(A_inv_cell_to_cell_x,B_u)

        """ third method"""
        nabla_phi_cell_lst = torch.linalg.solve(A_cell_to_cell_x, B_u)

        nabla_phi_cell = torch.cat(
            [nabla_phi_cell_lst[:, :, i_dim] for i_dim in range(phi_node.shape[-1])],
            dim=-1,
        )

        if inteploting:
            nabla_phi_node = self.interpolating_phi_c_to_node(
                phi_c=nabla_phi_cell,
                cells_node=cells_node,
                centroid=graph_cell.pos,
                mesh_pos=graph_node.pos,
                cells_index=cells_index,
            )

            # nabla_phi_face = self.interpolating_gradients_to_faces(nabala_phi_c=nabla_phi_cell,
            #                                                         phi_cell_convection_outer=phi_cell_outer,
            #                                                         phi_cell_convection_inner=phi_cell_inner,
            #                                                         out_centroid=out_centroid,
            #                                                         in_centroid=in_centroid,
            #                                                         edge_neighbour_index=graph_cell.edge_index,
            #                                                         edge_center_pos=graph_edge.pos,
            #                                                         cells_face=graph_edge.face)
            nabla_phi_face = torch.zeros(
                (graph_edge.pos.size(0), 2 * phi_cell.size(1))
            ).to(nabla_phi_cell.device)
        else:
            nabla_phi_node = torch.zeros(
                (graph_node.pos.size(0), 2 * phi_cell.size(1))
            ).to(nabla_phi_cell.device)
            nabla_phi_face = torch.zeros(
                (graph_edge.pos.size(0), 2 * phi_cell.size(1))
            ).to(nabla_phi_cell.device)

        return nabla_phi_node, nabla_phi_face, nabla_phi_cell

    def node_based_WLSQ(
        self,
        phi_node=None,
        phi_cell=None,
        node_contribu=True,
        node_x_contribu=True,
        cell_to_node_contribu=True,
        graph_node_x=None,
        graph_node=None,
        graph_edge=None,
        graph_cell=None,
    ):
        mesh_pos = graph_node.pos
        face_node = graph_node.edge_index
        face_node_x = graph_node_x.edge_index
        cells_node = graph_node.face[0]
        cells_index = graph_cell.face[0]
        cell_factor = graph_cell.cell_factor
        centroid = graph_cell.pos

        """node to node contribution"""
        if node_contribu:
            outdegree_node_index = torch.cat((face_node[0], face_node[1]), dim=0)
            indegree_node_index = torch.cat((face_node[1], face_node[0]), dim=0)
            if not "B_node_to_node" in graph_node.keys():

                mesh_pos_diff_on_edge = (
                    mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
                ).unsqueeze(2)
                mesh_pos_diff_on_edge_T = mesh_pos_diff_on_edge.transpose(1, 2)
                weight_node_to_node = 1.0 / torch.norm(
                    mesh_pos_diff_on_edge, dim=1, keepdim=True
                )
                left_on_edge = torch.matmul(
                    mesh_pos_diff_on_edge * weight_node_to_node,
                    mesh_pos_diff_on_edge_T * weight_node_to_node,
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
                    * weight_node_to_node
                    * mesh_pos_diff_on_edge
                )

                B_node_to_node = scatter_add(
                    phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
                )
            else:
                A_node_to_node = graph_node_x.A_node_to_node
                """node to node x contribution"""
                two_way_B_node_to_node = torch.cat(
                    (graph_node_x.B_node_to_node, -graph_node_x.B_node_to_node), dim=0
                )

                phi_diff_on_edge = two_way_B_node_to_node * (
                    (
                        phi_node[outdegree_node_index] - phi_node[indegree_node_index]
                    ).unsqueeze(1)
                )

                B_node_to_node = scatter_add(
                    phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
                )
        else:
            A_node_to_node=0.
            B_node_to_node=0.
        """node to node x contribution"""
        
        """node to node x contribution"""
        if node_x_contribu:
            outdegree_node_index_x = torch.cat((face_node_x[0], face_node_x[1]), dim=0)
            indegree_node_index_x = torch.cat((face_node_x[1], face_node_x[0]), dim=0)
            if not "B_node_to_node_x" in graph_node_x():

                mesh_pos_diff_on_edge_x = (
                    mesh_pos[outdegree_node_index_x] - mesh_pos[indegree_node_index_x]
                ).unsqueeze(2)
                mesh_pos_diff_on_edge_x_T = mesh_pos_diff_on_edge_x.transpose(1, 2)
                weight_node_to_node_x = 1.0 / torch.norm(
                    mesh_pos_diff_on_edge_x, dim=1, keepdim=True
                )
                left_on_edge_x = torch.matmul(
                    mesh_pos_diff_on_edge_x * weight_node_to_node_x,
                    mesh_pos_diff_on_edge_x_T * weight_node_to_node_x,
                )

                A_node_to_node_x = scatter_add(
                    left_on_edge_x, indegree_node_index_x, dim=0, dim_size=mesh_pos.shape[0]
                )

                phi_diff_on_edge_x = (
                    weight_node_to_node_x
                    * (
                        (
                            phi_node[outdegree_node_index_x] - phi_node[indegree_node_index_x]
                        ).unsqueeze(1)
                    )
                    * weight_node_to_node_x
                    * mesh_pos_diff_on_edge_x
                )

                B_node_to_node_x = scatter_add(
                    phi_diff_on_edge_x, indegree_node_index_x, dim=0, dim_size=mesh_pos.shape[0]
                )
            else:
                
                A_node_to_node_x = graph_node_x.A_node_to_node_x
                """node to node x contribution"""
                two_way_B_node_to_node_x = torch.cat(
                    (graph_node_x.B_node_to_node_x, -graph_node_x.B_node_to_node_x), dim=0
                )

                phi_diff_on_edge_x = two_way_B_node_to_node_x * (
                    (
                        phi_node[outdegree_node_index_x] - phi_node[indegree_node_index_x]
                    ).unsqueeze(1)
                )

                B_node_to_node_x = scatter_add(
                    phi_diff_on_edge_x, indegree_node_index_x, dim=0, dim_size=mesh_pos.shape[0]
                )
        else:
            A_node_to_node_x=0.
            B_node_to_node_x=0.
        """node to node x contribution"""
        
        """cell to node contribution"""
        if cell_to_node_contribu:
            
            if not "B_cell_to_node" in graph_node_x.keys():
                centriod_mesh_pos_diff = (
                    centroid[cells_index] - mesh_pos[cells_node]
                ).unsqueeze(2)
                centriod_mesh_pos_diff_T = centriod_mesh_pos_diff.transpose(1, 2)
                weight_cell_node = 1.0 / torch.norm(
                    centriod_mesh_pos_diff, dim=1, keepdim=True
                )
                left_on_edge_cell_to_node = torch.matmul(
                    centriod_mesh_pos_diff * weight_cell_node,
                    centriod_mesh_pos_diff_T * weight_cell_node,
                )
                A_cell_to_node = scatter_add(left_on_edge_cell_to_node, cells_node, dim=0)

                phi_cell_node_diff_on_edge = (
                    weight_cell_node
                    * ((phi_cell[cells_index] - phi_node[cells_node]).unsqueeze(1))
                    * weight_cell_node
                    * centriod_mesh_pos_diff
                )

                B_cell_to_node = scatter_add(
                    phi_cell_node_diff_on_edge, cells_node, dim=0
                )
            else:
                
                A_cell_to_node = graph_node_x.A_cell_to_node
                
                phi_diff_cell_to_node = ((phi_cell[cells_index] - phi_node[cells_node]).unsqueeze(1))*graph_node_x.B_cell_to_node

                B_cell_to_node = calc_node_centered_with_cell_attr(cell_attr = phi_diff_cell_to_node, 
                                                                    cells_node = cells_node, 
                                                                    cells_index = cells_index, 
                                                                    reduce="add",
                                                                    map=False)
        else:
            A_cell_to_node = 0.
            B_cell_to_node = 0.
        """cell to node contribution"""
                
        A_left = A_node_to_node+A_node_to_node_x+A_cell_to_node
        # A_inv_node_to_node_x = torch.linalg.inv(A_node_to_node_x)
        
        B_right = B_node_to_node+B_node_to_node_x+B_cell_to_node
                
        # if "R_inv_Q_t" in graph_node_x.keys():
            
        #     R_inv_Q_t=graph_node_x.R_inv_Q_t
            
        #     mask_node_neigbors_fil=graph_node_x.mask_node_neigbors_fil
            
        #     node_neigbors=graph_node_x.face.T
            
        #     B_phi_node_to_node_x = (phi_node[node_neigbors]-phi_node.unsqueeze(1))*mask_node_neigbors_fil
            
        nabla_phi_node = []

        """ first method"""
        # nabla_phi_node_lst = torch.linalg.lstsq(
        #     A_node_to_node_x, B_phi_node_to_node_x
        # ).solution

        """ second method"""
        # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

        """ third method"""
        nabla_phi_node_lst = torch.linalg.solve(A_left, B_right)
        
        """ fourth method"""
        # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)
        
        nabla_phi_node = torch.cat(
            [nabla_phi_node_lst[:, :, i_dim] for i_dim in range(phi_node.shape[-1])],
            dim=-1,
        )

        # interplote node value to face and cell
        nabla_phi_face, nabla_phi_cell = self.interpolate_node_to_face_and_cell_avg(
            node_phi=nabla_phi_node,
            cells_node=cells_node,
            cells_index=cells_index,
            face_node=face_node,
            cell_factor=cell_factor,
        )

        return nabla_phi_node, nabla_phi_face, nabla_phi_cell

    def compute_node_WSLQ_in_polar_axis(
        self,
        phi_node=None,
        phi_cell=None,
        mesh_pos=None,
        centroid=None,
        face_node=None,
        face_node_x=None,
        cells_node=None,
        cells_index=None,
        cell_factor=None,
        graph_node=None,
        graph_node_x=None,
        graph_edge=None,
        graph_cell=None,
    ):
        def transform_cartesian_to_polar(mesh_pos):
            gamma = torch.sqrt(mesh_pos[:, 0:1] ** 2 + mesh_pos[:, 1:2] ** 2)

            theta = torch.atan2(mesh_pos[:, 1:2], mesh_pos[:, 0:1])

            polar_pos = torch.cat((gamma, theta), dim=-1)

            return polar_pos

        polar_pos = transform_cartesian_to_polar(mesh_pos)

        polar_centroid = transform_cartesian_to_polar(centroid)

        (
            nabla_phi_node_polar,
            nabla_phi_face_polar,
            nabla_phi_cell_polar,
        ) = self.node_based_WLSQ(
            phi_node=phi_node,
            graph_node=graph_node,
            graph_node_x=graph_node_x,
            graph_edge=graph_edge,
            graph_cell=graph_cell,
        )

        def transform_nabla_polar_to_cartesian(nabla_phi_node_polar, polar_pos):
            cartesian_nabla_phi = []
            for i_dim in range(int(nabla_phi_node_polar.shape[-1] / 2.0)):
                current_polar_nabla_phi = nabla_phi_node_polar[
                    ..., 2 * i_dim : 2 * (i_dim + 1)
                ]
                current_cartesian_nabla_phi_x = current_polar_nabla_phi[
                    :, 0:1
                ] * torch.cos(polar_pos[:, 1:2]) - (
                    1.0 / polar_pos[:, 0:1]
                ) * current_polar_nabla_phi[
                    :, 1:2
                ] * torch.sin(
                    polar_pos[:, 1:2]
                )
                current_cartesian_nabla_phi_y = current_polar_nabla_phi[
                    :, 0:1
                ] * torch.sin(polar_pos[:, 1:2]) + (
                    1.0 / polar_pos[:, 0:1]
                ) * current_polar_nabla_phi[
                    :, 1:2
                ] * torch.cos(
                    polar_pos[:, 1:2]
                )
                current_cartesian_nabla_phi = torch.cat(
                    (current_cartesian_nabla_phi_x, current_cartesian_nabla_phi_y),
                    dim=-1,
                )
                cartesian_nabla_phi.append(current_cartesian_nabla_phi)
            return torch.cat(cartesian_nabla_phi, dim=-1)

        cartesian_nabla_phi = transform_nabla_polar_to_cartesian(
            nabla_phi_node_polar, polar_pos
        )

        # interplote node value to face and cell
        nabla_phi_face, nabla_phi_cell = self.interpolate_node_to_face_and_cell_avg(
            node_phi=cartesian_nabla_phi,
            cells_node=cells_node,
            cells_index=cells_index,
            face_node=face_node,
            cell_factor=cell_factor,
        )

        return cartesian_nabla_phi, nabla_phi_face, nabla_phi_cell

    def laplace_node(
        self,
        nabla_node_phi=None,
        nabla_cell_phi=None,
        graph_node=None,
        graph_node_x=None,
        graph_edge=None,
        graph_cell=None,
    ):
        laplace_uvp_node = []
        laplace_uvp_face = []
        laplace_uvp_cell = []

        for i_dim in range(int(nabla_node_phi.shape[-1] / 2)):
            current_nabla_node_phi = nabla_node_phi[..., 2 * i_dim : 2 * (i_dim + 1)]

            current_nabla_cell_phi = nabla_cell_phi[..., 2 * i_dim : 2 * (i_dim + 1)]

            (
                current_laplace_uvp_node,
                current_laplace_uvp_face,
                current_laplace_uvp_cell,
            ) = self.node_based_WLSQ(
                phi_node=current_nabla_node_phi,
                graph_node_x=graph_node_x,
                graph_node=graph_node,
                graph_edge=graph_edge,
                graph_cell=graph_cell,
            )

            laplace_uvp_node.append(
                torch.cat(
                    (
                        current_laplace_uvp_node[:, 0:1],
                        current_laplace_uvp_node[:, -2:-1],
                    ),
                    dim=-1,
                )
            )
            laplace_uvp_face.append(
                torch.cat(
                    (
                        current_laplace_uvp_face[:, 0:1],
                        current_laplace_uvp_face[:, -2:-1],
                    ),
                    dim=-1,
                )
            )
            laplace_uvp_cell.append(
                torch.cat(
                    (
                        current_laplace_uvp_cell[:, 0:1],
                        current_laplace_uvp_cell[:, -2:-1],
                    ),
                    dim=-1,
                )
            )

        laplace_uvp_node = torch.cat(laplace_uvp_node, dim=-1)
        laplace_uvp_face = torch.cat(laplace_uvp_face, dim=-1)
        laplace_uvp_cell = torch.cat(laplace_uvp_cell, dim=-1)

        return laplace_uvp_node, laplace_uvp_face, laplace_uvp_cell

    def plot_and_validation(
        self, graph_node, graph_edge, graph_cell, nabla_uvp_node=None
    ):
        mesh_pos = graph_node.pos.cpu()
        edge_index = graph_node.edge_index.cpu()
        face_center_pos = graph_edge.pos.cpu()
        unit_norm_v = graph_cell.unv.cpu()
        centroid = graph_cell.pos.cpu()
        origin_cells_area = graph_cell.cell_area.cpu()
        node_type = graph_node.node_type.cpu()
        # compute cell_area
        cells_index = graph_cell.face[0].cpu()
        cells_face = graph_edge.face[0].cpu()
        face_length = graph_edge.x[:, 1:2].cpu()

        surface_vector = unit_norm_v * face_length[cells_face]
        full_synataic_function = 0.5 * face_center_pos[cells_face.view(-1)]

        cells_area = calc_cell_centered_with_node_attr(
            node_attr=(full_synataic_function * surface_vector).sum(
                dim=1, keepdim=True
            ),
            cells_node=cells_face,
            cells_index=cells_index,
            reduce="sum",
            map=False,
        )

        # use shoelace formula to validate
        # cells_face_node_unbiased = decomposed_cells["cells_face_node_unbiased"]
        # test_cells_area = []
        # for i in range(cells_index.max().numpy()+1):
        #     test_cells_area.append(polygon_area(mesh_pos[cells_node[(cells_index==i).view(-1)].view(-1)]))
        # test_cells_area = torch.from_numpy(np.asarray(test_cells_area))

        # valid_cells_area = (cells_area.view(-1)-test_cells_area).sum()

        if not self.plotted:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            self.plotted = True
            self.ax_scatter = ax.twinx()  # 创建一个新的Axes对象与原始Axes对象共享x轴

            ax.set_aspect("equal")
            self.ax_scatter.set_aspect("equal")
            # 通过索引获取每一条边的两个点的坐标
            point1 = mesh_pos[edge_index[0]]
            point2 = mesh_pos[edge_index[1]]

            # 将每一对点的坐标合并，方便绘图
            lines = np.hstack([point1, point2]).reshape(-1, 2, 2)

            # 使用plot绘制所有的边
            ax.plot(lines[:, :, 0].T, lines[:, :, 1].T, "k-", lw=1, alpha=0.2)
        else:
            # 清除上一次的scatter内容
            self.ax_scatter.clear()

        node_size = 5
        # if face_center_pos is not None:
        #     plt.scatter(face_center_pos[:,0],face_center_pos[:,1],c='red',linewidths=1,s=1)

        # if centroid is not None:
        #     plt.scatter(centroid[:,0],centroid[:,1],c='blue',linewidths=1,s=1)

        # if node_type is not None:
        #     try:
        #         node_type=node_type.view(-1)
        #     except:
        #         node_type=node_type.reshape(-1)
        #     plt.scatter(mesh_pos[node_type==NodeType.NORMAL,0],mesh_pos[node_type==NodeType.NORMAL,1],c='red',linewidths=1,s=node_size)
        #     plt.scatter(mesh_pos[node_type==NodeType.WALL_BOUNDARY,0],mesh_pos[node_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=node_size)
        #     plt.scatter(mesh_pos[node_type==NodeType.OUTFLOW,0],mesh_pos[node_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=node_size)
        #     plt.scatter(mesh_pos[node_type==NodeType.INFLOW,0],mesh_pos[node_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=node_size)

        # ax.quiver(centroid[cells_index,0],centroid[cells_index,1],unit_norm_v[:,0],unit_norm_v[:,1],units='height',color="red", angles='xy',scale_units='xy', scale=200,width=0.01, headlength=3, headwidth=2, headaxislength=3.5)

        mask_nabla_u_node = (nabla_uvp_node[:, 0] == 0.0).view(-1).cpu()
        self.ax_scatter.scatter(
            mesh_pos[mask_nabla_u_node, 0],
            mesh_pos[mask_nabla_u_node, 1],
            c="red",
            linewidths=1,
            s=node_size,
        )

        mask_nabla_v_node = (nabla_uvp_node[:, 1] == 0.0).view(-1).cpu()
        self.ax_scatter.scatter(
            mesh_pos[mask_nabla_v_node, 0],
            mesh_pos[mask_nabla_v_node, 1],
            c="blue",
            linewidths=1,
            s=node_size,
        )

        mask_nabla_p_node = (nabla_uvp_node[:, 2] == 0.0).view(-1).cpu()
        self.ax_scatter.scatter(
            mesh_pos[mask_nabla_p_node, 0],
            mesh_pos[mask_nabla_p_node, 1],
            c="green",
            linewidths=1,
            s=node_size,
        )

        plt.show()

    def forward(
        self,
        predicted_node_uvp=None,
        graph_node=None,
        graph_node_x=None,
        graph_edge=None,
        graph_cell=None,
        graph_cell_x=None,
        params=None,
        inteplote=False,
        device=None,
    ):
        # prepare face neighbour cell`s index
        cells_node = graph_node.face[0]
        cells_face = graph_edge.face[0]
        cells_index = graph_cell.face[0]
        cell_area = graph_cell.cell_area
        cells_type = graph_cell.cells_type.view(-1)
        cells_node_face_unv = graph_cell.cells_node_face_unv
        cells_node_surface_vector = graph_cell.cells_node_surface_vector
        face_type = graph_edge.x[:, 0:1]
        pde_theta_cell = graph_cell.pde_theta_cell
        senders_node, recivers_node = graph_node.edge_index
        mesh_pos = graph_node.pos
        face_area = graph_edge.x[:, 1:2]
        node_type = graph_node.node_type[:, 0:1]

        # pde coefficent
        continuity_eq_coefficent = pde_theta_cell[:, 0:1]
        convection_coefficent = pde_theta_cell[:, 1:2]
        grad_p_coefficent = pde_theta_cell[:, 2:3]
        diffusion_coefficent = pde_theta_cell[:, 3:4]
        source_term = pde_theta_cell[:, 4:5] * cell_area

        """>>> Reconstruct Gradient >>>"""
        nabla_phi_node_1st, nabla_phi_face_1st, nabla_phi_cell_1st = self.node_based_WLSQ(
            phi_node=predicted_node_uvp[:, 0:3],
            phi_cell=None,
            node_contribu=False,
            node_x_contribu=True,
            cell_to_node_contribu=False,
            graph_node_x=graph_node_x,
            graph_node=graph_node,
            graph_edge=graph_edge,
            graph_cell=graph_cell,
        )
        
        predicted_cell_uvp = self.interploate_node_to_cell_2nd_order(
                    node_phi=predicted_node_uvp[:, 0:3], 
                    nabla_node_phi=nabla_phi_node_1st, 
                    graph_node=graph_node, 
                    graph_cell=graph_cell
                )
         
        nabla_uvp_node, nabla_uvp_face, nabla_uvp_cell = self.node_based_WLSQ(
            phi_node=predicted_node_uvp[:, 0:3],
            phi_cell=predicted_cell_uvp,
            node_contribu=True,
            node_x_contribu=False,
            cell_to_node_contribu=True,
            graph_node_x=graph_node_x,
            graph_node=graph_node,
            graph_edge=graph_edge,
            graph_cell=graph_cell,
        )


        nabla_p_cell = nabla_uvp_cell[:, 4:6]
        """<<< Reconstruct Gradient <<<"""

        if not params.GG_convection:
            """>>> WLSQ continuity equation >>>"""
            """>>> At continuity equation, we use predicted_node_uvp instead of uv_node_hat to ensure mass conservation >>>"""
            if params.projection_method <= 0:
                continutiy_eq_cell = continuity_eq_coefficent*(
                    nabla_uvp_cell[:,0:1]
                    +nabla_uvp_cell[:,3:4]
                    )*cell_area

                # continutiy_eq_node = nabla_uvp_node[:, 0:1] + nabla_uvp_node[:, 3:4]

                # continutiy_eq_cell = (
                #     continutiy_eq_coefficent
                #     * self.interploate_node_to_cell_2nd_order(
                #         node_phi=continutiy_eq_node,
                #         nabla_node_phi=None,
                #         graph_node=graph_node,
                #         graph_cell=graph_cell,
                #     )
                #     * cell_area
                # )

            else:
                continutiy_eq_cell = 0.0
            """<<< WLSQ continuity equation <<<"""

            """>>> WLSQ convection term >>>"""
            if params.conserved_form:
                uu = predicted_node_uvp[:, 0:1] * predicted_node_uvp[:, 0:2]
                vu = predicted_node_uvp[:, 1:2] * predicted_node_uvp[:, 0:2]

                uu_vu_node = torch.cat((uu, vu), dim=-1)

                uu_vu_cell = (
                    continuity_eq_coefficent
                    * self.interploate_node_to_cell_2nd_order(
                        node_phi=uu_vu_node,
                        nabla_node_phi=None,
                        graph_node=graph_node,
                        graph_cell=graph_cell,
                    )
                )

                (
                    nabla_uuvu_node,
                    nabla_uuvu_face,
                    nabla_uuvu_cell,
                ) = self.node_based_WLSQ(
                    phi_node=uu_vu_node,
                    graph_node_x=graph_node_x,
                    graph_node=graph_node,
                    graph_edge=graph_edge,
                    graph_cell=graph_cell,
                )

                convection_x_node = nabla_uuvu_node[:, 0:1] + nabla_uuvu_node[:, 3:4]

                convection_y_node = nabla_uuvu_node[:, 4:5] + nabla_uuvu_node[:, 7:8]

                convection_node = torch.cat(
                    (convection_x_node, convection_y_node), dim=1
                )

                convection_cell = (
                    self.interploate_node_to_cell_2nd_order(
                        node_phi=convection_node,
                        nabla_node_phi=None,
                        graph_node=graph_node,
                        graph_cell=graph_cell,
                    )
                    * cell_area
                )

                convection_x_cell = convection_cell[:, 0:1]
                convection_y_cell = convection_cell[:, 1:2]

            else:
                # convection_x_node_hat = torch.sum(
                #     uv_node_hat[:, 0:2] * nabla_uv_hat_node[:, 0:2], dim=1, keepdim=True
                # )

                # convection_y_node_hat = torch.sum(
                #     uv_node_hat[:, 0:2] * nabla_uv_hat_node[:, 2:4], dim=1, keepdim=True
                # )

                # convection_node_hat = torch.cat(
                #     (convection_x_node_hat, convection_y_node_hat), dim=1
                # )

                # convection_cell_hat = (
                #     self.interploate_node_to_cell_2nd_order(
                #         node_phi=convection_node_hat,
                #         nabla_node_phi=None,
                #         graph_node=graph_node,
                #         graph_cell=graph_cell,
                #     )
                #     * cell_area
                # )

                # convection_x_cell = convection_cell_hat[:, 0:1]
                # convection_y_cell = convection_cell_hat[:, 1:2]

                # convection_x_cell = torch.sum(
                    # uv_cell_hat[:,0:2]*nabla_uv_cell_hat[:,0:2],dim=1,keepdim=True
                    # )*cell_area

                # convection_y_cell = torch.sum(
                    # uv_cell_hat[:,0:2]*nabla_uv_cell_hat[:,2:4],dim=1,keepdim=True
                    # )*cell_area
                    
                # conv_uv_cell_hat = self.interploate_node_to_cell_2nd_order(
                #         node_phi=uv_node_hat,
                #         nabla_node_phi=nabla_uv_hat_node,
                #         graph_node=graph_node,
                #         graph_cell=graph_cell,
                #     )
                
                convection_x_cell = torch.sum(
                    predicted_cell_uvp[:,0:2]*nabla_uvp_cell[:,0:2],dim=1,keepdim=True
                    )*cell_area

                convection_y_cell = torch.sum(
                    predicted_cell_uvp[:,0:2]*nabla_uvp_cell[:,2:4],dim=1,keepdim=True
                    )*cell_area
                
            """<<< WLSQ convection term  <<<"""
        else:
            """>>> GG continuity equation >>>"""
            """>>> At continuity equation, we use predicted_node_uvp instead of uv_node_hat to ensure mass conservation >>>"""
            # continuity_flux = self.chain_dot_product(predicted_node_uvp[cells_node,0:2],cells_node_surface_vector)

            # continutiy_eq_cell  = continutiy_eq_coefficent*calc_cell_centered_with_node_attr(node_attr = continuity_flux,
            #                                                                                 cells_node = cells_node,
            #                                                                                 cells_index = cells_index,
            #                                                                                 reduce="sum",
            #                                                                                 map=False)
            # '''<<< GG continuity equation <<<'''

            # '''>>> GG convection term >>>'''
            # # uu_face = predicted_uv_face[:,0:1]*predicted_uv_face[:,0:2]
            # # vu_face = predicted_uv_face[:,1:2]*predicted_uv_face[:,0:2]
            # uu_node = predicted_node_uvp[:,0:1]*predicted_node_uvp[:,0:2]
            # vu_node = predicted_node_uvp[:,1:2]*predicted_node_uvp[:,0:2]

            # convection_flux = self.chain_vector_dot_product(torch.cat((uu_node,vu_node),dim=-1)[cells_node],cells_node_surface_vector)

            # convection_cell  = calc_cell_centered_with_node_attr(node_attr = convection_flux,
            #                                                         cells_node = cells_node,
            #                                                         cells_index = cells_index,
            #                                                         reduce="sum",
            #                                                         map=False)

            # convection_x_cell = convection_cell[:,0:1]
            # convection_y_cell = convection_cell[:,1:2]
            """<<< GG convection term  <<<"""

        """>>> grad p term >>>"""
        cell_node_outflow_mask = node_type[cells_node] == NodeType.OUTFLOW

        if params.integrate_p or (params.pressure_open_bc > 0.0):
            viscosity_force_pressure_outlet = diffusion_coefficent[
                cells_index
            ] * self.chain_vector_dot_product(
                nabla_uvp_node[cells_node, 0:4], cells_node_face_unv
            )

        if params.pressure_open_bc > 0.0:
            # pressure outlet condition
            surface_p = predicted_node_uvp[cells_node, 2:3] * cells_node_face_unv

            loss_pressure_outlet = (
                viscosity_force_pressure_outlet.clone().detach() - surface_p
            ) * (cell_node_outflow_mask)

        else:
            loss_pressure_outlet = predicted_node_uvp[cells_node, 2:3] * (
                cell_node_outflow_mask
            )

        volume_integrate_P = nabla_p_cell * cell_area
        volume_integrate_P_x = volume_integrate_P[:, 0:1]
        volume_integrate_P_y = volume_integrate_P[:, 1:2]
        """<<< grad p term  <<<"""

        """>>> projection method >>>"""
        projection_method = 0.0
        """<<< projection method <<<"""

        """>>> diffusion term >>>"""
        viscosity_force = self.chain_vector_dot_product(
            nabla_uvp_node[cells_node, 0:4], cells_node_surface_vector
        )

        viscosity_force = calc_cell_centered_with_node_attr(
            node_attr=viscosity_force,
            cells_node=cells_node,
            cells_index=cells_index,
            reduce="sum",
            map=False,
        )

        laplace_cell_x = viscosity_force[:, 0:1]
        laplace_cell_y = viscosity_force[:, 1:2]
        """<<< diffusion term  <<<"""

        loss_momtentum_x_cell = (
            convection_coefficent * convection_x_cell
            + grad_p_coefficent * volume_integrate_P_x
            - diffusion_coefficent * laplace_cell_x
            - source_term
        )

        loss_momtentum_y_cell = (
            convection_coefficent * convection_y_cell
            + grad_p_coefficent * volume_integrate_P_y
            - diffusion_coefficent * laplace_cell_y
            - source_term
        )

        if params.ncn_smooth:
            uv_node_new = self.interpolating_phi_c_to_node(
                phi_c=predicted_cell_uvp[:,0:2],
                nabla_phi_c=None,
                cells_node=graph_node.face[0],
                centroid=graph_cell.pos,
                mesh_pos=graph_node.pos,
                cells_index=graph_cell.face[0],
            )

            boundary_fluid_mask = (
                (graph_node.node_type == NodeType.INFLOW)
                | (graph_node.node_type == NodeType.WALL_BOUNDARY)
            ).squeeze(1)

            uv_node_new[boundary_fluid_mask, 0:2] = graph_node.y[
                boundary_fluid_mask, 0:2
            ]

            smoothed_predicted_uvp_node = torch.cat(
                (uv_node_new[:, 0:2], predicted_node_uvp[:,2:3]), dim=-1
            )
            # smoothed_predicted_uvp_node = uvp_node_new
        else:
            smoothed_predicted_uvp_node = predicted_node_uvp

        if params.rollout:
  
            return (
                smoothed_predicted_uvp_node,
                continutiy_eq_cell,
                loss_momtentum_x_cell,
                loss_momtentum_y_cell,
                )
                
        return (
            continutiy_eq_cell,
            loss_momtentum_x_cell,
            loss_momtentum_y_cell,
            loss_pressure_outlet,
            smoothed_predicted_uvp_node,
            nabla_uvp_node,
            projection_method,
        )

    def interploate_node_to_cell_2nd_order(
        self, node_phi=None, nabla_node_phi=None, graph_node=None, graph_cell=None
    ):
        if nabla_node_phi is not None:
            if node_phi.shape[-1] >= 2:
                cells_node = graph_node.face[0]
                cells_index = graph_cell.face[0]
                r_n_2_c = graph_cell.pos[cells_index] - graph_node.pos[cells_node]
                cells_node_value = node_phi[cells_node] + self.chain_vector_dot_product(
                    nabla_node_phi[cells_node], r_n_2_c
                )

                cell_center_attr = calc_cell_centered_with_node_attr(
                    node_attr=cells_node_value,
                    cells_node=cells_node,
                    cells_index=cells_index,
                    reduce="mean",
                    map=False,
                )

                return cell_center_attr

            elif node_phi.shape[-1] == 1:
                cells_node = graph_node.face[0]
                cells_index = graph_cell.face[0]
                r_n_2_c = graph_cell.pos[cells_index] - graph_node.pos[cells_node]
                cells_node_value = node_phi[cells_node] + self.chain_dot_product(
                    nabla_node_phi[cells_node], r_n_2_c
                )

                cell_center_attr = calc_cell_centered_with_node_attr(
                    node_attr=cells_node_value,
                    cells_node=cells_node,
                    cells_index=cells_index,
                    reduce="mean",
                    map=False,
                )

                return cell_center_attr
            else:
                raise ValueError("Wrong node phi channel dimension")
        else:
            cells_node = graph_node.face[0]
            cells_index = graph_cell.face[0]
            cell_center_attr = calc_cell_centered_with_node_attr(
                node_attr=node_phi,
                cells_node=cells_node,
                cells_index=cells_index,
                reduce="mean",
                map=True,
            )
            return cell_center_attr

    def interploate_face_to_node_and_cell(
        self,
        face_phi=None,
        cells_face=None,
        cells_index=None,
        face_node=None,
        choice=["node", "cell"],
    ):
        if "node" in choice:
            node_phi = scatter_mean(
                torch.cat((face_phi, face_phi), dim=0),
                torch.cat((face_node[1], face_node[0]), dim=0),
                dim=0,
            )
        else:
            node_phi = None

        if "cell" in choice:
            cell_center_attr = calc_cell_centered_with_node_attr(
                node_attr=face_phi,
                cells_node=cells_face,
                cells_index=cells_index,
                reduce="mean",
                map=True,
            )
        else:
            cell_center_attr = None

        if (node_phi is not None) and (cell_center_attr is not None):
            return node_phi, cell_center_attr
        elif node_phi is not None:
            return node_phi
        elif cell_center_attr is not None:
            return cell_center_attr
        else:
            raise ValueError("HEllo?????????????")

    def interpolate_node_to_face_and_cell_avg(
        self,
        node_phi=None,
        cells_node=None,
        cells_index=None,
        face_node=None,
        cell_factor=None,
    ):
        cell_center_attr = calc_cell_centered_with_node_attr(
            node_attr=node_phi,
            cells_node=cells_node,
            cells_index=cells_index,
            reduce="mean",
            map=True,
        )

        return (node_phi[face_node[0]] + node_phi[face_node[1]]) / 2.0, cell_center_attr

    def interpolating_phi_c_to_node(
        self,
        phi_c=None,
        nabla_phi_c=None,
        cells_node=None,
        centroid=None,
        mesh_pos=None,
        cells_index=None,
    ):
        """
        Interpolates phi_c values to nodes with weights.

        Parameters:
        phi_c (Tensor): The values of phi at cell centers.
        nabla_phi_c (Tensor): The gradient of phi at cell centers.
        cells_node (Tensor): The nodes of the cells.
        centroid (Tensor): The centroids of the cells.
        mesh_pos (Tensor): The positions of the mesh nodes.
        cells_index (Tensor): The indices of the cells.

        Returns:
        Tensor: The interpolated values at the nodes.
        """
        mesh_pos_to_centriod = mesh_pos[cells_node] - centroid[cells_index]

        weight = 1.0 / torch.norm(mesh_pos_to_centriod, dim=-1, keepdim=True)

        if nabla_phi_c is not None:
            correction_term = self.chain_vector_dot_product(
                nabla_phi_c[cells_index], mesh_pos_to_centriod
            )

            aggrate_cell_attr = (phi_c[cells_index] + correction_term) * weight

        else:
            aggrate_cell_attr = phi_c[cells_index] * weight

        cell_to_node = scatter_add(aggrate_cell_attr, cells_node, dim=0) / scatter_add(
            weight, cells_node, dim=0
        )

        return cell_to_node

    def interpolating_phi_c_to_node_non_weights(
        self,
        phi_c=None,
        nabla_phi_c=None,
        cells_node=None,
        centroid=None,
        mesh_pos=None,
        cells_index=None,
    ):
        """
        Placeholder function for vector dot product.
        Replace with actual implementation.

        Parameters:
        vector_a (Tensor): First vector.
        vector_b (Tensor): Second vector.

        Returns:
        Tensor: The dot product of the vectors.
        """

        mesh_pos_to_centriod = mesh_pos[cells_node] - centroid[cells_index]

        if nabla_phi_c is not None:
            correction_term = self.chain_vector_dot_product(
                nabla_phi_c[cells_index], mesh_pos_to_centriod
            )

            aggrate_cell_attr = phi_c[cells_index] + correction_term

        else:
            aggrate_cell_attr = phi_c[cells_index]

        cell_to_node = scatter_mean(aggrate_cell_attr, cells_node, dim=0)

        return cell_to_node

    # a and b has to be the same size
    def chain_dot_product(self, a, b):
        return torch.sum(a * b, dim=-1, keepdim=True)

    # 4dim a dot product 2dim b
    def chain_vector_dot_product(self, a, b):
        # nabla_u = self.chain_dot_product(a[...,0:2],b)
        # nabla_v = self.chain_dot_product(a[...,2:4],b)
        rt_val = []
        if a.size(-1)<=1:
            return self.chain_dot_product(a, b)
        
        for i_dim in range(int(a.shape[-1] / 2)):
            nabla_u = self.chain_dot_product(a[..., 2 * i_dim : 2 * (i_dim + 1)], b)
            rt_val.append(nabla_u)

        return torch.cat(rt_val, dim=-1)

    # a and b has to be the different size
    def chain_flux_dot_product(self, a, b):
        c = a * (b.repeat(1, 2))
        return torch.stack(
            (torch.add(c[:, 0], c[:, 1]), torch.add(c[:, 2], c[:, 3])), dim=-1
        )

    # a 6 dim and b 2 dim
    def chain_flux_dot_product_up_three(self, a, b):
        c = a * (b.repeat(1, 3))
        return torch.stack(
            (
                torch.add(c[:, 0], c[:, 1]),
                torch.add(c[:, 2], c[:, 3]),
                torch.add(c[:, 4], c[:, 5]),
            ),
            dim=-1,
        )

    # 4dim a element-wise product 2dim b
    def chain_element_wise_vector_product_down(self, a, b):
        return a * (b.repeat(1, 2))

    # 2dim element-wise product 4dim b
    def chain_element_wise_vector_product_up(self, a, b):
        return torch.cat(
            ((a[:, 0:1].repeat(1, 2)) * b, (a[:, 1:2].repeat(1, 2)) * b), dim=1
        )

    # 3dim element-wise product 6dim b
    def chain_element_wise_vector_product_up_three(self, a, b):
        return torch.cat(
            (
                (a[:, 0:1].repeat(1, 2)) * b,
                (a[:, 1:2].repeat(1, 2)) * b,
                (a[:, 2:3].repeat(1, 2)) * b,
            ),
            dim=1,
        )

    def chain_vector_div(self, a, b):
        return torch.cat((a[:, 0:2] / b, a[:, 2:4] / b), dim=1)

    # 2dim a dot product 4dim b
    def chain_vector_dot_product_b(self, a, b):
        c = (a.repeat(1, 2)) * b
        return torch.cat(
            (torch.add(c[:, 0:1], c[:, 1:2]), torch.add(c[:, 2:3], c[:, 3:4])), dim=1
        )
