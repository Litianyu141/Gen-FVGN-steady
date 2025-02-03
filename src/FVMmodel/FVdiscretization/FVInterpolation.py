import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import torch
import torch.nn as nn
from Utils.utilities import (
    decompose_and_trans_node_attr_to_cell_attr_graph,
    copy_geometric_data,
    NodeType,
    calc_cell_centered_with_node_attr,
    calc_node_centered_with_cell_attr,
)
from torch_geometric.data import Data
import numpy as np
from torch_scatter import scatter_add, scatter_mean
import matplotlib.pyplot as plt

class Interplot(nn.Module):
    def __init__(self):
        super().__init__()
        self.plotted = False

    def node_to_cell_2nd_order(
        self,
        node_phi=None,
        node_grad=None,
        node_hessian=None,
        graph_node=None,
        graph_cell=None,
        
        cells_node = None,
        cells_index = None,
        mesh_pos = None,
        centroid = None,

    ):
        if (cells_node is None) and (cells_index is None) and (mesh_pos is None) and (centroid is None):
            cells_node = graph_node.face
            cells_index = graph_cell.face
            mesh_pos = graph_node.pos
            centroid = graph_cell.pos

        r_n_2_c = (centroid[cells_index] - mesh_pos[cells_node]).unsqueeze(1)

        if node_grad is not None:
            if len(node_grad.size()) > 3:
                first_order_correction = (
                    torch.matmul(
                        r_n_2_c.unsqueeze(1).unsqueeze(1),
                        node_grad[cells_node].unsqueeze(-1),
                    )
                    .squeeze()
                )
            else:
                first_order_correction = (
                    torch.matmul(r_n_2_c.unsqueeze(1),node_grad[cells_node].unsqueeze(-1))
                    .squeeze()
                )
            if len(first_order_correction.shape)<2:
                first_order_correction=first_order_correction[:,None]
        else:
            first_order_correction = 0.0
         
        if node_hessian is not None:

            # 计算 second order correction
            r_t_c_expanded_T = r_n_2_c.unsqueeze(1)
            r_t_c_expanded = r_t_c_expanded_T.transpose(2, 3)

            # 计算 r^T * H * r
            intermediate = torch.matmul(
                r_t_c_expanded_T, node_hessian[cells_node]
            )  # [num_nodes, num_vars, 1, 2]
            second_order_correction = 0.5 * torch.matmul(
                intermediate, r_t_c_expanded
            ).squeeze()  # [num_nodes, num_vars, 1, 1]
            
            if len(second_order_correction.shape)<2:
                second_order_correction=second_order_correction[:,None]
                
        else:
            second_order_correction = 0.0

        cells_node_value = (
            node_phi[cells_node] + first_order_correction + second_order_correction
        )

        cell_center_attr = calc_cell_centered_with_node_attr(
            node_attr=cells_node_value,
            cells_node=cells_node,
            cells_index=cells_index,
            reduce="mean",
            map=False,
        )

        return cell_center_attr

    def node_to_face_2nd_order(
        self,
        node_phi=None,
        node_grad=None,
        node_hessian=None,
        graph_node=None,
        graph_edge=None,
    ):

        senders_node, recivers_node = (
            graph_node.edge_index[0],
            graph_node.edge_index[1],
        )
        two_way_senders_node = torch.cat((senders_node, recivers_node), dim=0)
        two_way_face_center_pos = graph_edge.pos.repeat(2, 1)
        r_2_f = (two_way_face_center_pos - graph_node.pos[two_way_senders_node]).unsqueeze(1)

        if node_grad is not None:
            if len(node_grad.size()) > 3:
                first_order_correction = (
                    torch.matmul(
                        r_2_f.unsqueeze(1).unsqueeze(1),
                        node_grad[two_way_senders_node].unsqueeze(-1),
                    )
                    .squeeze()
                )
            else:
                first_order_correction = (
                    torch.matmul(
                        r_2_f.unsqueeze(1),
                        node_grad[two_way_senders_node].unsqueeze(-1),
                    )
                    .squeeze()
                )
                
            if len(first_order_correction.shape)<2:
                first_order_correction=first_order_correction[:,None]
                
        else:
            first_order_correction = 0.0

        if node_hessian is not None:
            # 计算 second order correction
            r_t_c_expanded_T = r_2_f.unsqueeze(1)  # 变成 [num_nodes, 1, 2] 以便广播
            r_t_c_expanded = r_t_c_expanded_T.transpose(
                2, 3
            )  # 变成 [num_nodes, 2, 1] 以便广播

            # 计算 r^T * H * r
            intermediate = torch.matmul(
                r_t_c_expanded_T, node_hessian[two_way_senders_node]
            )  # [num_nodes, num_vars, 1, 2]
            second_order_correction = 0.5 * torch.matmul(
                intermediate, r_t_c_expanded
            )  # [num_nodes, num_vars, 1, 1]
            second_order_correction = second_order_correction.squeeze()  # 去掉多余的维度 -> [num_nodes, num_vars]
            
            if len(second_order_correction.shape)<2:
                second_order_correction=second_order_correction[:,None]
        else:
            second_order_correction = 0.0
 
        two_face_node_value = (
            node_phi[two_way_senders_node]
            + first_order_correction
            + second_order_correction
        )

        num_edges = two_face_node_value.size(0)

        face_node_value = (
            two_face_node_value[: num_edges // 2] + two_face_node_value[num_edges // 2 :]
        ) / 2.0

        return face_node_value

    def face_to_node(
        self,
        face_phi=None, # must be single way
        face_node=None, # must be single way
    ):

        node_phi = scatter_mean(
            torch.cat((face_phi, face_phi), dim=0),
            torch.cat((face_node[1], face_node[0]), dim=0),
            dim=0,
        )

        return node_phi
    
    def face_to_cell(
        self,
        face_phi=None,
        cells_face=None,
        cells_index=None,
    ):

        cell_center_attr = calc_cell_centered_with_node_attr(
            node_attr=face_phi,
            cells_node=cells_face,
            cells_index=cells_index,
            reduce="mean",
            map=True,
        )

        raise cell_center_attr

    def cell_to_node_2nd_order(
        self,
        cell_phi=None,
        cell_grad=None,
        cells_node=None,
        cells_index=None,
        centroid=None,
        mesh_pos=None,
    ):
        """
        Interpolates cell_phi values to nodes with weights.

        Parameters:
        cell_phi (Tensor): The values of phi at cell centers.
        cell_grad (Tensor): The gradient of phi at cell centers.
        cells_node (Tensor): The nodes of the cells.
        centroid (Tensor): The centroids of the cells.
        mesh_pos (Tensor): The positions of the mesh nodes.
        cells_index (Tensor): The indices of the cells.

        Returns:
        Tensor: The interpolated values at the nodes.
        """
        mesh_pos_to_centriod = mesh_pos[cells_node] - centroid[cells_index]

        weight = 1.0 / torch.norm(mesh_pos_to_centriod, dim=-1, keepdim=True)

        if (cell_grad is not None):
            if len(cell_grad.size()) < 3:
                raise ValueError("cell_grad must be 3 dim [N,C,2] N is the number of cells, C is the number of variables")
        
        if cell_grad is not None:

            first_order_correction = (
                torch.matmul(mesh_pos_to_centriod.unsqueeze(1).unsqueeze(1),cell_grad[cells_node].unsqueeze(-1))
                .squeeze()
            )
            
            aggrate_cell_attr = (cell_phi[cells_index] + first_order_correction) * weight

        else:
            aggrate_cell_attr = cell_phi[cells_index] * weight

        cell_to_node = scatter_add(aggrate_cell_attr, cells_node, dim=0) / scatter_add(
            weight, cells_node, dim=0
        )

        return cell_to_node

    # a and b has to be the same size
    def chain_dot_product(self, a, b):
        return torch.sum(a * b, dim=-1, keepdim=True)

    # 4dim a dot product 2dim b
    def chain_vector_dot_product(self, a, b):
        # nabla_u = self.chain_dot_product(a[...,0:2],b)
        # nabla_v = self.chain_dot_product(a[...,2:4],b)
        rt_val = []
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


    def interpolating_phic_to_faces(
        self,
        cell_phi=None,
        edge_neighbour_index=None,
        cells_face=None,
        edge_Euclidean_distance=None,
        cells_area=None,
        unv=None,
        cells_type=None,
        face_type=None,
        edge_center_pos=None,
        centroid=None,
        include_pressure=False,
    ):
        # mask_face_interior = ((face_type==NodeType.NORMAL)|(face_type==NodeType.OUTFLOW)|(face_type==NodeType.WALL_BOUNDARY)|(face_type==NodeType.INFLOW)).view(-1,1).long()

        total_area = (
            cells_area[edge_neighbour_index[0]] + cells_area[edge_neighbour_index[1]]
        )

        phi_f_hat = torch.index_select(cell_phi, 0, edge_neighbour_index[0]) * (
            cells_area[edge_neighbour_index[0]] / total_area
        ) + torch.index_select(cell_phi, 0, edge_neighbour_index[1]) * (
            cells_area[edge_neighbour_index[1]] / total_area
        )

        nabala_phi_c = self.intergre_f2c_2d(
            phi_f=phi_f_hat,
            cells_face=cells_face,
            edge_Euclidean_distance=edge_Euclidean_distance,
            cells_area=cells_area,
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
                cells_area=cells_area,
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
    ):
        raise NotImplementedError
    
    
    def rhie_chow_interpolation(
        self,
    ):
        raise NotImplementedError
