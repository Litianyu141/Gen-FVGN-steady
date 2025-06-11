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
from FVMmodel.FVdiscretization.FVInterpolation import Interplot
from FVMmodel.FVdiscretization.FVflux import FV_flux
from FVMmodel.FVdiscretization.FVgrad import weighted_lstsq
from torch_geometric.nn import global_add_pool,global_mean_pool

class Intergrator(FV_flux):
    def __init__(self, ):
        """
        Integrator class for finite volume schemes.
        Inherits from FV_flux. Used for integrating fluxes and assembling system equations.
        """
        super(Intergrator, self).__init__()
        self.epoch = 0

    def _fix_face_flux_BC(self, face_flux, face_type, y_node, face_node):
        """
        Fixes the face flux at boundary conditions.
        Args:
            face_flux (Tensor): [N_faces, D] Face flux values.
            face_type (Tensor): [N_faces] Face type (boundary, inflow, etc).
            y_node (Tensor): [N_nodes, D] Node values.
            face_node (Tensor): [2, N_faces] Node indices for each face.
        Returns:
            Tensor: Modified face_flux with BC applied.
        """
        mask_inflow = (face_type == NodeType.INFLOW).squeeze()
        mask_wall = (face_type == NodeType.WALL_BOUNDARY).squeeze()
        y_face = (y_node[face_node[0]]+y_node[face_node[1]])/2.
        face_flux[mask_inflow,0:2] = y_face[mask_inflow,0:2]
        face_flux[mask_wall,0:2] = 0.
        return face_flux

    def conserved_form(
        self,
        uvp_new=None,
        uv_hat=None,
        uv_old=None,
        uvp_collection=None,
        grad_phi=None,
        hessian_phi=None,
        graph_node=None,
        graph_cell_x=None,
        graph_edge=None,
        graph_cell=None,
        graph_Index=None,
        ncn_smooth=None,
    ):
        """
        Compute the conserved form of the finite volume equations.
        Args:
            uvp_new (Tensor): [N_cells, C] New cell values.
            uv_hat (Tensor): [N_cells, C] Intermediate cell values.
            uv_old (Tensor): [N_cells, C] Previous cell values.
            uvp_collection (Tensor): [N_cells, C] Collection of cell variables.
            grad_phi (Tensor): [N_cells, C, 2] cell gradients.
            hessian_phi (Tensor): [N_cells, C, 2, 2] cell Hessians.
            graph_node, graph_cell_x, graph_edge, graph_cell, graph_Index: Graph data objects.
            ncn_smooth (bool): Whether to smooth node values for visualization.
        Returns:
            Tuple of loss terms and interpolated node/cell values.
        """
        # prepare face neighbor cell`s index
        cells_node = graph_node.face
        cells_face = graph_edge.face
        cells_index = graph_cell.cells_index
        cells_area = graph_cell.cells_area.view(-1,1)
        cells_face_unv = graph_cell.cells_face_unv.view(-1,2)
        face_node = graph_node.edge_index
        face_type = graph_edge.face_type
        face_center_pos = graph_edge.pos
        face_area = graph_edge.face_area.view(-1,1)

        cell_type = graph_cell.cell_type
        cpd_centroid = graph_cell.cpd_centroid
        cpd_neighbor_cell = graph_cell.edge_index
        mask_interior_face = (face_type==NodeType.NORMAL).squeeze()
        mask_boundary_face = ~mask_interior_face
        mask_interior_cell = (cell_type==NodeType.NORMAL).squeeze()
        mask_boundary_cell = ~mask_interior_cell
        cells_face_surface_vec = cells_face_unv * ((face_area.view(-1,1))[cells_face])

        # pde coefficent
        theta_PDE_cell = graph_Index.theta_PDE[graph_cell.batch[mask_interior_cell]]
        unsteady_coefficent = theta_PDE_cell[:, 0:1]
        continuity_eq_coefficent = theta_PDE_cell[:, 1:2]
        convection_coefficent = theta_PDE_cell[:, 2:3]
        grad_p_coefficent = theta_PDE_cell[:, 3:4]
        diffusion_coefficent = theta_PDE_cell[:, 4:5]
        source_term = theta_PDE_cell[:, 5:6] * cells_area
        dt_cell = graph_Index.dt_graph[graph_cell.batch, :]

        """>>> Interpolation >>>"""
        uvp_cell_new = uvp_new
        uv_cell_hat = uv_hat
        uv_cell_old = uv_old

        grad_phi_cell_new_hat = grad_phi
        grad_uvp_cell_new = grad_phi[:,0:3]
        grad_uv_cell_hat = grad_phi[:,3:5]
        
        phi_face_new_hat = torch.full((cpd_neighbor_cell.shape[1],uvp_collection.shape[1]),0.,device=grad_phi.device)
        phi_face_new_hat[mask_interior_face] = self.interpolating_phic_to_faces(
            phi_cell=uvp_collection[:,0:5], # [Cell，Nc] Cell是单元数, Nc是通道数
            grad_phi_cell=grad_phi_cell_new_hat[:,0:5], # [Cell，Nc，2] Cell是单元数, Nc是通道数, 2为2维情况下梯度分量
            neighbor_cell=cpd_neighbor_cell[:, mask_interior_face], # [2, Ec] Ec表示单元与单元之间的邻接关系，且为单向边
            centroid=cpd_centroid, # [Cell, 2] 单元中心位置
            face_center_pos=face_center_pos[mask_interior_face], # [2, Ec] Ec表示每一个面的中心坐标，其维度大小且元素index与neighbor_cell中的元素index保持一直
        )
        phi_face_new_hat[mask_boundary_face] = uvp_collection[mask_boundary_cell]
        
        grad_phi_face_new_hat = torch.full((cpd_neighbor_cell.shape[1],grad_phi.shape[1],2),0.,device=grad_phi.device)
        grad_phi_face_new_hat[mask_interior_face] = self.interpolating_gradients_to_faces(
            phi_cell=uvp_collection[:,0:5], # [Cell，Nc] Cell是单元数, Nc是通道数
            grad_phi_cell=grad_phi_cell_new_hat[:,0:5], # [Cell，Nc，2] Cell是单元数, Nc是通道数, 2为2维情况下梯度分量
            neighbor_cell=cpd_neighbor_cell[:, mask_interior_face], # [2, Ec] Ec表示单元与单元之间的邻接关系，且为单向边
            centroid=cpd_centroid, # [Cell, 2] 单元中心位置
            face_center_pos=face_center_pos[mask_interior_face], # [2, Ec] Ec表示每一个面的中心坐标，其维度大小且元素index与neighbor_cell中的元素index保持一直
        )
        grad_phi_face_new_hat[mask_boundary_face] = grad_phi_cell_new_hat[mask_boundary_cell]

        grad_uvp_face_new = grad_phi_face_new_hat[:,0:3]
        grad_uv_face_hat = grad_phi_face_new_hat[:,3:5]
        
        uv_face_new = phi_face_new_hat[:, 0:2]
        uv_face_hat = phi_face_new_hat[:, 3:5]
        p_face_new = phi_face_new_hat[:, 2:3]
        """<<< Interpolation <<<"""

        """>>> pressure outlet >>>"""
        cells_face_outflow_mask = (
            face_type[cells_face] == NodeType.OUTFLOW
        ).squeeze() 
        if cells_face_outflow_mask.any():
            viscosity_force_pressure_outlet = (
                diffusion_coefficent[cells_index]
                * torch.matmul(
                    grad_uvp_face_new[cells_face, 0:2],
                    cells_face_surface_vec.unsqueeze(2),
                ).squeeze()
            )
            surface_p = p_face_new[cells_face, :] * cells_face_surface_vec
            loss_press = (viscosity_force_pressure_outlet - surface_p)[cells_face_outflow_mask]
            loss_press = torch.sqrt(
                global_add_pool(
                    (loss_press)**2, 
                    batch=graph_edge.batch[cells_face[cells_face_outflow_mask]],
                    size=graph_edge.batch.max(dim=0).values + 1
                ).sum(dim=-1,keepdim=True)
            )
        else:
            loss_press = torch.zeros((graph_cell.num_graphs,1),device=p_face_new.device)
        """<<< pressure outlet <<<"""

        """>>> unsteady term >>>"""
        unsteady_cell = ((uvp_cell_new[:, 0:2] - uv_cell_old) / dt_cell)[mask_interior_cell] * cells_area
        """<<< unsteady term <<<"""

        """>>> conserved continuity equation >>>"""
        loss_cont = (
            scatter_add(
                torch.matmul(
                    uv_face_new[cells_face, None, 0:2], cells_face_surface_vec[:,:,None]
                ).squeeze(),
                cells_index,
                dim=0,
                dim_size=graph_cell.pos.shape[0],
            ).view(-1,1)
        )
        loss_cont = torch.sqrt(
            global_add_pool(
                (loss_cont)**2, batch=graph_cell.batch[mask_interior_cell]
            )
        )*graph_Index.theta_PDE[:,1:2]
        """<<< conserved continuity equation <<<"""

        """>>> conserved convection flux >>>"""
        uu_flux = torch.matmul(
            uv_face_hat[:,:,None],uv_face_hat[:,None,:]
        )
        convection_flux = uu_flux[cells_face] * \
            convection_coefficent[cells_index].unsqueeze(1)
        # [N, DIM, 2], DIM is axis dimension
        """<<< conserved convection flux <<<"""

        """>>> viscous flux >>>"""
        vis_flux = grad_uv_face_hat[cells_face] * \
                diffusion_coefficent[cells_index,None]
        # [N, DIM, 2], DIM is axis dimension
        """<<< viscous flux <<<"""

        """>>> pressure term >>>"""
        # 使用P_flux会在四边形网格上导致交错现象发生
        P_flux = torch.diag_embed(p_face_new[cells_face].expand(-1, 2))*\
            grad_p_coefficent[cells_index,None]
        """<<< pressure term <<<"""

        """>>> total_flux >>>"""
        J_flux = torch.matmul(
            convection_flux + P_flux - vis_flux, cells_face_surface_vec.unsqueeze(-1)
        ).squeeze()
        """<<< total_flux <<<"""
        
        total_RHS = (
            scatter_add(
                J_flux,
                cells_index,
                dim=0,
                dim_size=graph_cell.pos.shape[0],
            ) - source_term
        )
        loss_momtentum = unsteady_coefficent * unsteady_cell + total_RHS
        
        loss_momtentum = torch.sqrt(
            global_add_pool(
                (loss_momtentum)**2, batch=graph_cell.batch[mask_interior_cell]
            )
        )*graph_Index.sigma[:,0:2]
        
        loss_mom_x = loss_momtentum[:,0:1]
        loss_mom_y = loss_momtentum[:,1:2]
        
        # interpolate uvp_new_ell to node for smooth visualization
        if ncn_smooth:

            rt_uvp_new = self.cell_to_node_2nd_order(
                cell_phi=uvp_cell_new[:,0:3],
                cell_grad=None,
                cells_node=cells_node,
                cells_index=cells_index,
                centroid=graph_cell.pos,
                mesh_pos=graph_node.pos,
            )
            # rt_uvp_new = torch.cat((uv_new,uvp_new[:,2:3]),dim=-1)
        else:
            rt_uvp_new = uvp_new

        return (
            loss_cont,
            loss_mom_x,
            loss_mom_y,
            loss_press,
            rt_uvp_new,
            uvp_cell_new,
        )

    def non_conserved_form(
        self,
        uvp_new=None,
        uv_hat=None,
        uv_old=None,
        uvp_collection=None,
        grad_phi=None,
        hessian_phi=None,
        graph_node=None,
        graph_cell_x=None,
        graph_edge=None,
        graph_cell=None,
        graph_Index=None,
        ncn_smooth=None,
    ):
        """
        Compute the non-conserved form of the finite volume equations.
        Args:
            uvp_new (Tensor): [N_nodes, C] New node values.
            uv_hat (Tensor): [N_nodes, C] Intermediate node values.
            uv_old (Tensor): [N_nodes, C] Previous node values.
            uvp_collection (Tensor): [N_nodes, C] Collection of node variables.
            grad_phi (Tensor): [N_nodes, C, 2] Node gradients.
            hessian_phi (Tensor): [N_nodes, C, 2, 2] Node Hessians.
            graph_node, graph_cell_x, graph_edge, graph_cell, graph_Index: Graph data objects.
            ncn_smooth (bool): Whether to smooth node values for visualization.
        Returns:
            Tuple of loss terms and interpolated node/cell values.
        """
        # prepare face neighbor cell`s index
        cells_node = graph_node.face
        cells_face = graph_edge.face
        cells_index = graph_cell.cells_index
        cells_area = graph_cell.cells_area.view(-1,1)
        cells_face_unv = graph_cell.cells_face_unv.view(-1,2)
        face_node = graph_node.edge_index
        face_type = graph_edge.face_type
        face_center_pos = graph_edge.pos
        face_area = graph_edge.face_area.view(-1,1)

        cell_type = graph_cell.cell_type
        cpd_centroid = graph_cell.cpd_centroid
        cpd_neighbor_cell = graph_cell.edge_index
        mask_interior_face = (face_type==NodeType.NORMAL).squeeze()
        mask_boundary_face = ~mask_interior_face
        mask_interior_cell = (cell_type==NodeType.NORMAL).squeeze()
        mask_boundary_cell = ~mask_interior_cell
        cells_face_surface_vec = cells_face_unv * ((face_area.view(-1,1))[cells_face])

        # pde coefficent
        theta_PDE_cell = graph_Index.theta_PDE[graph_cell.batch]
        unsteady_coefficent = theta_PDE_cell[:, 0:1]
        continuity_eq_coefficent = theta_PDE_cell[:, 1:2]
        convection_coefficent = theta_PDE_cell[:, 2:3]
        grad_p_coefficent = theta_PDE_cell[:, 3:4]
        diffusion_coefficent = theta_PDE_cell[:, 4:5]
        source_term = theta_PDE_cell[mask_interior_cell, 5:6] * cells_area
        dt_cell = graph_Index.dt_graph[graph_cell.batch, :]

        """>>> Interpolation >>>"""
        uvp_cell_new = uvp_new
        uv_cell_hat = uv_hat
        uv_cell_old = uv_old

        grad_phi_cell_new_hat = grad_phi
        grad_uvp_cell_new = grad_phi[:,0:3]
        grad_uv_cell_hat = grad_phi[:,3:5]
        
        phi_face_new_hat = torch.full((cpd_neighbor_cell.shape[1],uvp_collection.shape[1]),0.,device=grad_phi.device)
        phi_face_new_hat[mask_interior_face] = self.interpolating_phic_to_faces(
            phi_cell=uvp_collection[:,0:5], # [Cell，Nc] Cell是单元数, Nc是通道数
            grad_phi_cell=grad_phi_cell_new_hat[:,0:5], # [Cell，Nc，2] Cell是单元数, Nc是通道数, 2为2维情况下梯度分量
            neighbor_cell=cpd_neighbor_cell[:, mask_interior_face], # [2, Ec] Ec表示单元与单元之间的邻接关系，且为单向边
            centroid=cpd_centroid, # [Cell, 2] 单元中心位置
            face_center_pos=face_center_pos[mask_interior_face], # [2, Ec] Ec表示每一个面的中心坐标，其维度大小且元素index与neighbor_cell中的元素index保持一直
        )
        phi_face_new_hat[mask_boundary_face] = uvp_collection[mask_boundary_cell]
        
        grad_phi_face_new_hat = torch.full((cpd_neighbor_cell.shape[1],grad_phi.shape[1],2),0.,device=grad_phi.device)
        grad_phi_face_new_hat[mask_interior_face] = self.interpolating_gradients_to_faces(
            phi_cell=uvp_collection[:,0:5], # [Cell，Nc] Cell是单元数, Nc是通道数
            grad_phi_cell=grad_phi_cell_new_hat[:,0:5], # [Cell，Nc，2] Cell是单元数, Nc是通道数, 2为2维情况下梯度分量
            neighbor_cell=cpd_neighbor_cell[:, mask_interior_face], # [2, Ec] Ec表示单元与单元之间的邻接关系，且为单向边
            centroid=cpd_centroid, # [Cell, 2] 单元中心位置
            face_center_pos=face_center_pos[mask_interior_face], # [2, Ec] Ec表示每一个面的中心坐标，其维度大小且元素index与neighbor_cell中的元素index保持一直
        )
        grad_phi_face_new_hat[mask_boundary_face] = grad_phi_cell_new_hat[mask_boundary_cell]

        grad_uvp_face_new = grad_phi_face_new_hat[:,0:3]
        grad_uv_face_hat = grad_phi_face_new_hat[:,3:5]
        
        p_face_new = phi_face_new_hat[:, 2:3]
        """<<< Interpolation <<<"""

        """>>> pressure outlet >>>"""
        cells_face_outflow_mask = (
            face_type[cells_face] == NodeType.OUTFLOW
        ).squeeze() 
        if cells_face_outflow_mask.any():
            viscosity_force_pressure_outlet = (
                diffusion_coefficent[cells_index]
                * torch.matmul(
                    grad_uvp_face_new[cells_face, 0:2],
                    cells_face_surface_vec.unsqueeze(2),
                ).squeeze()
            )
            surface_p = p_face_new[cells_face, :] * cells_face_surface_vec
            loss_press = (viscosity_force_pressure_outlet - surface_p)[cells_face_outflow_mask]
            loss_press = torch.sqrt(
                global_add_pool(
                    (loss_press)**2, 
                    batch=graph_edge.batch[cells_face[cells_face_outflow_mask]],
                    size=graph_edge.batch.max(dim=0).values + 1
                ).sum(dim=-1,keepdim=True)
            )
        else:
            loss_press = torch.zeros((graph_cell.num_graphs,1),device=p_face_new.device)
        """<<< pressure outlet <<<"""

        """>>> unsteady term >>>"""
        unsteady_cell = ((uvp_cell_new[:, 0:2] - uv_cell_old) / dt_cell)[mask_interior_cell] * cells_area
        """<<< unsteady term <<<"""

        """>>> Grad-based continuity equation >>>"""
        loss_cont = (
            (grad_uvp_cell_new[mask_interior_cell, 0:1, 0] + grad_uvp_cell_new[mask_interior_cell, 1:2, 1])
            * cells_area
        )
        loss_cont = torch.sqrt(
            global_add_pool(
                (loss_cont)**2, batch=graph_cell.batch[mask_interior_cell]
            )
        )*graph_Index.theta_PDE[:,1:2]
        """<<< Grad-based continuity equation <<<"""

        """>>> Grad-based convection term >>>"""
        convection_cell = (
            torch.matmul(grad_uv_cell_hat[mask_interior_cell], uv_cell_hat[mask_interior_cell].unsqueeze(2)).squeeze()
            * cells_area
        )
        """<<< Grad-based convection term  <<<"""

        """>>> grad p term >>>"""
        volume_integrate_P = grad_uvp_cell_new[mask_interior_cell, 2] * cells_area
        """<<< grad p term  <<<"""

        """>>> Divergence-based diffusion term >>>"""
        viscosity_force_cells_face = torch.matmul(
            grad_uv_face_hat[cells_face, 0:2],
            cells_face_surface_vec.unsqueeze(2),
        ).squeeze()

        viscosity_force = calc_cell_centered_with_node_attr(
            node_attr=viscosity_force_cells_face,
            cells_node=cells_face,
            cells_index=cells_index,
            reduce="sum",
            map=False,
        )
        """<<< diffusion term  <<<"""

        loss_mom = (
            unsteady_coefficent[mask_interior_cell] * unsteady_cell
            + convection_coefficent[mask_interior_cell] * convection_cell
            + grad_p_coefficent[mask_interior_cell] * volume_integrate_P
            - diffusion_coefficent[mask_interior_cell] * viscosity_force
            - source_term
        )

        loss_mom = torch.sqrt(
            global_add_pool(
                (loss_mom)**2, batch=graph_cell.batch[mask_interior_cell]
            )
        )*graph_Index.sigma[:,0:2]
        
        loss_mom_x = loss_mom[:,0:1]
        loss_mom_y = loss_mom[:,1:2]

        # interpolate uvp_new_ell to node for smooth visualization
        if ncn_smooth:

            rt_uvp_new = self.cell_to_node_2nd_order(
                cell_phi=uvp_cell_new[mask_interior_cell,0:3],
                cell_grad=None,
                cells_node=cells_node,
                cells_index=cells_index,
                centroid=graph_cell.pos,
                mesh_pos=graph_node.pos,
            )
            # rt_uvp_new = uv_new
        else:
            rt_uvp_new = uvp_new
            
        return (
            loss_cont,
            loss_mom_x,
            loss_mom_y,
            loss_press,
            rt_uvp_new,
            uvp_cell_new,
        )

    # @torch.compile
    def forward(
        self,
        uvp_new_cell=None,
        uv_hat_cell=None,
        uv_old_cell=None,
        graph_node=None,
        graph_cell_x=None,
        graph_edge=None,
        graph_cell=None,
        graph_Index=None,
        params=None,
    ):
        """
        Forward pass for the integrator. Reconstructs gradients and computes loss terms.
        Args:
            uvp_new_cell (Tensor): [N_cells, C] New cell values.
            uv_hat_cell (Tensor): [N_cells, C] Intermediate cell values.
            uv_old_cell (Tensor): [N_cells, C] Previous cell values.
            graph_node, graph_cell_x, graph_edge, graph_cell, graph_Index: Graph data objects.
            params: Parameter object with attributes like order, conserved_form, ncn_smooth.
        Returns:
            Tuple of loss terms and interpolated cell/cell values.
        """
        """>>> Reconstruct Gradient >>>"""
        # 1st reconstruct cell to cell x gradient
        uvp_new_uv_hat = torch.cat(
            (uvp_new_cell[:, 0:3], uv_hat_cell[:, 0:2]),
            dim=-1,
        )

        grad_phi_larg = weighted_lstsq(
            phi_node=uvp_new_uv_hat,
            edge_index=graph_cell_x.neighbor_cell_x,
            mesh_pos=graph_cell.cpd_centroid,
            order=params.order,
            precompute_Moments=[graph_cell_x.A_cell_to_cell, graph_cell_x.single_B_cell_to_cell],
        )  # return: [N, C, 2] ,2 is the grad dimension， if higher order method was used
           # it returns [N,C,5](2nd), [N,C,9](3rd), [N,C,14](4th)
           
        grad_phi = grad_phi_larg[:, :, 0:2]  # return: [N, C, 2], 2 is u_x, u_y
        hessian_phi = None  
        """<<< Reconstruct Gradient <<<"""

        if params.conserved_form:

            (
                continutiy_eq,
                loss_momtentum_x,
                loss_momtentum_y,
                loss_press,
                uvp_node_new,
                uvp_cell_new,
            ) = self.conserved_form(
                uvp_new=uvp_new_cell,
                uv_hat=uv_hat_cell,
                uv_old=uv_old_cell,
                uvp_collection=uvp_new_uv_hat,
                grad_phi=grad_phi,
                hessian_phi=hessian_phi,
                graph_node=graph_node,
                graph_cell_x=graph_cell_x,
                graph_edge=graph_edge,
                graph_cell=graph_cell,
                graph_Index=graph_Index,
                ncn_smooth=params.ncn_smooth,
            )
        else:
            (
                continutiy_eq,
                loss_momtentum_x,
                loss_momtentum_y,
                loss_press,
                uvp_node_new,
                uvp_cell_new,
            ) = self.non_conserved_form(
                uvp_new=uvp_new_cell,
                uv_hat=uv_hat_cell,
                uv_old=uv_old_cell,
                uvp_collection=uvp_new_uv_hat,
                grad_phi=grad_phi,
                hessian_phi=hessian_phi,
                graph_node=graph_node,
                graph_cell_x=graph_cell_x,
                graph_edge=graph_edge,
                graph_cell=graph_cell,
                graph_Index=graph_Index,
                ncn_smooth=params.ncn_smooth,
            )

        return (
            continutiy_eq,
            loss_momtentum_x,
            loss_momtentum_y,
            loss_press,
            uvp_node_new,
            uvp_cell_new,
        )
