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
from FVMmodel.FVdiscretization.FVgrad import node_based_WLSQ
from torch_geometric.nn import global_add_pool,global_mean_pool

class Intergrator(FV_flux):
    def __init__(self, ):
        super(Intergrator, self).__init__()
        self.epoch = 0
        
    def _fix_face_flux_BC(self, face_flux, face_type, y_node, face_node):
        
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
        graph_node_x=None,
        graph_edge=None,
        graph_cell=None,
        graph_Index=None,
        ncn_smooth=None,
    ):
        # prepare face neighbour cell`s index
        cells_node = graph_node.face
        cells_face = graph_edge.face
        cells_index = graph_cell.face
        cells_area = graph_cell.cells_area.view(-1,1)
        cells_face_unv = graph_cell.cells_face_unv.view(-1,2)
        face_node = graph_node.edge_index
        face_type = graph_edge.face_type
        face_area = graph_edge.face_area
        theta_PDE_cell = graph_Index.theta_PDE[graph_cell.batch]
        cells_face_surface_vec = cells_face_unv * (face_area.view(-1,1))[cells_face]

        # pde coefficent
        unsteady_coefficent = theta_PDE_cell[:, 0:1]
        continuity_eq_coefficent = theta_PDE_cell[:, 1:2]
        convection_coefficent = theta_PDE_cell[:, 2:3]
        grad_p_coefficent = theta_PDE_cell[:, 3:4]
        diffusion_coefficent = theta_PDE_cell[:, 4:5]
        source_term = theta_PDE_cell[:, 5:6] * cells_area
        dt_cell = graph_Index.dt_graph[graph_cell.batch, :]

        """>>> Interpolation >>>"""
        phi_cell = self.node_to_cell_2nd_order(
            node_phi=uvp_collection,
            node_grad=grad_phi,
            node_hessian=hessian_phi,
            graph_node=graph_node,
            graph_cell=graph_cell,
        )

        phi_face = self.node_to_face_2nd_order(
            node_phi=uvp_collection[:, 0:5],
            node_grad=grad_phi[:, 0:5],
            node_hessian=hessian_phi[:, 0:5] if hessian_phi is not None else None,
            graph_node=graph_node,
            graph_edge=graph_edge,
        )

        nabla_phi_face_collection = self.node_to_face_2nd_order(
            node_phi=grad_phi[:, 0:5],
            node_grad=hessian_phi[:, 0:5] if hessian_phi is not None else None,
            graph_node=graph_node,
            graph_edge=graph_edge,
        )

        ''' >>> fix face flux BC >>> '''
        uv_face_new = self._fix_face_flux_BC(
            face_flux = phi_face[:, 0:2], face_type=face_type, y_node=graph_node.y, face_node=face_node
        )
        uv_face_hat = self._fix_face_flux_BC(
            face_flux = phi_face[:, 3:5], face_type=face_type, y_node=graph_node.y, face_node=face_node
        )
        # uv_face_new = phi_face[:, 0:2]
        # uv_face_hat = phi_face[:, 3:5]
        ''' <<< fix face flux BC <<< '''

        p_face_new = phi_face[:, 2:3]
        
        uvp_cell_new = phi_cell[:, 0:3]
        uv_cell_old = phi_cell[:, 5:7]
        
        nabla_uvp_face = nabla_phi_face_collection[:, 0:3]
        nabla_uv_face_hat = nabla_phi_face_collection[:, 3:5]
        """<<< Interpolation <<<"""

        """>>> pressure outlet >>>"""
        cells_face_outflow_mask = (
            face_type[cells_face] == NodeType.OUTFLOW
        ).squeeze()
        if cells_face_outflow_mask.any():
            viscosity_force_pressure_outlet = (
                diffusion_coefficent[cells_index]
                * torch.matmul(
                    nabla_uvp_face[cells_face, 0:2],
                    cells_face_surface_vec.unsqueeze(2),
                ).squeeze()
            )
            surface_p = p_face_new[cells_face] * cells_face_surface_vec
            loss_press = (viscosity_force_pressure_outlet - surface_p)[cells_face_outflow_mask]
            
            loss_press = torch.sqrt(
                global_add_pool(
                    (loss_press)**2, batch=graph_edge.batch[cells_face[cells_face_outflow_mask]]
                ).sum(dim=-1,keepdim=True)
            )
        else:
            loss_press = torch.zeros((graph_cell.num_graphs,1),device=p_face_new.device)
        """<<< pressure outlet <<<"""

        """>>> unsteady term >>>"""
        unsteady_cell = ((uvp_cell_new[:, 0:2] - uv_cell_old) / dt_cell) * cells_area
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
                (loss_cont)**2, batch=graph_cell.batch
            )
        )*graph_Index.theta_PDE[:,1:2]
        """<<< conserved continuity equation <<<"""

        """>>> conserved convection flux >>>"""
        # u_uv_face = (uv_face_hat[:, 0:1] * uv_face_hat).unsqueeze(1)
        # v_uv_face = (uv_face_hat[:, 1:2] * uv_face_hat).unsqueeze(1)
        # uu_flux = torch.cat((u_uv_face, v_uv_face), dim=1)
        uu_flux = torch.matmul(
            uv_face_hat[:,:,None],uv_face_hat[:,None,:]
        )
        convection_flux = uu_flux[cells_face] * \
            convection_coefficent[cells_index].unsqueeze(1)
        # [N, DIM, 2], DIM is axis dimension
        """<<< conserved convection flux <<<"""

        """>>> viscous flux >>>"""
        vis_flux = nabla_uv_face_hat[cells_face] * \
                diffusion_coefficent[cells_index,None]
        # [N, DIM, 2], DIM is axis dimension
        """<<< viscous flux <<<"""

        """>>> pressure term >>>"""
        # 使用P_flux会在四边形网格上导致交错现象发生
        P_flux = torch.diag_embed(p_face_new[cells_face].expand(-1, 2))*\
            grad_p_coefficent[cells_index,None]
        volume_integrate_P=0.
        
        # # 直接将navla_p_cell作为体积积分项,多边形网格会失效
        # P_flux=0
        # grad_p_cell = self.node_to_cell_2nd_order(
        #     node_phi=grad_phi[:,2],
        #     node_grad=hessian_phi[:,2] if hessian_phi is not None else None,
        #     graph_node=graph_node,
        #     graph_cell=graph_cell,
        # ).squeeze()
        # volume_integrate_P = grad_p_coefficent * grad_p_cell * cells_area
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
            )
            + volume_integrate_P - source_term
        )
        loss_momtentum = unsteady_coefficent * unsteady_cell + total_RHS
        
        loss_momtentum = torch.sqrt(
            global_add_pool(
                (loss_momtentum)**2, batch=graph_cell.batch
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
        graph_node_x=None,
        graph_edge=None,
        graph_cell=None,
        graph_Index=None,
        ncn_smooth=None,
    ):
        # prepare face neighbour cell`s index
        cells_node = graph_node.face
        cells_face = graph_edge.face
        cells_index = graph_cell.face
        cells_area = graph_cell.cells_area.view(-1,1)
        cells_face_unv = graph_cell.cells_face_unv.view(-1,2)
        face_node = graph_node.edge_index
        face_type = graph_edge.face_type
        face_area = graph_edge.face_area.view(-1,1)
        theta_PDE_cell = graph_Index.theta_PDE[graph_cell.batch]
        cells_face_surface_vec = cells_face_unv * ((face_area.view(-1,1))[cells_face])

        # pde coefficent
        unsteady_coefficent = theta_PDE_cell[:, 0:1]
        continuity_eq_coefficent = theta_PDE_cell[:, 1:2]
        convection_coefficent = theta_PDE_cell[:, 2:3]
        grad_p_coefficent = theta_PDE_cell[:, 3:4]
        diffusion_coefficent = theta_PDE_cell[:, 4:5]
        source_term = theta_PDE_cell[:, 5:6] * cells_area
        dt_cell = graph_Index.dt_graph[graph_cell.batch, :]

        """>>> Interpolation >>>"""
        phi_cell = self.node_to_cell_2nd_order(
            node_phi=uvp_collection,
            node_grad=grad_phi,
            node_hessian=hessian_phi,
            graph_node=graph_node,
            graph_cell=graph_cell,
        )
        
        uvp_cell_new = phi_cell[:, 0:3]
        uv_cell_hat = phi_cell[:, 3:5]
        uv_cell_old= phi_cell[:, 5:7]
        
        phi_face = self.node_to_face_2nd_order(
            node_phi=uvp_collection[:, 0:5],
            node_grad=grad_phi[:, 0:5],
            node_hessian=hessian_phi[:, 0:5] if hessian_phi is not None else None,
            graph_node=graph_node,
            graph_edge=graph_edge,
        )
        
        # uvp_face_new = phi_face[:, 0:3]
        p_face_new = phi_face[:, 2:3]

        nabla_phi_face_collection = self.node_to_face_2nd_order(
            node_phi=grad_phi[:, 0:5],
            node_grad=hessian_phi[:, 0:5] if hessian_phi is not None else None,
            graph_node=graph_node,
            graph_edge=graph_edge,
        )

        nabla_phi_cell_collection = self.node_to_cell_2nd_order(
            node_phi=grad_phi[:, 0:5],
            node_grad=hessian_phi[:, 0:5] if hessian_phi is not None else None,
            graph_node=graph_node,
            graph_cell=graph_cell,
        )

        nabla_uvp_face, nabla_uvp_cell = (
            nabla_phi_face_collection[:, 0:3],
            nabla_phi_cell_collection[:, 0:3],
        )

        nabla_uv_face_hat, nabla_uv_cell_hat = (
            nabla_phi_face_collection[:, 3:5],
            nabla_phi_cell_collection[:, 3:5],
        )
        """<<< Interpolation <<<"""

        """>>> pressure outlet >>>"""
        cells_face_outflow_mask = (
            face_type[cells_face] == NodeType.OUTFLOW
        ).squeeze()
        if cells_face_outflow_mask.any():
            viscosity_force_pressure_outlet = (
                diffusion_coefficent[cells_index]
                * torch.matmul(
                    nabla_uvp_face[cells_face, 0:2],
                    cells_face_surface_vec.unsqueeze(2),
                ).squeeze()
            )
            surface_p = p_face_new[cells_face, :] * cells_face_surface_vec
            loss_press = (viscosity_force_pressure_outlet - surface_p)[cells_face_outflow_mask]
            loss_press = torch.sqrt(
                global_add_pool(
                    (loss_press)**2, batch=graph_edge.batch[cells_face[cells_face_outflow_mask]]
                ).sum(dim=-1).sum(dim=-1,keepdim=True)
            )
        else:
            loss_press = torch.zeros((graph_cell.num_graphs,1),device=p_face_new.device)
        """<<< pressure outlet <<<"""

        """>>> unsteady term >>>"""
        unsteady_cell = ((uvp_cell_new[:, 0:2] - uv_cell_old) / dt_cell) * cells_area
        """<<< unsteady term <<<"""

        """>>> Grad-based continuity equation >>>"""
        loss_cont = (
            (nabla_uvp_cell[:, 0:1, 0] + nabla_uvp_cell[:, 1:2, 1])
            * cells_area
        )
        loss_cont = torch.sqrt(
            global_add_pool(
                (loss_cont)**2, batch=graph_cell.batch
            )
        )*graph_Index.theta_PDE[:,1:2]
        
        # loss_cont_hat = (
        #     (nabla_uv_cell_hat[:, 0:1, 0] + nabla_uv_cell_hat[:, 1:2, 1])
        #     * cells_area
        # )
        # loss_cont_hat = torch.sqrt(
        #     global_add_pool(
        #         (loss_cont_hat)**2, batch=graph_cell.batch
        #     )
        # )*graph_Index.theta_PDE[:,1:2]
        # loss_cont += loss_cont_hat
        """<<< Grad-based continuity equation <<<"""
        
        """>>> conserved continuity equation >>>"""
        # loss_cont = (
        #     scatter_add(
        #         torch.matmul(
        #             uv_face_new[cells_face, None, 0:2], cells_face_surface_vec[:,:,None]
        #         ).squeeze(),
        #         cells_index,
        #         dim=0,
        #         dim_size=graph_cell.pos.shape[0],
        #     ).view(-1,1)
        #     * continuity_eq_coefficent
        # )
        # loss_cont = torch.sqrt(
        #     global_add_pool(
        #         (loss_cont)**2, batch=graph_cell.batch
        #     )
        # )
        """<<< conserved continuity equation <<<"""
        
        """>>> Grad-based convection term >>>"""
        convection_cell = (
            torch.matmul(nabla_uv_cell_hat, uv_cell_hat.unsqueeze(2)).squeeze()
            * cells_area
        )
        """<<< Grad-based convection term  <<<"""

        """>>> grad p term >>>"""
        volume_integrate_P = nabla_uvp_cell[:, 2] * cells_area
        """<<< grad p term  <<<"""

        """>>> Divergence-based diffusion term >>>"""
        viscosity_force_cells_face = torch.matmul(
            nabla_uv_face_hat[cells_face, 0:2],
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
            unsteady_coefficent * unsteady_cell
            + convection_coefficent * convection_cell
            + grad_p_coefficent * volume_integrate_P
            - diffusion_coefficent * viscosity_force
            - source_term
        )

        loss_mom = torch.sqrt(
            global_add_pool(
                (loss_mom)**2, batch=graph_cell.batch
            )
        )*graph_Index.sigma[:,0:2]
        
        loss_mom_x = loss_mom[:,0:1]
        loss_mom_y = loss_mom[:,1:2]

        # interpolate uvp_new_ell to node for smooth visualization
        if ncn_smooth:

            uv_new = self.cell_to_node_2nd_order(
                cell_phi=uvp_cell_new[:,0:3],
                cell_grad=None,
                cells_node=cells_node,
                cells_index=cells_index,
                centroid=graph_cell.pos,
                mesh_pos=graph_node.pos,
            )
            rt_uvp_new = uv_new
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
        uvp_new_node=None,
        uv_hat_node=None,
        uv_old_node=None,
        graph_node=None,
        graph_node_x=None,
        graph_edge=None,
        graph_cell=None,
        graph_Index=None,
        params=None,
    ):
        """>>> Reconstruct Gradient >>>"""
        # 1st reconstruct node to node x gradient
        uvp_new_uv_hat_uv_old = torch.cat(
            (uvp_new_node[:, 0:3], uv_hat_node[:, 0:2], uv_old_node[:,0:2]),
            dim=-1,
        )

        grad_phi_larg = node_based_WLSQ(
            phi_node=uvp_new_uv_hat_uv_old,
            edge_index=graph_node_x.support_edge,
            mesh_pos=graph_node.pos,
            dual_edge=False,
            order=params.order,
            precompute_Moments=[graph_node_x.A_node_to_node,graph_node_x.B_node_to_node],
        )  # return: [N, C, 2] ,2 is the grad dimension， if higher order method was used
           # it returns [N,C,5](2nd), [N,C,9](3rd), [N,C,14](4th)
           
        grad_phi = grad_phi_larg[:, :, 0:2]  # return: [N, C, 2], 2 is u_x, u_y
        
        # if params.order != "1st":
        #     hessian_phi = torch.stack(
        #     (
        #         torch.stack((grad_phi_larg[:,:,2],grad_phi_larg[:,:,4]),dim=2), # [N,C,[uxx,uxy]]
        #         torch.stack((grad_phi_larg[:,:,4],grad_phi_larg[:,:,3]),dim=2)
        #     ), dim=2) # [N,C,2,2]
        # else:
        #     hessian_phi = None
        hessian_phi = None  
        # hessian_phi = node_based_WLSQ(
        #      phi_node=torch.cat([grad_phi[:,idx,:] for idx in range(grad_phi.shape[1])],dim=-1),
        #      edge_index=graph_node_x.support_edge,
        #      mesh_pos=graph_node.pos,
        #      dual_edge=False,
        #      order=params.order,
        #      precompute_Moments=[graph_node_x.A_node_to_node,graph_node_x.B_node_to_node],
        # )  # return: [N, 2*C, 2], 2*C is u_x, u_y, v_x, v_y, p_x, p_y
        # hessian_phi = hessian_phi.view(hessian_phi.shape[0], -1, 2, 2) # return: [N, C, 2, 2],
        # hessian_phi=None
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
                uvp_new=uvp_new_node,
                uv_hat=uv_hat_node,
                uv_old=uv_old_node,
                uvp_collection=uvp_new_uv_hat_uv_old,
                grad_phi=grad_phi,
                hessian_phi=hessian_phi,
                graph_node=graph_node,
                graph_node_x=graph_node_x,
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
                uvp_new=uvp_new_node,
                uv_hat=uv_hat_node,
                uv_old=uv_old_node,
                uvp_collection=uvp_new_uv_hat_uv_old,
                grad_phi=grad_phi,
                hessian_phi=hessian_phi,
                graph_node=graph_node,
                graph_node_x=graph_node_x,
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
