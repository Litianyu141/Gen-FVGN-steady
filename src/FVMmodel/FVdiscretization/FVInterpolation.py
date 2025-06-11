import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import torch
import torch.nn as nn
import torch.jit as jit
from typing import Optional, Tuple
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

class Interplot(nn.Module):
    """
    A comprehensive interpolation module for finite volume methods.
    
    This class provides various interpolation schemes for transferring values between
    different locations in a computational mesh (nodes, faces, and cell centers).
    It supports first-order and second-order accurate interpolation schemes as well
    as RBF-based interpolation.
    
    The interpolation methods are based on the finite volume discretization techniques
    described in Moukalled et al., "The Finite Volume Method in Computational Fluid 
    Dynamics" (2016).
    
    Attributes:
        mesh_pos (torch.Tensor): Node positions for RBF interpolation
        centroid (torch.Tensor): Cell center positions for RBF interpolation  
        cells_node (torch.Tensor): Cell-to-node connectivity for RBF interpolation
        cells_index (torch.Tensor): Cell indices for RBF interpolation
    """
    
    def __init__(
        self, 
        mesh_pos: Optional[torch.Tensor] = None,
        centroid: Optional[torch.Tensor] = None,
        cells_node: Optional[torch.Tensor] = None,
        cells_index: Optional[torch.Tensor] = None
    ):
        """
        Initialize the interpolation module.
        
        Args:
            mesh_pos (torch.Tensor, optional): Node positions [num_nodes, 2]
            centroid (torch.Tensor, optional): Cell center positions [num_cells, 2]
            cells_node (torch.Tensor, optional): Cell-to-node connectivity
            cells_index (torch.Tensor, optional): Cell indices
        """
        super().__init__()

        # Store mesh data for RBF interpolation
        self.mesh_pos = mesh_pos
        self.centroid = centroid
        self.cells_node = cells_node
        self.cells_index = cells_index

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
            cells_index = graph_cell.cells_index
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
        face_phi: torch.Tensor = None,
        face_node: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Interpolate values from face centers to nodes using scatter-based averaging.
        
        This method takes face-centered values and distributes them to connected nodes
        using a simple averaging scheme. Each node receives the average of all faces
        connected to it.
        
        Args:
            face_phi (torch.Tensor): Face-centered values [num_faces, num_channels]
            face_node (torch.Tensor): Face-to-node connectivity [2, num_faces], where
                                    face_node[0] and face_node[1] contain the two nodes
                                    connected to each face
                                    
        Returns:
            torch.Tensor: Node-centered values [num_nodes, num_channels]
            
        Raises:
            TypeError: If required arguments are None
            ValueError: If input tensors have invalid shapes
            
        Note:
            The input connectivity must be single-way (not bidirectional).
        """
        # Input validation
        if face_phi is None:
            raise TypeError("face_phi cannot be None")
        if face_node is None:
            raise TypeError("face_node cannot be None")
            
        # Shape validation
        if not isinstance(face_phi, torch.Tensor) or face_phi.dim() != 2:
            raise ValueError("face_phi must be a 2D tensor [num_faces, num_channels]")
        if not isinstance(face_node, torch.Tensor) or face_node.shape[0] != 2:
            raise ValueError("face_node must have shape [2, num_faces]")
            
        num_faces = face_phi.shape[0]
        if face_node.shape[1] != num_faces:
            raise ValueError("face_node and face_phi must have same number of faces")
            
        # Check for valid node indices
        max_node_idx = torch.max(face_node)
        min_node_idx = torch.min(face_node)
        if min_node_idx < 0:
            raise ValueError(f"face_node contains negative indices: min={min_node_idx}")
            
        # Create bidirectional connectivity for scatter operation
        # Each face contributes to both of its nodes
        bidirectional_face_phi = torch.cat((face_phi, face_phi), dim=0)
        bidirectional_node_indices = torch.cat((face_node[1], face_node[0]), dim=0)
        
        # Average face values at each node
        node_phi = scatter_mean(
            bidirectional_face_phi,
            bidirectional_node_indices,
            dim=0,
        )

        return node_phi
    
    def face_to_cell(
        self,
        face_phi: torch.Tensor = None,
        cells_face: torch.Tensor = None,
        cells_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Interpolate values from face centers to cell centers using averaging.
        
        This method takes face-centered values and averages them to get cell-centered
        values. Each cell receives the average of all faces that belong to it.
        
        Args:
            face_phi (torch.Tensor): Face-centered values [num_faces, num_channels]
            cells_face (torch.Tensor): Cell-to-face connectivity, where each element
                                     indicates which face belongs to which cell
            cells_index (torch.Tensor): Cell indices for proper grouping
                                       
        Returns:
            torch.Tensor: Cell-centered values [num_cells, num_channels]
            
        Raises:
            TypeError: If required arguments are None
            ValueError: If input tensors have invalid shapes
        """
        # Input validation
        if face_phi is None:
            raise TypeError("face_phi cannot be None")
        if cells_face is None:
            raise TypeError("cells_face cannot be None")
        if cells_index is None:
            raise TypeError("cells_index cannot be None")
            
        # Shape validation
        if not isinstance(face_phi, torch.Tensor) or face_phi.dim() != 2:
            raise ValueError("face_phi must be a 2D tensor [num_faces, num_channels]")
        if not isinstance(cells_face, torch.Tensor):
            raise ValueError("cells_face must be a tensor")
        if not isinstance(cells_index, torch.Tensor):
            raise ValueError("cells_index must be a tensor")

        # Calculate cell-centered attributes using utility function
        cell_center_attr = calc_cell_centered_with_node_attr(
            node_attr=face_phi,
            cells_node=cells_face,
            cells_index=cells_index,
            reduce="mean",
            map=True,
        )

        return cell_center_attr

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
        mesh_pos_to_centroid = mesh_pos[cells_node] - centroid[cells_index]

        weight = 1.0 / torch.norm(mesh_pos_to_centroid, dim=-1, keepdim=True)

        if (cell_grad is not None):
            if len(cell_grad.size()) < 3:
                raise ValueError("cell_grad must be 3 dim [N,C,2] N is the number of cells, C is the number of variables")
        
        if cell_grad is not None:

            first_order_correction = (
                torch.matmul(mesh_pos_to_centroid.unsqueeze(1).unsqueeze(1),cell_grad[cells_node].unsqueeze(-1))
                .squeeze()
            )
            
            aggrate_cell_attr = (cell_phi[cells_index] + first_order_correction) * weight

        else:
            aggrate_cell_attr = cell_phi[cells_index] * weight

        cell_to_node = scatter_add(aggrate_cell_attr, cells_node, dim=0) / scatter_add(
            weight, cells_node, dim=0
        )

        return cell_to_node

    def interpolating_gradients_to_faces(
        self,
        phi_cell: torch.Tensor = None,
        grad_phi_cell: torch.Tensor = None,
        neighbor_cell: torch.Tensor = None,
        centroid: torch.Tensor = None,
        face_center_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Interpolate gradients from cell centers to face centers using high-order interpolation.
        
        This method implements the gradient interpolation scheme described in Moukalled et al.,
        "The Finite Volume Method in Computational Fluid Dynamics", Section 9.4, page 289.
        
        Args:
            phi_cell (torch.Tensor): Cell-centered scalar values [num_cells, num_channels]
            grad_phi_cell (torch.Tensor): Cell-centered gradients [num_cells, num_channels, 2] 
                                          for 2D case, 2 is grad value per axis
            neighbor_cell (torch.Tensor): Cell adjacency [2, num_edges], where neighbor_cell[0] 
                                        contains sender cells (C) and neighbor_cell[1] contains 
                                        receiver cells (F)
            centroid (torch.Tensor): Cell center positions [num_cells, 2]
            face_center_pos (torch.Tensor): Face center positions [num_edges, 2]
            
        Returns:
            torch.Tensor: Face-centered gradients [num_edges, num_channels, 2]
            
        Raises:
            ValueError: If input tensors have invalid shapes or contain invalid values
            TypeError: If required arguments are None
            
        References:
            Moukalled, F., Mangani, L., & Darwish, M. (2016). The finite volume method 
            in computational fluid dynamics. Springer, Section 9.4.
        """
        # Input validation
        if phi_cell is None:
            raise TypeError("phi_cell cannot be None")
        if grad_phi_cell is None:
            raise TypeError("grad_phi_cell cannot be None")
        if neighbor_cell is None:
            raise TypeError("neighbor_cell cannot be None")
        if centroid is None:
            raise TypeError("centroid cannot be None")
        if face_center_pos is None:
            raise TypeError("face_center_pos cannot be None")
            
        # Shape validation
        if not isinstance(phi_cell, torch.Tensor) or phi_cell.dim() != 2:
            raise ValueError("phi_cell must be a 2D tensor [num_cells, num_channels]")
        if not isinstance(grad_phi_cell, torch.Tensor) or grad_phi_cell.dim() != 3:
            raise ValueError("grad_phi_cell must be a 3D tensor [num_cells, num_channels, 2]")
        if not isinstance(neighbor_cell, torch.Tensor) or neighbor_cell.shape[0] != 2:
            raise ValueError("neighbor_cell must have shape [2, num_edges]")
        if not isinstance(centroid, torch.Tensor) or centroid.shape[1] != 2:
            raise ValueError("centroid must have shape [num_cells, 2]")
        if not isinstance(face_center_pos, torch.Tensor) or face_center_pos.shape[1] != 2:
            raise ValueError("face_center_pos must have shape [num_edges, 2]")
            
        # Consistency checks
        num_cells = phi_cell.shape[0]
        num_channels = phi_cell.shape[1]
        num_edges = neighbor_cell.shape[1]
        
        if grad_phi_cell.shape[0] != num_cells:
            raise ValueError("grad_phi_cell and phi_cell must have same number of cells")
        if grad_phi_cell.shape[1] != num_channels:
            raise ValueError("grad_phi_cell and phi_cell must have same number of channels")
        if grad_phi_cell.shape[2] != 2:
            raise ValueError("grad_phi_cell must have 2 gradient components for 2D case")
        if centroid.shape[0] != num_cells:
            raise ValueError("centroid and phi_cell must have same number of cells")
        if face_center_pos.shape[0] != num_edges:
            raise ValueError("face_center_pos and neighbor_cell must have same number of edges")
            
        # Check for valid cell indices
        max_cell_idx = torch.max(neighbor_cell)
        if max_cell_idx >= num_cells:
            raise ValueError(f"neighbor_cell contains invalid indices: max={max_cell_idx}, num_cells={num_cells}")
            
        # edge_neighbor_index[0] is C,edge_neighbor_index[1] is F
        C_senders,F_recivers = neighbor_cell[0],neighbor_cell[1] # 对应Moukalled书中的C和F

        phi_cell = phi_cell[:,:,None]
        
        #开始计算gC和gF
        CF = (centroid[F_recivers]-centroid[C_senders])[:,:,None]
        dCF = torch.norm(CF,dim=1,keepdim=True)
        gC = (torch.norm(centroid[F_recivers]-face_center_pos,dim=1,keepdim=True)[:,:,None]/dCF)
        gF = 1.-gC
        
        grad_f_hat = grad_phi_cell[C_senders]*gC+grad_phi_cell[F_recivers]*gF
        
        eCF = CF/dCF
        correction = ((phi_cell[F_recivers]-phi_cell[C_senders])/dCF-grad_f_hat@eCF)*(eCF.transpose(1,2))

        return (grad_f_hat+correction).squeeze()


    def interpolating_phic_to_faces(
        self,
        phi_cell: torch.Tensor = None,
        grad_phi_cell: torch.Tensor = None,
        neighbor_cell: torch.Tensor = None,
        centroid: torch.Tensor = None,
        face_center_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Interpolate scalar values from cell centers to face centers using high-order interpolation.
        
        This method implements the scalar interpolation scheme described in Moukalled et al.,
        "The Finite Volume Method in Computational Fluid Dynamics", Section 9.2, page 276.
        The interpolation uses distance-weighted linear interpolation with gradient-based correction.
        
        Args:
            phi_cell (torch.Tensor): Cell-centered scalar values [num_cells, num_channels]
            grad_phi_cell (torch.Tensor): Cell-centered gradients [num_cells, num_channels, 2] 
                                          for 2D case, 2 is grad value per axis
            neighbor_cell (torch.Tensor): Cell adjacency [2, num_edges], where neighbor_cell[0] 
                                        contains sender cells (C) and neighbor_cell[1] contains 
                                        receiver cells (F)
            centroid (torch.Tensor): Cell center positions [num_cells, 2]
            face_center_pos (torch.Tensor): Face center positions [num_edges, 2]
            
        Returns:
            torch.Tensor: Face-centered scalar values [num_edges, num_channels]
            
        Raises:
            ValueError: If input tensors have invalid shapes or contain invalid values
            TypeError: If required arguments are None
            
        References:
            Moukalled, F., Mangani, L., & Darwish, M. (2016). The finite volume method 
            in computational fluid dynamics. Springer, Section 9.2.
        """
        # Input validation
        if phi_cell is None:
            raise TypeError("phi_cell cannot be None")
        if grad_phi_cell is None:
            raise TypeError("grad_phi_cell cannot be None")
        if neighbor_cell is None:
            raise TypeError("neighbor_cell cannot be None")
        if centroid is None:
            raise TypeError("centroid cannot be None")
        if face_center_pos is None:
            raise TypeError("face_center_pos cannot be None")
            
        # Shape validation
        if not isinstance(phi_cell, torch.Tensor) or phi_cell.dim() != 2:
            raise ValueError("phi_cell must be a 2D tensor [num_cells, num_channels]")
        if not isinstance(grad_phi_cell, torch.Tensor) or grad_phi_cell.dim() != 3:
            raise ValueError("grad_phi_cell must be a 3D tensor [num_cells, num_channels, 2]")
        if not isinstance(neighbor_cell, torch.Tensor) or neighbor_cell.shape[0] != 2:
            raise ValueError("neighbor_cell must have shape [2, num_edges]")
        if not isinstance(centroid, torch.Tensor) or centroid.shape[1] != 2:
            raise ValueError("centroid must have shape [num_cells, 2]")
        if not isinstance(face_center_pos, torch.Tensor) or face_center_pos.shape[1] != 2:
            raise ValueError("face_center_pos must have shape [num_edges, 2]")
            
        # Consistency checks
        num_cells = phi_cell.shape[0]
        num_channels = phi_cell.shape[1]
        num_edges = neighbor_cell.shape[1]
        
        if grad_phi_cell.shape[0] != num_cells:
            raise ValueError("grad_phi_cell and phi_cell must have same number of cells")
        if grad_phi_cell.shape[1] != num_channels:
            raise ValueError("grad_phi_cell and phi_cell must have same number of channels")
        if grad_phi_cell.shape[2] != 2:
            raise ValueError("grad_phi_cell must have 2 gradient components for 2D case")
        if centroid.shape[0] != num_cells:
            raise ValueError("centroid and phi_cell must have same number of cells")
        if face_center_pos.shape[0] != num_edges:
            raise ValueError("face_center_pos and neighbor_cell must have same number of edges")
            
        # Check for valid cell indices
        max_cell_idx = torch.max(neighbor_cell)
        if max_cell_idx >= num_cells:
            raise ValueError(f"neighbor_cell contains invalid indices: max={max_cell_idx}, num_cells={num_cells}")
        
        # edge_neighbor_index[0] is C,edge_neighbor_index[1] is F
        C_senders,F_recivers = neighbor_cell[0],neighbor_cell[1] # 对应Moukalled书中的C和F， pdf中276页

        phi_cell = phi_cell[:,:,None]
        
        #开始计算gC和gF
        dCF = torch.norm(centroid[F_recivers]-centroid[C_senders],dim=1,keepdim=True)
        gC = (torch.norm(centroid[F_recivers]-face_center_pos,dim=1,keepdim=True)/dCF)[:,:,None]
        gF = 1.-gC
        
        eCf = (face_center_pos-centroid[C_senders])[:,:,None]
        eFf = (face_center_pos-centroid[F_recivers])[:,:,None]
        phi_f_hat = phi_cell[C_senders]*gC+phi_cell[F_recivers]*gF
        correction = gC*grad_phi_cell[C_senders]@eCf+gF*grad_phi_cell[F_recivers]@eFf
        
        return (phi_f_hat+correction).squeeze()

    def rbf_interpolate(
        self,
        phi_values: torch.Tensor,  # [N_source, C] 源点值
        source_pos: torch.Tensor,  # [N_source, 2] 源点位置
        target_pos: torch.Tensor,  # [N_target, 2] 目标点位置
        source_indices: torch.Tensor,  # [N_connections] 源点索引
        target_indices: torch.Tensor,  # [N_connections] 目标点索引
        k: int = 4,  # 每个目标点的邻居数
        shape_param: float = 0.23  # RBF形状参数
    ) -> torch.Tensor:
        """
        通用极速RBF插值函数 - 可处理任意源到目标的插值
        
        Args:
            phi_values: 源点值 [N_source, C]
            source_pos: 源点位置 [N_source, 2]
            target_pos: 目标点位置 [N_target, 2]
            source_indices: 源点索引 [N_connections]
            target_indices: 目标点索引 [N_connections]
            k: 每个目标点的邻居数
            shape_param: RBF形状参数
            
        Returns:
            目标点处的插值结果 [N_target, C]
            
        使用示例:
            # Node到Cell插值
            cell_values = self.rbf_interpolate_universal(
                phi_values=node_phi, source_pos=mesh_pos, target_pos=centroid,
                source_indices=cells_node, target_indices=cells_index
            )
            
            # Cell到Node插值  
            node_values = self.rbf_interpolate_universal(
                phi_values=cell_phi, source_pos=centroid, target_pos=mesh_pos,
                source_indices=cells_index, target_indices=cells_node
            )
        """
        n_target = target_pos.size(0)
        n_features = phi_values.size(1)
        
        # 重组数据 - 直接reshape，最大化内存访问效率
        source_pos_neighbors = source_pos[source_indices].view(n_target, k, 2)  # [N_target, k, 2]
        source_phi_neighbors = phi_values[source_indices].view(n_target, k, n_features)  # [N_target, k, C]
        
        # 计算源点间距离矩阵 - 极度优化的向量化计算
        neighbors_diff = source_pos_neighbors.unsqueeze(2) - source_pos_neighbors.unsqueeze(1)  # [N_target, k, k, 2]
        distances_squared = torch.sum(neighbors_diff * neighbors_diff, dim=-1)  # [N_target, k, k]
        
        # RBF核矩阵 - 使用固定形状参数
        shape_param_sq = shape_param * shape_param
        kernel = torch.sqrt(distances_squared + shape_param_sq)  # [N_target, k, k]
        
        # 批量求解RBF系数 - 使用torch.linalg.solve的批量版本
        coeffs = torch.linalg.solve(kernel, source_phi_neighbors)  # [N_target, k, C]
        
        # 计算目标点到源点的距离 - 一步到位
        target_pos_expanded = target_pos[target_indices].view(n_target, k, 2)  # [N_target, k, 2]
        target_diff = target_pos_expanded - source_pos_neighbors  # [N_target, k, 2]
        target_distances_squared = torch.sum(target_diff * target_diff, dim=-1)  # [N_target, k]
        
        # 目标点的核值
        kernel_target = torch.sqrt(target_distances_squared + shape_param_sq).unsqueeze(-1)  # [N_target, k, 1]
        
        # 最终插值计算 - 极致优化的矩阵乘法
        result = torch.sum(kernel_target * coeffs, dim=1)  # [N_target, C]
        
        return result
