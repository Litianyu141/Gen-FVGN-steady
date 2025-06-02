"""Utility functions for reading the datasets."""

import sys
import os
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)

import multiprocessing
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import matplotlib
matplotlib.use("Agg")
import pyvista as pv

import torch
import numpy as np
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as torch_geometric_DataLoader
from torch.utils.data import Sampler
import datetime
from Post_process.to_tecplot import write_tecplot_in_subprocess
from Post_process.to_vtk import write_hybrid_mesh_to_vtu_2D,write_vtu_file_2D_poly_to_tri,to_pv_cells_nodes_and_cell_types
from FVMmodel.FVdiscretization.FVgrad import node_based_WLSQ
from FVMmodel.FVdiscretization.FVInterpolation import Interplot
from Load_mesh.Load_mesh import H5CFDdataset, CFDdatasetBase

class Data_Pool:
    def __init__(self, params=None,device=None,state_save_dir=None,):
        """
        Initializes the Data_Pool.

        Args:
            params: Parameters for the dataset.
            device: The device to use (e.g., 'cpu', 'cuda').
            state_save_dir: Directory to save training states.
        """
        self.params = params
        self.device = device
        
        try:
            if not (state_save_dir.find("traing_results") != -1):
                os.makedirs(f"{state_save_dir}/traing_results", exist_ok=True)
                self.state_save_dir = f"{state_save_dir}/traing_results"
        except:
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>Warning, no state_save_dir is specified, check if traing states is specified<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
        
        # Plot the current state of the case being reset
        self._plot_env=True
        self.intp = Interplot()
        
    def _set_reset_env_flag(self, flag=False, rst_time=1):
        """
        Sets the flag for resetting the environment.

        Args:
            flag (bool): The flag to indicate whether to reset the environment.
            rst_time (int): The number of times to reset.
        """
        self.reset_env_flag = flag
        self.rst_time = rst_time

    def load_mesh_to_cpu(
        self,
        dataset_dir=None,
    ):
        """
        Loads the mesh dataset to CPU.

        Args:
            dataset_dir (str): The directory of the dataset.

        Returns:
            tuple: A tuple containing the dataset size and parameters.
        """
        
        valid_h5file_paths = []
        for subdir, _, files in os.walk(dataset_dir):
            for data_name in files:
                if data_name.endswith(".h5"):
                    valid_h5file_paths.append(os.path.join(subdir, data_name))

        mesh_dataset = H5CFDdataset(
            params=self.params, file_list=valid_h5file_paths
        )

        mesh_loader = torch_DataLoader(
            mesh_dataset,
            batch_size=4,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

        print("loading whole dataset to cpu")
        self.meta_pool = []
        self.uvp_node_pool = []
        start_idx = 0
        while True:
            for _, trajs in enumerate(mesh_loader):
                tmp = list(trajs)
                for meta_data, init_uvp_node in tmp: # init_uvp_node: [num_nodes_in_sample, C]
                    meta_data["global_idx"] = torch.arange(start_idx,start_idx+init_uvp_node.shape[0])
                    self.meta_pool.append(meta_data)
                    self.uvp_node_pool.append(init_uvp_node)
                    start_idx += init_uvp_node.shape[0]
 
                    if len(self.meta_pool)>=self.params.dataset_size:
                        break
                    
            if len(self.meta_pool)>=self.params.dataset_size:
                break
            
        print("Successfully load whole dataset to cpu")
        
        self.uvp_node_pool = torch.cat(self.uvp_node_pool, dim=0) # [total_num_nodes, C]
        self.dataset_size = len(self.meta_pool)
        self.params.dataset_size = self.dataset_size
        
        # loss_cont, loss_mom[2], loss_press, store the first step residual and use it to calculate the relative residual
        self.init_loss = torch.full((self.dataset_size,), 1.0)
        self.init_loss_mask = torch.full((self.dataset_size,), True)
        
        # Control the folder grouping for the number of plots
        self.plot_count = 0
        return self.dataset_size, self.params
    
    @staticmethod
    def datapreprocessing(
        graph_node, graph_node_x, graph_edge, graph_cell, graph_Index
    ):
        """
        Preprocesses the graph data.

        Args:
            graph_node: Graph node data. graph_node.x: [N, C_in], where N is number of nodes.
            graph_node_x: Additional graph node data.
            graph_edge: Graph edge data.
            graph_cell: Graph cell data.
            graph_Index: Graph index data. graph_Index.theta_PDE: [batch_size, C_theta].

        Returns:
            tuple: A tuple containing the preprocessed graph data.
                   graph_node.x will be [N, 3 + C_theta].
        """
        uvp_node = graph_node.x[:, 0:3] # [N, 3]
        theta_PDE_node = graph_Index.theta_PDE[graph_node.batch] # [N, C_theta] (after broadcasting by batch)
        graph_node.x = torch.cat((uvp_node, theta_PDE_node), dim=1) # [N, 3 + C_theta]
        
        return (graph_node, graph_node_x, graph_edge, graph_cell, graph_Index)
    
    def reset_env(self, plot=False):
        """
        Resets the environment by removing the oldest mesh and adding a new one.

        Args:
            plot (bool): Whether to plot the environment state.
        """

        # Pop the mesh data of the 0-th grid
        old_mesh = self.meta_pool.pop(0)
        old_global_idx = old_mesh["global_idx"] # [num_nodes_in_old_mesh]
        
        # Plotting
        if plot:
            uvp_node = self.uvp_node_pool[old_global_idx] # [num_nodes_in_old_mesh, C]
            
            if not ("poly" in old_mesh["case_name"]):
            # if False:
                ''' >>> plot at cell-center >>> '''
                grad_phi_larg = node_based_WLSQ(
                    phi_node=uvp_node, # [N, C]
                    edge_index=old_mesh["face_node_x"].long(), # [2, num_edges]
                    extra_edge_index=old_mesh["support_edge"].long(), # [2, num_support_edges]
                    mesh_pos=old_mesh["node|pos"].to(torch.float32), # [N, D_pos]
                    order=self.params.order,
                )  # return: [N, C, K_grad_dim] , K_grad_dim depends on order
                
                grad_phi = grad_phi_larg[:, :, 0:2]  # return: [N, C, 2], 2 is u_x, u_y
                
                # hessian_phi = torch.stack(
                # (
                #     torch.stack((grad_phi_larg[:,:,2],grad_phi_larg[:,:,4]),dim=2), # [N,C,[uxx,uxy]]
                #     torch.stack((grad_phi_larg[:,:,4],grad_phi_larg[:,:,3]),dim=2)
                # ), dim=2) # [N,C,2,2]
                hessian_phi=None
                
                uvp_cell = self.intp.node_to_cell_2nd_order(
                    node_phi=uvp_node, # [N, C]
                    node_grad=grad_phi, # [N, C, 2]
                    node_hessian=hessian_phi, # None or [N, C, 2, 2]
                    cells_node=old_mesh["cells_node"].long(), # [num_cells, max_nodes_per_cell]
                    cells_index=old_mesh["cells_index"].long(), # [num_cells]
                    mesh_pos=old_mesh["node|pos"].to(torch.float32), # [N, D_pos]
                    centroid=old_mesh["cell|centroid"].to(torch.float32), # [num_cells, D_pos]
                ) # [num_cells, C]
                self.export_to_tecplot(old_mesh, uvp_cell, datalocation="cell")
                ''' <<< plot at cell-center <<< '''
                
            else:
                ''' >>> plot at node-center >>> '''
                self.export_to_tecplot(old_mesh, uvp_node, datalocation="node")
                ''' <<< plot at node-center <<< '''
            
            self._plot_env = False

        # Remove uvp data belonging to the 0-th grid
        self.uvp_node_pool = self.uvp_node_pool[old_global_idx.shape[0]:] 
        self.init_loss = self.init_loss[1:]
        self.init_loss_mask = self.init_loss_mask[1:]
        
        for iidx in range(len(self.meta_pool)):
            cur_meta_data = self.meta_pool[iidx]
            cur_meta_data["global_idx"] -= old_global_idx.shape[0]

        # Then generate new mesh data, i.e., re-select a boundary condition
        new_mesh, init_uvp = CFDdatasetBase.transform_mesh( # init_uvp: [num_new_nodes, C]
            old_mesh, 
            self.params
        )
        new_mesh["global_idx"] = torch.arange(
            self.uvp_node_pool.shape[0], self.uvp_node_pool.shape[0]+init_uvp.shape[0]
        )
        self.uvp_node_pool = torch.cat((self.uvp_node_pool, init_uvp), dim=0)
        self.init_loss = torch.cat((self.init_loss,torch.full((1,), 1.0)),dim=0) # Changed 1 to 1.0 to match existing type
        self.init_loss_mask = torch.cat((self.init_loss_mask,torch.full((1,), True)),dim=0)
        self.meta_pool.append(new_mesh)

    def export_to_tecplot(self, mesh, uvp, datalocation="node", file_name=None):
        """
        Exports data to Tecplot format, supports dynamic variable identification.

        Args:
            mesh (dict): Mesh data.
                         Contains 'node|pos': [N, D_pos], 'case_name', 'cells_node': [num_cells, max_nodes_per_cell],
                         'cells_face': [num_cells, max_faces_per_cell], 'cells_index': [num_cells], 'dt', 'source', 'aoa',
                         'Re' (optional), 'face|face_node': [num_faces, max_nodes_per_face],
                         'face|neighbour_cell': [num_faces, 2], 'rho', 'mu'.
            uvp (torch.Tensor): Main physical variable data, typically U, V, P. Shape: [num_elements, C] where num_elements
                                depends on datalocation (N for nodes, num_cells for cells).
            datalocation (str): Data location ("node" or "cell").
            file_name (str, optional): Output file name. Defaults to None.
        """
        
        # Temporarily write vtk for visualization
        mesh_pos = mesh["node|pos"] # [N, D_pos]
        case_name = mesh["case_name"]
        cells_node = mesh["cells_node"].long().squeeze() # [num_cells, max_nodes_per_cell] or [total_cell_nodes] if squeezed from 1D
        cells_face = mesh["cells_face"].long().squeeze() # [num_cells, max_faces_per_cell] or [total_cell_faces]
        cells_index = mesh["cells_index"].long().squeeze() # [num_cells] or scalar
        dt = mesh["dt"].squeeze().item()
        source = mesh["source"].squeeze().item()
        aoa = mesh["aoa"].squeeze().item()
        
        try:
            Re=mesh["Re"].squeeze().item()
        except:
            Re=0
            Warning("No Re number in the mesh set to 0")

        pv_cells_node,pv_cells_type = to_pv_cells_nodes_and_cell_types(
            cells_node=cells_node, cells_face=cells_face, cells_index=cells_index
        )
        
        if file_name is None:
            save_dir_num = self.plot_count//50
            saving_dir = f"{self.state_save_dir}/{save_dir_num*50}-{(save_dir_num+1)*50}"
            os.makedirs(saving_dir, exist_ok=True)
            saving_path = f"{saving_dir}/NO.{self.plot_count}_{case_name}_Re={Re:.2f}_dt={dt:.3f}_source={source:.2f}_aoa={aoa:.2f}"
        else:
            saving_path = file_name
        
        # Check if there is a polygon type to decide whether to use tecplot or VTU
        if pv.CellType.POLYGON in pv_cells_type:
        # if True:
            face_node = mesh["face|face_node"].long().squeeze() # [num_faces, max_nodes_per_face] or [total_face_nodes]
            neighbour_cell = mesh["face|neighbour_cell"].long().squeeze() # [num_faces, 2] or [total_neighbor_pairs]
            ''' >>> test to tecplot >>> '''
            interior_zone = {
                "name": "Fluidfield", 
                "rho": mesh["rho"].item(), 
                "mu": mesh["mu"].item(), 
                "dt": mesh["dt"].item()
            }
            
            # Add coordinate data
            interior_zone["node|X"] = mesh["node|pos"][:, 0:1].unsqueeze(0).numpy() # [1, N, 1]
            interior_zone["node|Y"] = mesh["node|pos"][:, 1:2].unsqueeze(0).numpy() # [1, N, 1]
            
            interior_zone[f"{datalocation}|U"] = uvp[None,:,0:1].numpy() # [1, num_elements, 1]
            interior_zone[f"{datalocation}|V"] = uvp[None,:,1:2].numpy() # [1, num_elements, 1]
            interior_zone[f"{datalocation}|P"] = uvp[None,:,2:3].numpy() # [1, num_elements, 1]
            interior_zone["cells_node"] = cells_node.unsqueeze(0).numpy() # [1, num_cells, max_nodes_per_cell] or [1, total_cell_nodes]
            interior_zone["cells_index"] = cells_index.unsqueeze(0).numpy() # [1, num_cells] or [1]
            interior_zone["face_node"] = face_node.transpose(0, 1).unsqueeze(0).numpy() # [1, max_nodes_per_face, num_faces] or [1, total_face_nodes_dim1, total_face_nodes_dim0]
            interior_zone["neighbour_cell"] = neighbour_cell.transpose(0, 1).unsqueeze(0).numpy() # [1, 2, num_faces] or [1, total_neighbor_pairs_dim1, total_neighbor_pairs_dim0]

            write_zone = [interior_zone, None]

            process = multiprocessing.Process(
                target=write_tecplot_in_subprocess, args=(f"{saving_path}.dat", write_zone, 1)
            )
            process.start()
            process.join()
            ''' <<< test to tecplot <<< '''
        else:
            write_hybrid_mesh_to_vtu_2D(
                mesh_pos=mesh_pos.cpu().numpy(), # [N, D_pos]
                data={
                    f"{datalocation}|U":uvp[:,0].cpu().numpy(), # [num_elements]
                    f"{datalocation}|V":uvp[:,1].cpu().numpy(), # [num_elements]
                    f"{datalocation}|P":uvp[:,2].cpu().numpy(), # [num_elements]
                }, 
                cells_node=pv_cells_node.cpu().numpy(), 
                cells_type=pv_cells_type.cpu().numpy(),
                filename=f"{saving_path}.vtu",
            )
        
        self.plot_count+=1

    def update_env(self, mesh):
        """
        Updates the environment, for example, by advancing time steps or changing boundary conditions.

        Args:
            mesh (dict): The mesh data to update.
                         Expected to have 'time_steps', 'flow_type'.
                         If 'wave' in 'flow_type', expects 'mean_u', 'rho', 'mu', 'source', 'aoa', 'dt',
                         'source_frequency', 'source_strength', 'wave_uvp_on_node'.

        Returns:
            dict: The updated mesh data.
        """
        
        mesh["time_steps"] += 1

        if "wave" in mesh["flow_type"]:
            (
                mesh,
                theta_PDE, # [batch_size_in_cfddatasetbase, C_theta]
                sigma, # [batch_size_in_cfddatasetbase, C_sigma]
                source_pressure_node, # [N, 1]
            ) = CFDdatasetBase.set_Wave_case(
                mesh,
                self.params,
                mesh["mean_u"].item(),
                mesh["rho"].item(),
                mesh["mu"].item(),
                mesh["source"].item(),
                mesh["aoa"].item(),
                mesh["dt"].item(),
                mesh["source_frequency"].item(),
                mesh["source_strength"].item(),
                time_index=mesh["time_steps"],
            )
            mesh["theta_PDE"] = theta_PDE
            mesh["sigma"] = sigma
            mesh["wave_uvp_on_node"][0, :, 2:3] += source_pressure_node # Assuming wave_uvp_on_node is [1, N, C_uvp]

            return mesh

        else: 

            mesh = CFDdatasetBase.To_Cartesian(mesh,resultion=(300,100))

        return mesh

    def payback(self, uvp_new, global_idx, new_loss=None, graph_index=None):
        """
        Updates the uvp_node_pool with new uvp values and potentially resets the environment.

        Args:
            uvp_new (torch.Tensor): New UVP data. Shape: [num_nodes_in_sample, C].
            global_idx (torch.Tensor): Global indices for the UVP data. Shape: [num_nodes_in_sample].
            new_loss (torch.Tensor, optional): New loss values. Shape: [batch_size_of_graph_index]. Defaults to None.
            graph_index (torch.Tensor, optional): Graph indices for the loss update. Shape: [batch_size_of_graph_index]. Defaults to None.
        """
        
        # update uvp pool
        self.uvp_node_pool[global_idx] = uvp_new.data
        
        if new_loss is not None:
            valid_idx = graph_index[self.init_loss_mask[graph_index]]
            self.init_loss[valid_idx] = new_loss[self.init_loss_mask[graph_index]].data
            self.init_loss_mask[graph_index] = False
        
        if self.reset_env_flag:
            for _ in range(self.rst_time):
                
                # Reset the 0-th grid each time, then generate a new grid and append it to the end of the pool
                self.reset_env(plot=self._plot_env)
                
            self.reset_env_flag=False    
            self._plot_env = True
        
class CustomGraphData(Data):
    def __init__(self, **kwargs):
        """
        Custom graph data structure inheriting from torch_geometric.data.Data.
        """
        super().__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        """
        Specifies how to increment attributes when concatenating graphs in a batch.
        This is crucial for attributes like edge_index.
        """
        
        if not hasattr(self, 'num_nodes') or self.num_nodes is None:
            # Ensure num_nodes is available for most offset rules.
            # For keys that don't require num_nodes (offset 0), this check might be too strict
            # if those keys are present in a graph without num_nodes set.
            # However, standard PyG practice is to have num_nodes for batching.
            if key not in {"init_loss", "case_name", "query", "grids", "pos", "A_node_to_node", 
                           "A_node_to_node_x", "B_node_to_node", "B_node_to_node_x", 
                           "single_B_node_to_node", "extra_B_node_to_node", "cells_area", 
                           "node_type", "graph_index", "theta_PDE", "sigma", "uvp_dim", 
                           "dt_graph", "x", "y", "m_ids", "m_gs", "global_idx"}:
                 raise ValueError("The number of nodes must be set before incrementing for key: {}".format(key))
    
        offset_rules = {
            "edge_index": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0,
            "face": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0, # Assuming face indices are node-based
            "cells_node": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0,
            "face_node": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0,
            "cells_face": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0, # Assuming cells_face refers to face indices, which might need their own offset if not node-based
            "neighbour_cell": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0, # Assuming neighbour_cell refers to cell indices, which might need their own offset
            "face_node_x": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0,
            "support_edge": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0,
            "periodic_idx": self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes is not None else 0,
            "init_loss":0,
            "case_name":0,
            "query": 0,
            "grids": 0,
            "pos": 0,
            "A_node_to_node": 0,
            "A_node_to_node_x": 0,
            "B_node_to_node": 0,
            "B_node_to_node_x": 0,
            "single_B_node_to_node":0,
            "extra_B_node_to_node":0,
            "cells_area": 0,
            "node_type": 0,
            "graph_index": 0,
            "theta_PDE": 0,
            "sigma": 0,
            "uvp_dim": 0,
            "dt_graph": 0,
            "x": 0,
            "y": 0,
            "m_ids": 0,
            "m_gs": 0,
            "global_idx": 0,
        }
        return offset_rules.get(key, super().__inc__(key, value, *args, **kwargs))

    def __cat_dim__(self, key, value, *args, **kwargs):
        """
        Specifies the dimension along which attributes should be concatenated when creating a batch.
        """
        cat_dim_rules = {
            "x": 0, # Node features: [N, F_node]
            "pos": 0, # Node positions: [N, D_pos]
            "y": 0, # Target values: [N_target, F_target] or [F_target]
            "norm_y": 0,
            "query": 0,  # Keep query as a list, do not concatenate
            "grids": 0,  # Keep query as a list, do not concatenate
            "edge_index": 1,  # Edge indices: [2, num_edges], concatenate along dim 1
            "face":0, # Cell-node connectivity or similar, typically [num_cells, nodes_per_cell] or flattened
            "voxel": 0,
            "init_loss":0, # [batch_size] or [1]
            "support_edge":1, # [2, num_support_edges]
            "face_node_x":1, # [2, num_face_node_x_edges]
            "graph_index": 0, # [batch_size] or [1]
            "global_idx": 0, # [N]
            "periodic_idx": 1, # [2, num_periodic_pairs]
        }
        return cat_dim_rules.get(key, super().__cat_dim__(key, value, *args, **kwargs))
    
class GraphNodeDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        """
        Dataset for graph nodes.

        Args:
            base_dataset (Data_Pool): The base Data_Pool instance.
        """
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        """Accesses the meta_pool from the base_dataset."""
        # Here you can filter out GraphNode data from the base class's pool as needed
        return self.base_dataset.meta_pool
    
    def len(self):
        """Returns the number of samples in the dataset."""
        return len(self.pool)

    def get(self, idx):
        """
        Gets a single graph data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            CustomGraphData: A graph data object for the specified index.
                - x (torch.Tensor): Node features (uvp_node). Shape: [N, C_uvp].
                - edge_index (torch.Tensor): Graph connectivity (face_node). Shape: [2, num_edges].
                - face (torch.Tensor): Cell to node connectivity (cells_node). Shape: [num_cells, max_nodes_per_cell].
                - pos (torch.Tensor): Node positions (mesh_pos). Shape: [N, D_pos].
                - node_type (torch.Tensor): Node types. Shape: [N].
                - y (torch.Tensor): Target node values (target_on_node). Shape: [N, C_target_uvp].
                - global_idx (torch.Tensor): Global indices of nodes. Shape: [N].
                - case_name (torch.Tensor): Case name encoded as tensor of ordinals. Shape: [L_case_name].
                - graph_index (torch.Tensor): Index of the graph in the batch. Shape: [1].
        """
        minibatch_data = self.pool[idx]

        mesh_pos = minibatch_data["node|pos"].to(torch.float32) # [N, D_pos]
        face_node = minibatch_data["face|face_node"].long() # [2, num_edges] (assuming it's edge_index like) or [num_faces, nodes_per_face]
        cells_node = minibatch_data["cells_node"].long() # [num_cells, max_nodes_per_cell]
        node_type = minibatch_data["node|node_type"].long() # [N]
        case_name = minibatch_data["case_name"]
        global_idx = minibatch_data["global_idx"].long() # [N]
        uvp_node = self.base_dataset.uvp_node_pool[global_idx] # [N, C_uvp]
        target_on_node = minibatch_data["target|uvp"].to(torch.float32) # [N, C_target_uvp]
            
        graph_node = CustomGraphData(
            x=uvp_node,
            edge_index=face_node,
            # edge_index_interior=face_node,
            face=cells_node,
            pos=mesh_pos,
            node_type=node_type,
            y=target_on_node,
            global_idx=global_idx,
            case_name=torch.tensor([ord(char) for char in (case_name)], dtype=torch.long),
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_node

class GraphNode_X_Dataset(InMemoryDataset):
    """This graph is undirected. Dataset for auxiliary node features and connectivity."""

    def __init__(self, base_dataset):
        """
        Initializes the dataset for auxiliary node features.

        Args:
            base_dataset (Data_Pool): The base Data_Pool instance.
        """
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        """Accesses the meta_pool from the base_dataset."""
        # Here you can filter out GraphNode data from the base class's pool as needed
        return self.base_dataset.meta_pool

    def len(self):
        """Returns the number of samples in the dataset."""
        return len(self.pool)

    def get(self, idx):
        """
        Gets a single graph data sample with auxiliary node features.

        Args:
            idx (int): Index of the sample.

        Returns:
            CustomGraphData: A graph data object.
                - face_node_x (torch.Tensor): Extended face-node connectivity. Shape: [2, num_face_node_x_edges].
                - support_edge (torch.Tensor): Support edge connectivity. Shape: [2, num_support_edges].
                - num_nodes (int): Number of nodes in the graph.
                - A_node_to_node (torch.Tensor): Node-to-node matrix A. Shape: [N, N] or other.
                - single_B_node_to_node (torch.Tensor): Node-to-node matrix single_B. Shape: [N, N] or other.
                - extra_B_node_to_node (torch.Tensor): Node-to-node matrix extra_B. Shape: [N, N] or other.
                - graph_index (torch.Tensor): Index of the graph in the batch. Shape: [1].
        """
        minibatch_data = self.pool[idx]
        """Optional node attr"""
        mesh_pos = minibatch_data["node|pos"].to(torch.float32) # [N, D_pos]
        face_node_x = minibatch_data["face_node_x"].long() # [2, num_face_node_x_edges]
        support_edge = minibatch_data["support_edge"].long() # [2, num_support_edges]
        A_node_to_node = minibatch_data["A_node_to_node"].to(torch.float32) # Shape depends on definition
        single_B_node_to_node = minibatch_data["single_B_node_to_node"].to(torch.float32) # Shape depends on definition
        extra_B_node_to_node = minibatch_data["extra_B_node_to_node"].to(torch.float32) # Shape depends on definition

        graph_node_x = CustomGraphData(
            face_node_x=face_node_x,
            support_edge=support_edge,
            num_nodes=mesh_pos.shape[0],
            A_node_to_node=A_node_to_node,
            single_B_node_to_node=single_B_node_to_node,
            extra_B_node_to_node=extra_B_node_to_node,
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_node_x

class GraphEdgeDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        """
        Dataset for graph edges.

        Args:
            base_dataset (Data_Pool): The base Data_Pool instance.
        """
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        """Accesses the meta_pool from the base_dataset."""
        # Here you can filter out GraphNode data from the base class's pool as needed
        return self.base_dataset.meta_pool

    def len(self):
        """Returns the number of samples in the dataset."""
        return len(self.pool)

    def get(self, idx):
        """
        Gets a single graph data sample for edge features.

        Args:
            idx (int): Index of the sample.

        Returns:
            CustomGraphData: A graph data object for edge attributes.
                - face_type (torch.Tensor): Type of each face/edge. Shape: [num_faces].
                - face_area (torch.Tensor): Area of each face/edge. Shape: [num_faces, 1] or [num_faces].
                - face (torch.Tensor): Cell to face connectivity (cells_face). Shape: [num_cells, max_faces_per_cell].
                - pos (torch.Tensor): Positions of face centers (face_center_pos). Shape: [num_faces, D_pos].
                - graph_index (torch.Tensor): Index of the graph in the batch. Shape: [1].
        """
        minibatch_data = self.pool[idx]

        # edge_attr
        face_area = minibatch_data["face|face_area"].to(torch.float32) # [num_faces, 1] or [num_faces]
        face_type = minibatch_data["face|face_type"].long() # [num_faces]
        face_center_pos = minibatch_data["face|face_center_pos"].to(torch.float32) # [num_faces, D_pos]
        cells_face = minibatch_data["cells_face"].long() # [num_cells, max_faces_per_cell]

        graph_edge = CustomGraphData(
            face_type=face_type,
            face_area=face_area,
            face=cells_face,
            pos=face_center_pos,
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_edge

class GraphCellDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        """
        Dataset for graph cells.

        Args:
            base_dataset (Data_Pool): The base Data_Pool instance.
        """
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        """Accesses the meta_pool from the base_dataset."""
        # Here you can filter out GraphNode data from the base class's pool as needed
        return self.base_dataset.meta_pool

    def len(self):
        """Returns the number of samples in the dataset."""
        return len(self.pool)

    def get(self, idx):
        """
        Gets a single graph data sample for cell features.

        Args:
            idx (int): Index of the sample.

        Returns:
            CustomGraphData: A graph data object for cell attributes.
                - x (torch.Tensor): Placeholder cell features. Shape: [num_cells, 3].
                - edge_index (torch.Tensor): Cell connectivity (neighbour_cell). Shape: [2, num_cell_edges] or [num_faces, 2].
                - cells_face_unv (torch.Tensor): Unit normal vectors for cell faces. Shape: [num_cells, max_faces_per_cell, D_norm] or other.
                - cells_area (torch.Tensor): Area of each cell. Shape: [num_cells, 1] or [num_cells].
                - pos (torch.Tensor): Cell centroids. Shape: [num_cells, D_pos].
                - face (torch.Tensor): Cell indices (cells_index). Shape: [num_cells].
                - init_loss (torch.Tensor): Initial loss for the cell/graph. Shape: [1].
                - graph_index (torch.Tensor): Index of the graph in the batch. Shape: [1].
        """
        minibatch_data = self.pool[idx]

        # cell_attr
        neighbour_cell = minibatch_data["face|neighbour_cell"].long() # [num_faces, 2] (cell adjacency through faces)
        cells_area = minibatch_data["cell|cells_area"].to(torch.float32) # [num_cells, 1] or [num_cells]
        centroid = minibatch_data["cell|centroid"].to(torch.float32) # [num_cells, D_pos]
        cells_face_unv = minibatch_data['unit_norm_v'].to(torch.float32) # Shape depends on definition, e.g., [num_cells, max_faces_per_cell, D_norm]
        cells_index = minibatch_data["cells_index"].long() # [num_cells]
        init_loss = self.base_dataset.init_loss[idx:idx+1] # [1]
        
        graph_cell = CustomGraphData(
            x=torch.empty((centroid.shape[0],3),dtype=torch.float32), # [num_cells, 3]
            edge_index=neighbour_cell, # This might represent cell-to-cell graph via faces
            cells_face_unv=cells_face_unv,
            cells_area=cells_area,
            pos=centroid,
            # global_idx=global_idx, # Not present in original
            face=cells_index, # This seems to be an identifier for cells rather than connectivity
            init_loss=init_loss,
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_cell

class Graph_INDEX_Dataset(InMemoryDataset):
    def __init__(self, base_dataset):
        """
        Dataset for graph-level index information.

        Args:
            base_dataset (Data_Pool): The base Data_Pool instance.
        """
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        """Accesses the meta_pool from the base_dataset."""
        # Here you can filter out GraphNode data from the base class's pool as needed
        return self.base_dataset.meta_pool
    
    @property
    def params(self):
        """Accesses the params from the base_dataset."""
        return self.base_dataset.params

    def len(self):
        """Returns the number of samples in the dataset."""
        return len(self.pool)

    def get(self, idx):
        """
        Gets a single graph data sample for graph-level indices/parameters.

        Args:
            idx (int): Index of the sample.

        Returns:
            CustomGraphData: A graph data object.
                - x (torch.Tensor): Index of the graph. Shape: [1].
                - theta_PDE (torch.Tensor): PDE parameters. Shape: [C_theta].
                - sigma (torch.Tensor): Sigma values. Shape: [C_sigma].
                - uvp_dim (torch.Tensor): Dimensions of UVP. Shape: [C_uvp_dim].
                - dt_graph (torch.Tensor): Timestep for the graph. Shape: [1] or other.
                - graph_index (torch.Tensor): Index of the graph in the batch. Shape: [1].
        """
        minibatch_data = self.pool[idx]
        
        theta_PDE = minibatch_data["theta_PDE"].to(torch.float32) # Shape: [C_theta]
        sigma = minibatch_data["sigma"].to(torch.float32) # Shape: [C_sigma]
        uvp_dim = minibatch_data["uvp_dim"].to(torch.float32) # Shape: [C_uvp_dim]
        dt_graph = minibatch_data["dt_graph"].to(torch.float32) # Shape: [1] or other
        
        graph_Index = CustomGraphData(
            x=torch.tensor([idx],dtype=torch.long),
            theta_PDE=theta_PDE,
            sigma=sigma,
            uvp_dim=uvp_dim,
            dt_graph=dt_graph,
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_Index

class SharedSampler(Sampler):
    def __init__(self, data_source):
        """
        A sampler that can be shared and can return specific indices if set.

        Args:
            data_source: The dataset to sample from.
        """
        self.data_source = data_source
        self.epoch = 0
        self.specific_indices = None  # Used to store specific indices

    def __iter__(self):
        """Returns an iterator over sampler indices."""
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.specific_indices is not None:
            return iter(self.specific_indices)
        return iter(torch.randperm(len(self.data_source), generator=g).tolist())

    def __len__(self):
        """Returns the number of samples in the data source."""
        return len(self.data_source)

    def set_epoch(self, epoch):
        """
        Sets the epoch for the sampler. Uses current timestamp if not a number.
        This is often used to ensure different shuffling across epochs.
        """
        try:
            self.epoch = int(epoch)
        except ValueError: # If epoch is not an int (e.g. a string from a timestamp)
            self.epoch = int(datetime.datetime.now().timestamp())


    def set_specific_indices(self, indices):
        """
        Sets specific indices for the sampler to iterate over.

        Args:
            indices (list or torch.Tensor): The specific indices to use.
        """
        self.specific_indices = indices

class CustomDataLoader:
    def __init__(
        self,
        graph_node_dataset,
        graph_node_x_dataset,
        graph_edge_dataset,
        graph_cell_dataset,
        graph_Index_dataset,
        batch_size,
        sampler,
        num_workers=4,
        pin_memory=False,
    ):
        """
        Custom DataLoader that combines multiple PyTorch Geometric DataLoaders.

        Args:
            graph_node_dataset: Dataset for graph nodes.
            graph_node_x_dataset: Dataset for auxiliary node features.
            graph_edge_dataset: Dataset for graph edges.
            graph_cell_dataset: Dataset for graph cells.
            graph_Index_dataset: Dataset for graph-level indices.
            batch_size (int): Batch size.
            sampler (Sampler): Sampler to use for all DataLoaders.
            num_workers (int): Number of worker processes for data loading.
            pin_memory (bool): If True, tensors will be copied to CUDA pinned memory.
        """
        # Save input parameters to instance variables
        self.graph_node_dataset = graph_node_dataset
        self.graph_node_x_dataset = graph_node_x_dataset
        self.graph_edge_dataset = graph_edge_dataset
        self.graph_cell_dataset = graph_cell_dataset
        self.graph_Index_dataset = graph_Index_dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Initialize DataLoaders
        self.loader_A = torch_geometric_DataLoader(
            graph_node_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_B = torch_geometric_DataLoader(
            graph_node_x_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_C = torch_geometric_DataLoader(
            graph_edge_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_D = torch_geometric_DataLoader(
            graph_cell_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_E = torch_geometric_DataLoader(
            graph_Index_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self):
        """Returns an iterator that zips the outputs of the individual DataLoaders."""
        return zip(
            self.loader_A, self.loader_B, self.loader_C, self.loader_D, self.loader_E
        )

    def __len__(self):
        """Returns the length of the DataLoader, defined by the shortest individual DataLoader."""
        return min(
            len(self.loader_A),
            len(self.loader_B),
            len(self.loader_C),
            len(self.loader_D),
            len(self.loader_E),
        )

    def get_specific_data(self, indices):
        """
        Fetches a specific batch of data corresponding to the given indices.

        Args:
            indices (list or torch.Tensor): The specific indices to fetch.

        Returns:
            tuple: Contains batched graph data (graph_node, graph_node_x, graph_edge, graph_cell, graph_Index),
                   a boolean indicating if boundary conditions are present, and the original mesh path.
        """
        # Set specific indices for Sampler
        self.sampler.set_specific_indices(indices)

        # Recreate DataLoaders to use the updated Sampler
        # This ensures that the loaders will yield the batch corresponding to 'indices' next.
        # Note: This re-initialization might have performance implications if called very frequently.
        current_batch_size = self.batch_size if self.batch_size <= len(indices) else len(indices)
        self.loader_A = torch_geometric_DataLoader(
            self.graph_node_dataset,
            current_batch_size, 
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_B = torch_geometric_DataLoader(
            self.graph_node_x_dataset,
            current_batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_C = torch_geometric_DataLoader(
            self.graph_edge_dataset,
            current_batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_D = torch_geometric_DataLoader(
            self.graph_cell_dataset,
            current_batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_E = torch_geometric_DataLoader(
            self.graph_Index_dataset,
            current_batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        graph_node, graph_node_x, graph_edge, graph_cell, graph_Index = next(
            iter(self)
        )

        # The following assumes 'indices' refers to indices in the base_dataset.pool
        # and that the first index is representative for 'origin_mesh_path' and 'flow_type'.
        minibatch_data = self.graph_node_dataset.pool[indices[0]]


        origin_mesh_path = "".join(
            [chr(int(f)) for f in minibatch_data["origin_mesh_path"][0, :, 0].numpy()]
        ) # Assuming origin_mesh_path is [1, L, 1] and contains char codes

        flow_type = minibatch_data["flow_type"]
        if ("cavity" in flow_type) or ("possion" in flow_type): # Note: "possion" might be a typo for "poisson"
            has_boundary = False
        else:
            has_boundary = True
        
        # Reset sampler to avoid using specific_indices for subsequent iterations unless set again
        self.sampler.set_specific_indices(None)


        return (
            graph_node,
            graph_node_x,
            graph_edge,
            graph_cell,
            graph_Index,
            has_boundary,
            origin_mesh_path,
        )

class DatasetFactory:
    def __init__(
        self,
        params=None,
        dataset_dir=None,
        state_save_dir=None,
        device=None,
    ):
        """
        Factory class to create various graph datasets.

        Args:
            params: Parameters for the dataset.
            dataset_dir (str): Directory of the dataset.
            state_save_dir (str): Directory to save training states.
            device: The device to use (e.g., 'cpu', 'cuda').
        """
        self.base_dataset = Data_Pool(
            params=params,
            device=device,
            state_save_dir=state_save_dir,
        )

        self.dataset_size, self.params = self.base_dataset.load_mesh_to_cpu(
            dataset_dir=dataset_dir,
        )

    def create_datasets(self, batch_size=1, num_workers=0, pin_memory=True):
        """
        Creates and returns a CustomDataLoader with all graph component datasets.

        Args:
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of worker processes.
            pin_memory (bool): If True, tensors will be copied to CUDA pinned memory.

        Returns:
            CustomDataLoader: The initialized DataLoader.
        """
        graph_node_dataset = GraphNodeDataset(base_dataset=self.base_dataset)
        graph_node_x_dataset = GraphNode_X_Dataset(base_dataset=self.base_dataset)
        graph_edge_dataset = GraphEdgeDataset(base_dataset=self.base_dataset)
        graph_cell_dataset = GraphCellDataset(base_dataset=self.base_dataset)
        graph_Index_dataset = Graph_INDEX_Dataset(base_dataset=self.base_dataset)

        # Create a shared sampler
        shared_sampler = SharedSampler(graph_node_dataset)  # Sampler can be based on any of the datasets as they share indices

        # Create the custom DataLoader
        custom_loader = CustomDataLoader(
            graph_node_dataset=graph_node_dataset,
            graph_node_x_dataset=graph_node_x_dataset,
            graph_edge_dataset=graph_edge_dataset,
            graph_cell_dataset=graph_cell_dataset,
            graph_Index_dataset=graph_Index_dataset,
            batch_size=batch_size,
            sampler=shared_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return custom_loader
