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
from Post_process.to_vtk import write_hybrid_mesh_to_vtu_2D,write_to_vtk,to_pv_cells_nodes_and_cell_types
from FVMmodel.FVdiscretization.FVgrad import node_based_WLSQ
from FVMmodel.FVdiscretization.FVInterpolation import Interplot
from Load_mesh.Load_mesh import H5CFDdataset, CFDdatasetBase

class Data_Pool:
    def __init__(self, params=None,device=None,state_save_dir=None,):
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
        
        # 绘制被重置的这个case当前状态
        self._plot_env=True
        self.intp = Interplot()
        
    def _set_reset_env_flag(self, flag=False, rst_time=1):
        self.reset_env_flag = flag
        self.rst_time = rst_time

    def load_mesh_to_cpu(
        self,
        dataset_dir=None,
    ):
        
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
            num_workers=4,
            pin_memory=False,
            collate_fn=lambda x: x,
        )

        print("loading whole dataset to cpu")
        self.meta_pool = []
        self.uvp_node_pool = []
        start_idx = 0
        while True:
            for _, trajs in enumerate(mesh_loader):
                tmp = list(trajs)
                for meta_data, init_uvp_node in tmp:
                    meta_data["global_idx"] = torch.arange(start_idx,start_idx+init_uvp_node.shape[0])
                    self.meta_pool.append(meta_data)
                    self.uvp_node_pool.append(init_uvp_node)
                    start_idx += init_uvp_node.shape[0]
 
                    if len(self.meta_pool)>=self.params.dataset_size:
                        break
                    
            if len(self.meta_pool)>=self.params.dataset_size:
                break
            
        self.uvp_node_pool = torch.cat(self.uvp_node_pool, dim=0)
        self.dataset_size = len(self.meta_pool)
        self.params.dataset_size = self.dataset_size
        
        # loss_cont, loss_mom[2], loss_press, 存储第一步残差，并用于计算相对残差
        self.init_loss = torch.full((self.dataset_size,), 1.0)
        self.init_loss_mask = torch.full((self.dataset_size,), True)
        
        # 控制画图个数的文件夹分组
        self.plot_count = 0
        return self.dataset_size, self.params
    
    @staticmethod
    def datapreprocessing(
        graph_node, graph_node_x, graph_edge, graph_cell, graph_Index
    ):
        uvp_node = graph_node.x[:, 0:3]
        theta_PDE_node = graph_Index.theta_PDE[graph_node.batch]
        graph_node.x = torch.cat((uvp_node, theta_PDE_node), dim=1)
        
        return (graph_node, graph_node_x, graph_edge, graph_cell, graph_Index)
    
    def reset_env(self, plot=False):

        # 弹出第0个网格的mesh数据
        old_mesh = self.meta_pool.pop(0)
        old_global_idx = old_mesh["global_idx"]
        
        # 绘图
        if plot:
            uvp_node = self.uvp_node_pool[old_global_idx]
            
            if not ("poly" in old_mesh["case_name"]):
            # if False:
                ''' >>> plot at cell-center >>> '''
                grad_phi_larg = node_based_WLSQ(
                    phi_node=uvp_node,
                    edge_index=old_mesh["support_edge"].long(),
                    mesh_pos=old_mesh["node|pos"].to(torch.float32),
                    dual_edge=False,
                    order=self.params.order,
                )  # return: [N, C, 2] ,2 is the grad dimension， if higher order method was used
                # it returns [N,C,5](2nd), [N,C,9](3rd), [N,C,14](4th)
                
                grad_phi = grad_phi_larg[:, :, 0:2]  # return: [N, C, 2], 2 is u_x, u_y
                
                # hessian_phi = torch.stack(
                # (
                #     torch.stack((grad_phi_larg[:,:,2],grad_phi_larg[:,:,4]),dim=2), # [N,C,[uxx,uxy]]
                #     torch.stack((grad_phi_larg[:,:,4],grad_phi_larg[:,:,3]),dim=2)
                # ), dim=2) # [N,C,2,2]
                hessian_phi=None
                
                uvp_cell = self.intp.node_to_cell_2nd_order(
                    node_phi=uvp_node,
                    node_grad=grad_phi,
                    node_hessian=hessian_phi,
                    cells_node=old_mesh["cells_node"].long(),
                    cells_index=old_mesh["cells_index"].long(),
                    mesh_pos=old_mesh["node|pos"].to(torch.float32),
                    centroid=old_mesh["cell|centroid"].to(torch.float32),
                )
                self.export_to_tecplot(old_mesh, uvp_cell, datalocation="cell")
                ''' <<< plot at cell-center <<< '''
                
            else:
                ''' >>> plot at node-center >>> '''
                self.export_to_tecplot(old_mesh, uvp_node, datalocation="node")
                ''' <<< plot at node-center <<< '''
            
            self._plot_env = False

        # 移除属于第0个网格的uvp数据
        self.uvp_node_pool = self.uvp_node_pool[old_global_idx.shape[0]:] 
        self.init_loss = self.init_loss[1:]
        self.init_loss_mask = self.init_loss_mask[1:]
        
        for iidx in range(len(self.meta_pool)):
            cur_meta_data = self.meta_pool[iidx]
            cur_meta_data["global_idx"] -= old_global_idx.shape[0]

        # 接着生成新的网格数据，即重新选一个边界条件
        new_mesh, init_uvp = CFDdatasetBase.transform_mesh(
            old_mesh, 
            self.params
        )
        new_mesh["global_idx"] = torch.arange(
            self.uvp_node_pool.shape[0], self.uvp_node_pool.shape[0]+init_uvp.shape[0]
        )
        self.uvp_node_pool = torch.cat((self.uvp_node_pool, init_uvp), dim=0)
        self.init_loss = torch.cat((self.init_loss,torch.full((1,), 1)),dim=0)
        self.init_loss_mask = torch.cat((self.init_loss_mask,torch.full((1,), True)),dim=0)
        self.meta_pool.append(new_mesh)

    def export_to_tecplot(self, mesh, uvp, datalocation="node", file_name=None):
        
        # 暂时先写vtk来可视化
        mesh_pos = mesh["node|pos"]
        case_name = mesh["case_name"]
        cells_node = mesh["cells_node"].long().squeeze()
        cells_face = mesh["cells_face"].long().squeeze()
        cells_index = mesh["cells_index"].long().squeeze()
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
        
        if pv.CellType.POLYGON in pv_cells_type:
            face_node = mesh["face|face_node"].long().squeeze()
            neighbour_cell = mesh["face|neighbour_cell"].long().squeeze()
            ''' >>> test to tecplot >>> '''
            interior_zone = {"name": "Fluidfield", "rho": mesh["rho"].item(), "mu": mesh["mu"].item(), "dt": mesh["dt"].item()}
            interior_zone["node|X"] = mesh["node|pos"][:, 0:1].unsqueeze(0).numpy()
            interior_zone["node|Y"] = mesh["node|pos"][:, 1:2].unsqueeze(0).numpy()
            interior_zone[f"{datalocation}|U"] = uvp[None,:,0:1].numpy()
            interior_zone[f"{datalocation}|V"] = uvp[None,:,1:2].numpy()
            interior_zone[f"{datalocation}|P"] = uvp[None,:,2:3].numpy()
            interior_zone["cells_node"] = cells_node.unsqueeze(0).numpy()
            interior_zone["cells_index"] = cells_index.unsqueeze(0).numpy()
            interior_zone["face_node"] = face_node.transpose(0, 1).unsqueeze(0).numpy()
            interior_zone["neighbour_cell"] = neighbour_cell.transpose(0, 1).unsqueeze(0).numpy()

            write_zone = [interior_zone, None]

            # write_tecplotzone(
            #     f"{saving_path}.dat",
            #     datasets=write_zone,
            #     time_step_length=1,
            # )
            process = multiprocessing.Process(
                target=write_tecplot_in_subprocess, args=(f"{saving_path}.dat", write_zone, 1)
            )
            process.start()
            ''' <<< test to tecplot <<< '''
        else:
            write_hybrid_mesh_to_vtu_2D(
                mesh_pos=mesh_pos.cpu().numpy(), 
                data={
                    f"{datalocation}|U":uvp[:,0].cpu().numpy(),
                    f"{datalocation}|V":uvp[:,1].cpu().numpy(),
                    f"{datalocation}|P":uvp[:,2].cpu().numpy(),
                }, 
                cells_node=pv_cells_node.cpu().numpy(), 
                cells_type=pv_cells_type.cpu().numpy(),
                filename=f"{saving_path}.vtu",
            )
        
        self.plot_count+=1

    def update_env(self, mesh):
        
        mesh["time_steps"] += 1

        if "wave" in mesh["flow_type"]:
            (
                mesh,
                theta_PDE,
                sigma,
                source_pressure_node,
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
            mesh["wave_uvp_on_node"][0, :, 2:3] += source_pressure_node

            return mesh

        else: 

            mesh = CFDdatasetBase.To_Cartesian(mesh,resultion=(300,100))

        return mesh

    def payback(self, uvp_new, global_idx, new_loss=None, graph_index=None):
        
        # update uvp pool
        self.uvp_node_pool[global_idx] = uvp_new.data
        
        if new_loss is not None:
            valid_idx = graph_index[self.init_loss_mask[graph_index]]
            self.init_loss[valid_idx] = new_loss[self.init_loss_mask[graph_index]].data
            self.init_loss_mask[graph_index] = False
        
        if self.reset_env_flag:
            for _ in range(self.rst_time):
                
                # 每次都将第0个网格重置，然后生成新网格append到pool尾部
                self.reset_env(plot=self._plot_env)
                
            self.reset_env_flag=False    
            self._plot_env = True
        
class CustomGraphData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        offset_rules = {
            "edge_index": self.num_nodes,
            "face": self.num_nodes,
            "cells_node": self.num_nodes,
            "face_node": self.num_nodes,
            "cells_face": self.num_nodes,
            "neighbour_cell": self.num_nodes,
            "face_node_x": self.num_nodes,
            "support_edge": self.num_nodes,
            "periodic_idx": self.num_nodes,
            "init_loss":0,
            "case_name":0,
            "query": 0,
            "grids": 0,
            "pos": 0,
            "A_node_to_node": 0,
            "A_node_to_node_x": 0,
            "B_node_to_node": 0,
            "B_node_to_node_x": 0,
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
        cat_dim_rules = {
            "x": 0,
            "pos": 0,
            "y": 0,
            "norm_y": 0,
            "query": 0,  # 保持query为列表，不进行拼接
            "grids": 0,  # 保持query为列表，不进行拼接
            "edge_index": 1,  # edge_index保持默认的offset拼接
            "face":0,
            "voxel": 0,
            "init_loss":0,
            "support_edge":1,
            "graph_index": 0,
            "global_idx": 0,
            "periodic_idx": 1,
        }
        return cat_dim_rules.get(key, super().__cat_dim__(key, value, *args, **kwargs))
    
class GraphNodeDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool
    
    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]

        mesh_pos = minibatch_data["node|pos"].to(torch.float32)
        face_node = minibatch_data["face|face_node"].long()
        cells_node = minibatch_data["cells_node"].long()
        node_type = minibatch_data["node|node_type"].long()
        case_name = minibatch_data["case_name"]
        global_idx = minibatch_data["global_idx"].long()
        uvp_node = self.base_dataset.uvp_node_pool[global_idx]
        target_on_node = minibatch_data["target|uvp"].to(torch.float32)
        
        if "periodic_idx" in minibatch_data:
            periodic_idx = minibatch_data["periodic_idx"].long()
            face_node_with_periodic = torch.cat((face_node,periodic_idx),dim=1)
        else:
            periodic_idx = torch.full((uvp_node.shape[0],1), -1, dtype=torch.long)
            face_node_with_periodic = face_node
            
        graph_node = CustomGraphData(
            x=uvp_node,
            edge_index=face_node_with_periodic,
            edge_index_interior=face_node,
            face=cells_node,
            pos=mesh_pos,
            node_type=node_type,
            y=target_on_node,
            global_idx=global_idx,
            periodic_idx=periodic_idx,
            case_name=torch.tensor([ord(char) for char in (case_name)], dtype=torch.long),
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_node

class GraphNode_X_Dataset(InMemoryDataset):
    """This graph is undirected"""

    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
        """Optional node attr"""
        mesh_pos = minibatch_data["node|pos"].to(torch.float32)
        support_edge = minibatch_data["support_edge"].long()
        A_node_to_node = minibatch_data["A_node_to_node"].to(torch.float32)
        B_node_to_node = minibatch_data["B_node_to_node"].to(torch.float32)

        graph_node_x = CustomGraphData(
            support_edge=support_edge,
            num_nodes=mesh_pos.shape[0],
            A_node_to_node=A_node_to_node,
            B_node_to_node=B_node_to_node,
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_node_x

class GraphEdgeDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]

        # edge_attr
        face_area = minibatch_data["face|face_area"].to(torch.float32)
        face_type = minibatch_data["face|face_type"].long()
        face_center_pos = minibatch_data["face|face_center_pos"].to(torch.float32)
        cells_face = minibatch_data["cells_face"].long()

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
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]

        # cell_attr
        neighbour_cell = minibatch_data["face|neighbour_cell"].long()
        cells_area = minibatch_data["cell|cells_area"].to(torch.float32)
        centroid = minibatch_data["cell|centroid"].to(torch.float32)
        cells_face_unv = minibatch_data['unit_norm_v'].to(torch.float32)
        cells_index = minibatch_data["cells_index"].long()
        init_loss = self.base_dataset.init_loss[idx:idx+1]
        
        graph_cell = CustomGraphData(
            x=torch.empty((centroid.shape[0],3),dtype=torch.float32),
            edge_index=neighbour_cell,
            cells_face_unv=cells_face_unv,
            cells_area=cells_area,
            pos=centroid,
            # global_idx=global_idx,
            face=cells_index,
            init_loss=init_loss,
            graph_index=torch.tensor([idx],dtype=torch.long),
        )

        return graph_cell

class Graph_INDEX_Dataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.meta_pool
    
    @property
    def params(self):
        return self.base_dataset.params

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
        
        theta_PDE = minibatch_data["theta_PDE"].to(torch.float32)
        sigma = minibatch_data["sigma"].to(torch.float32)
        uvp_dim = minibatch_data["uvp_dim"].to(torch.float32)
        dt_graph = minibatch_data["dt_graph"].to(torch.float32)
        
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
        self.data_source = data_source
        self.epoch = 0
        self.specific_indices = None  # 用于存储特定的索引

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.specific_indices is not None:
            return iter(self.specific_indices)
        return iter(torch.randperm(len(self.data_source), generator=g).tolist())

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = int(datetime.datetime.now().timestamp())

    def set_specific_indices(self, indices):
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
        # 保存输入参数到实例变量
        self.graph_node_dataset = graph_node_dataset
        self.graph_node_x_dataset = graph_node_x_dataset
        self.graph_edge_dataset = graph_edge_dataset
        self.graph_cell_dataset = graph_cell_dataset
        self.graph_Index_dataset = graph_Index_dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # 初始化DataLoaders
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
        return zip(
            self.loader_A, self.loader_B, self.loader_C, self.loader_D, self.loader_E
        )

    def __len__(self):
        return min(
            len(self.loader_A),
            len(self.loader_B),
            len(self.loader_C),
            len(self.loader_D),
            len(self.loader_E),
        )

    def get_specific_data(self, indices):
        # 设置Sampler的特定索引
        self.sampler.set_specific_indices(indices)

        # 重新创建DataLoaders来使用更新的Sampler
        self.loader_A = torch_geometric_DataLoader(
            self.graph_node_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_B = torch_geometric_DataLoader(
            self.graph_node_x_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_C = torch_geometric_DataLoader(
            self.graph_edge_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_D = torch_geometric_DataLoader(
            self.graph_cell_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_E = torch_geometric_DataLoader(
            self.graph_Index_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        graph_node, graph_node_x, graph_edge, graph_cell, graph_Index = next(
            iter(self)
        )

        minibatch_data = self.graph_node_dataset.pool[indices[0]]

        origin_mesh_path = "".join(
            [chr(int(f)) for f in minibatch_data["origin_mesh_path"][0, :, 0].numpy()]
        )

        flow_type = minibatch_data["flow_type"]
        if ("cavity" in flow_type) or ("possion" in flow_type):
            has_boundary = False
        else:
            has_boundary = True

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
        self.base_dataset = Data_Pool(
            params=params,
            device=device,
            state_save_dir=state_save_dir,
        )

        self.dataset_size, self.params = self.base_dataset.load_mesh_to_cpu(
            dataset_dir=dataset_dir,
        )

    def create_datasets(self, batch_size=100, num_workers=4, pin_memory=True):
        graph_node_dataset = GraphNodeDataset(base_dataset=self.base_dataset)
        graph_node_x_dataset = GraphNode_X_Dataset(base_dataset=self.base_dataset)
        graph_edge_dataset = GraphEdgeDataset(base_dataset=self.base_dataset)
        graph_cell_dataset = GraphCellDataset(base_dataset=self.base_dataset)
        graph_Index_dataset = Graph_INDEX_Dataset(base_dataset=self.base_dataset)

        # 创建SharedSampler并将其传递给CustomDataLoader

        sampler = SharedSampler(graph_node_dataset)

        loader = CustomDataLoader(
            graph_node_dataset,
            graph_node_x_dataset,
            graph_edge_dataset,
            graph_cell_dataset,
            graph_Index_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return self.base_dataset, loader, sampler
