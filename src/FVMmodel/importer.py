import torch
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_min,scatter_max
from utils.utilities import NodeType
from utils.normalization import Normalizer
from FVMmodel.FVdiscretization.FVscheme import Intergrator
from timm.layers import trunc_normal_

class NNmodel(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()

        self.params = params
        if self.params.net == "Attu-FVGN2D":
            from FVMmodel.Models.AttuFVGN2D.AttuFVGN import Simulator
        elif self.params.net =='FVGN':
            from FVMmodel.Models.FVGN.GenFVGN import Simulator
        elif self.params.net =='TransFVGN_v1':
            from FVMmodel.Models.TransFVGN.TransFVGN_v1 import Simulator
        elif self.params.net =='TransFVGN_v2' or self.params.net =='TransFVGN':
            from FVMmodel.Models.TransFVGN.TransFVGN_v2 import Simulator
        
        self.simulator = Simulator(
            message_passing_num=params.message_passing_num,
            node_input_size=params.node_input_size,
            edge_input_size=params.node_input_size + 3,
            node_output_size=params.node_output_size,
            drop_out=False,
            hidden_size=params.hidden_size,
            params=params,
        )
        
        self.node_norm = Normalizer(size=params.node_input_size - params.node_phi_size,
                                    max_accumulations=params.dataset_size)
        
        self.integrator = Intergrator()
        
        self.params=params
        self.node_phi_size = params.node_phi_size # for torch.compile
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def cal_relative_edge_attr(self, graph):

        # No more permuting edge direction
        senders, receivers = graph.edge_index

        releative_x_attr = (
            graph.x[senders]
            - graph.x[receivers]
        )
        releative_x_pos = (
            graph.pos[senders]
            - graph.pos[receivers]
        )
        releative_x_attr = torch.cat(
            (
                releative_x_attr,
                releative_x_pos,
                torch.norm(releative_x_pos, p=2, dim=-1, keepdim=True),
            ),
            dim=-1,
        )

        graph.edge_attr = releative_x_attr

        return graph
    
    # def normalize_graph_features(self, x, batch):
    #     # 检查输入张量和 batch 索引的形状
    #     assert x.dim() == 2, "Input tensor x should be 2-dimensional"
    #     assert batch.dim() == 1, "Batch tensor should be 1-dimensional"
    #     assert x.size(0) == batch.size(0), "The first dimension of x and batch should be the same"

    #     mean = scatter_mean(x, batch, dim=0)
    #     residual = x - mean[batch]
    #     var = scatter_mean(residual**2, batch, dim=0)
    #     std = torch.sqrt(var)
    #     x = residual / (std[batch] + 1e-8)
    #     # x = residual
        
    #     return x
    
    def normalize_graph_features(self, x, batch):
        # 检查输入张量和 batch 索引的形状
        assert x.dim() == 2, "Input tensor x should be 2-dimensional"
        assert batch.dim() == 1, "Batch tensor should be 1-dimensional"
        assert x.size(0) == batch.size(0), "The first dimension of x and batch should be the same"

        # 计算每个 batch 的最小值和最大值
        min_val = scatter_min(x, batch, dim=0)[0]
        max_val = scatter_max(x, batch, dim=0)[0]
        # 计算每个元素的归一化值
        # range_val = max_val - min_val
        # x = ((x - min_val[batch]) / (range_val[batch] + 1e-8))*10.
        
        # 这里特别为泰勒格林涡算例去掉了除以max-min，使得神经网络可以感知到流场只在decaying
        # update: 这里不能去掉正则化，因为如果Taylor-Green一直在衰减，会导致输入的值特别小，导致模型无法训练
        x = x - min_val[batch]

        return x
    
    def update_x_attr(
        self,
        graph,
    ):

        if graph.norm_uvp: # This param was set in src/Load_mesh/Graph_loader.py --> datapreprocessing()
            graph.x[:,:self.node_phi_size] = self.normalize_graph_features(
                graph.x[:,:self.node_phi_size], 
                graph.batch
            )
            graph.norm_uvp=False
        else:
            raise ValueError("The graph node features have already been normalized, please check the graph.norm_uvp")
        
        if graph.norm_global: # This param was set in src/Load_mesh/Graph_loader.py --> datapreprocessing()
            graph.x[:,self.node_phi_size:] = self.node_norm(graph.x[:,self.node_phi_size:])
            graph.norm_global=False
        
        return graph

    def update_edge_attr(
        self,
        graph,
    ):
        # 计算relative u,v,p, pos(2维), norm(pos)， 一共6维
        graph = self.cal_relative_edge_attr(graph)

        return graph

    def _enforce_boundary_condition(self, uvp, graph_node, with_periodic=False):
        
        # 先施加周期边界条件
        if with_periodic:
            uvp[graph_node.periodic_idx[1]] = uvp[graph_node.periodic_idx[0]]
        
        mask_dirichlet  = (
            (graph_node.node_type == NodeType.WALL_BOUNDARY) | 
            (graph_node.node_type == NodeType.INFLOW)|
            (graph_node.node_type == NodeType.PRESS_POINT)|
            (graph_node.node_type == NodeType.IN_WALL)).squeeze()

        mask_press_constraint = (graph_node.node_type == NodeType.PRESS_POINT).squeeze()
        
        uvp[mask_dirichlet,0:2] = graph_node.y[mask_dirichlet,0:2]
        uvp[mask_press_constraint,2:3] = 0
        
        return uvp

    def forward(
        self,
        graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_Index,
        is_training=True,
    ):

        if is_training:
            # 取出上一时刻uv来作时间差分
            uv_old_node = (
                graph_node.x[:, 0:2] / graph_Index.uvp_dim[graph_node.batch, 0:2]
            )

            # update normalized value
            graph_node = self.update_x_attr(
                graph=graph_node,
            )

            graph_node = self.update_edge_attr(
                graph=graph_node,
            )  # 因为每次都会重复计算relative edge attr, 所以每次都必定norm edge attr

            uvp_new_node = self.simulator(graph_node, graph_edge, graph_cell)
            
            # 临时处理以下，将带有periodic的edge_index替换回原来的只有内部域的edge_index， 具体请查看src/Load_mesh/Graph_loader.py
            graph_node.edge_index = graph_node.edge_index_interior
            
            uvp_new_node = torch.tanh(uvp_new_node/10)*10
            
            uvp_new_node = self._enforce_boundary_condition(uvp_new_node, graph_node, with_periodic=True)
            
            # explicit / implicit / IMEX integration schemes
            if self.params.integrator == "explicit":
                uv_hat_node = uv_old_node[:, 0:2]

            if self.params.integrator == "implicit":
                uv_hat_node = uvp_new_node[:, 0:2]

            if self.params.integrator == "imex":
                uv_hat_node = (
                    uv_old_node[:, 0:2] + uvp_new_node[:, 0:2]
                ) / 2.0

            # Intergrate all flux at every edge`s of all cells
            (
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y,
                loss_press,
                smoothed_uvp_new_node,
                uvp_new_cell,
            ) = self.integrator(
                uvp_new_node=uvp_new_node,
                uv_hat_node=uv_hat_node,
                uv_old_node=uv_old_node,
                graph_node=graph_node,
                graph_node_x=graph_node_x,
                graph_edge=graph_edge,
                graph_cell=graph_cell,
                graph_Index=graph_Index,
                params=self.params,
            )
            
            smoothed_uvp_new_node = self._enforce_boundary_condition(
                smoothed_uvp_new_node, graph_node, with_periodic=False
            )
            
            # reverse dimless for storing
            uvp_node_new_with_dim = smoothed_uvp_new_node*graph_Index.uvp_dim[graph_node.batch]*\
                graph_Index.sigma[graph_node.batch]
            uvp_cell_new_with_dim = uvp_new_cell*graph_Index.uvp_dim[graph_cell.batch]*\
                graph_Index.sigma[graph_cell.batch]

            return (
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y,
                loss_press,
                uvp_node_new_with_dim,
                uvp_cell_new_with_dim,
            )
        else:
            
            graph_node = self.update_x_attr(
                graph=graph_node,
            )

            graph_node = self.update_edge_attr(
                graph=graph_node,
            )

            uvp_new_node = self.simulator(graph_node,graph_cell)
            
            uvp_new_node = self._enforce_boundary_condition(uvp_new_node, graph_node)*\
                graph_Index.uvp_dim[graph_node.batch]*\
                graph_Index.sigma[graph_node.batch]
            
            return uvp_new_node

    def load_checkpoint(self, optimizer=None, scheduler=None, ckpdir=None, device=None):
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir, map_location=device)
        self.load_state_dict(dicts["model"])
        keys = list(dicts.keys())
        keys.remove("model")

        if optimizer is not None and "optimizer0" in dicts:
            if not isinstance(optimizer, list):
                optimizer = [optimizer]
            for i, o in enumerate(optimizer):
                if f"optimizer{i}" in dicts:
                    o.load_state_dict(dicts[f"optimizer{i}"])
                    keys.remove(f"optimizer{i}")

        if scheduler is not None and "scheduler0" in dicts:
            if not isinstance(scheduler, list):
                scheduler = [scheduler]
            for i, s in enumerate(scheduler):
                if f"scheduler{i}" in dicts:
                    s.load_state_dict(dicts[f"scheduler{i}"])
                    keys.remove(f"scheduler{i}")

        if not self.training:
            keys = [
                key
                for key in keys
                if not (key.startswith("optimizer") or key.startswith("scheduler"))
            ]

        print(f"Simulator model and optimizer/scheduler loaded checkpoint {ckpdir}")

    def save_checkpoint(self, path=None, optimizer=None, scheduler=None):

        if path is None:
            path = self.model_dir

        to_save = {"model": self.state_dict()}

        if optimizer is not None:
            if not isinstance(optimizer, list):
                optimizer = [optimizer]
            for i, o in enumerate(optimizer):
                to_save[f"optimizer{i}"] = o.state_dict()

        if scheduler is not None:
            if not isinstance(scheduler, list):
                scheduler = [scheduler]
            for i, s in enumerate(scheduler):
                to_save[f"scheduler{i}"] = s.state_dict()

        torch.save(to_save, path)

        print(f"Simulator model saved at {path}")
