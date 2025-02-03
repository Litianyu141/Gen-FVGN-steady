import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from FVMmodel.Models.FVGN.EPD import EncoderProcesser
from FVMmodel.FVdiscretization.FVscheme import Intergrator
import torch.nn as nn
import torch
from torch_geometric.data import Data
import enum
from torch_scatter import scatter
from Utils.normalization import Normalizer
from Utils.utilities import NodeType

class Simulator(nn.Module):
    def __init__(
        self,
        message_passing_num,
        node_input_size,
        edge_input_size,
        cell_input_size,
        node_output_size,
        edge_output_size,
        cell_output_size,
        drop_out,
        attention,
        MultiHead,
        hidden_size,
        normlizer_steps,
        device,
        model_dir=None,
    ) -> None:
        super(Simulator, self).__init__()
        self._device = device
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.cell_input_size = cell_input_size
        self.model_dir = model_dir

        self.fvgn = EncoderProcesser(
            message_passing_num=message_passing_num,
            cell_input_size=cell_input_size,
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            cell_output_size=cell_output_size,
            edge_output_size=edge_output_size,
            node_output_size=node_output_size,
            drop_out=drop_out,
            attention=attention,
            MultiHead=MultiHead,
            hidden_size=hidden_size,
        )
        
        self.node_normlizer = Normalizer(
            size=node_input_size - 3,
            max_accumulations=normlizer_steps,
            epsilon=1e-8,
            device=device,
        )
        self.edge_normlizer = Normalizer(
            size=edge_input_size - 3,
            max_accumulations=normlizer_steps,
            epsilon=1e-8,
            device=device,
        )
        
        self.integrator = Intergrator(
            edge_input_size, cell_input_size, cell_output_size
        )

        print("Simulator model initialized")

    def enforce_BC(self,node_decode_attr,graph_node,graph_Index,params):
        
        def supress_reverse_flow(node_decode_p=None, graph_node=None, p_ref=0.0):
            """TODO: still have some problem, need to be fixed"""
            oundary_outflow_mask = (graph_node.node_type == NodeType.OUTFLOW).squeeze(1)

            node_decode_p[oundary_outflow_mask] = torch.where(
                node_decode_p[oundary_outflow_mask] > p_ref,
                p_ref,
                node_decode_p[oundary_outflow_mask],
            )

            return node_decode_p
        
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
            node_decode_p = supress_reverse_flow(
                node_decode_p=node_decode_p, graph_node=graph_node
            )

        # for cavity_flow pressure constraint point condition
        pressure_constraint_point = (graph_node.node_type == NodeType.IN_WALL).squeeze(
            1
        )
        node_decode_p[pressure_constraint_point] = 0.0

        node_decoded_uvp = (
            torch.cat((node_decode_uv, node_decode_p), dim=1)
            * graph_Index.sigma[graph_node.batch]
        )   
        
        return node_decoded_uvp
    
    def normalize_graph_features(self, x, batch, dimless=False):
        if dimless:
            mean = scatter(x, batch, dim=0, reduce="mean")
            residual = x - mean[batch]
            var = scatter(residual**2, batch, dim=0, reduce="mean")
            std = torch.sqrt(var)
            x = residual / (std[batch] + 1e-8)

            return x
        else:
            return x
        
    def norm_pixel_features(self, graph_node, norm=True):
        
        pixel = graph_node.pixel  # Assuming graph_node is a tensor with shape [B, H, W, C]
        
        if norm:
            # Calculate mean and std for each sample in the batch
            mean = pixel.mean(dim=(1, 2, 3), keepdim=True)  # Mean for each sample
            std = pixel.std(dim=(1, 2, 3), keepdim=True)    # Std for each sample
            
            # Normalize each sample
            pixel = (pixel - mean) / (std + 1e-8)  # Add a small epsilon to avoid division by zero
        
        return pixel
    
    def update_cell_attr(
        self,
        frames,
        one_hot: int,
        types: torch.Tensor,
        is_training=True,
        cell_batch=None,
    ):
        cell_feature = []
        cell_feature.append(frames)  # velocity
        if is_training:
            cell_type_boundary = types.long().unsqueeze(1)
            cell_type_in = torch.logical_not(types).long().unsqueeze(1)
        else:
            cell_type_boundary = types.long()
            cell_type_in = torch.logical_not(types).long()
        cell_feature.append(cell_type_boundary)
        cell_feature.append(cell_type_in)
        cell_feats = torch.cat(cell_feature, dim=1)

        attr = cell_feats
        return attr

    def update_edge_attr(
        self,
        edge_attr,
        one_hot: int,
        types: torch.Tensor,
        dimless=False,
        edge_batch=None,
        params=None,
    ):
        if one_hot > 0:
            # face type one-hot
            edge_feature = []
            edge_feature.append(edge_attr[:, 3:])  # init edge_relative_cor_diff
            face_type = types.view(-1).to(torch.long)
            one_hot_feature = torch.nn.functional.one_hot(face_type, one_hot)
            edge_feature.append(one_hot_feature)
            edge_feature = torch.cat(edge_feature, dim=1)

            norm_edge_feature = self.edge_normlizer(
                edge_feature, accumulate=params.accumulated_flag
            )

            norm_edge_uvp_diff_ahead = self.normalize_graph_features(
                edge_attr[:, 0:3], edge_batch, dimless=dimless
            )

            norm_edge_feature = torch.cat(
                (norm_edge_uvp_diff_ahead, norm_edge_feature), dim=-1
            )

            return norm_edge_feature
        else:
            edge_feature = []
            edge_feature.append(edge_attr[:, 3:])  # init edge_relative_cor_diff

            edge_feature = torch.cat(edge_feature, dim=1)

            norm_edge_feature = self.edge_normlizer(
                edge_feature, accumulate=params.accumulated_flag
            )

            norm_edge_feature_ahead = self.normalize_graph_features(
                edge_attr[:, 0:3], edge_batch, dimless=dimless
            )

            norm_edge_feature = torch.cat(
                (norm_edge_feature_ahead, norm_edge_feature), dim=-1
            )

            return norm_edge_feature

    def update_node_attr(
        self,
        node_attr,
        one_hot: int,
        types: torch.Tensor,
        dimless=False,
        node_batch=None,
        graph_node=None,
        params=None,
    ):
        if one_hot > 0:
            # face type one-hot
            node_feature = []
            node_feature.append(node_attr[:, 3:])  # theta pde
            node_type = types.view(-1).to(torch.long)
            one_hot_feature = torch.nn.functional.one_hot(node_type, one_hot)
            node_feature.append(one_hot_feature)
            node_feature = torch.cat(node_feature, dim=-1)
            
            norm_node_feature = self.node_normlizer(
                node_feature, accumulate=params.accumulated_flag
            )

            norm_node_uvp = self.normalize_graph_features(
                node_attr[:, 0:3], node_batch, dimless=dimless
            )

            norm_node_feature = torch.cat(
                (norm_node_uvp, norm_node_feature), dim=-1
            )

            return norm_node_feature
        else:
            node_feature = []
            node_feature.append(node_attr[:, 3:])  # init edge velocity and edge_EU_RP
            node_feature = torch.cat(node_feature, dim=-1)
            norm_node_feature = self.node_normlizer(
                node_feature, accumulate=params.accumulated_flag
            )

            norm_node_uvp = self.normalize_graph_features(
                node_attr[:, 0:3], node_batch, dimless=dimless
            )

            norm_node_feature = torch.cat(
                (norm_node_uvp, norm_node_feature), dim=-1
            )

            return norm_node_feature

    def forward(
        self,
        graph_cell: Data = None,
        graph_node_x: Data = None,
        graph_edge: Data = None,
        graph_node: Data = None,
        graph_Index: Data = None,
        uv_node_old=None,
        params=None,
        inteplote=False,
        norm=True,
    ):

        """perform *************************FORWARD*********************** at cell attr and edge attributes"""
        uv_node_old = graph_node.x[:, 0:2].clone() / graph_Index.uvp_dim[graph_node.batch, 0:2]
        
        if norm:
            # forward model
            graph_node.x = self.update_node_attr(
                node_attr=graph_node.x,
                one_hot=params.node_one_hot,
                types=graph_node.node_type.view(-1),
                dimless=params.dimless,
                node_batch=graph_node.batch,
                graph_node=graph_node,
                params=params,
            )

            graph_node.edge_attr = self.update_edge_attr(
                edge_attr=graph_node.edge_attr,
                one_hot=params.edge_one_hot,
                types=graph_edge.x[:, 1],
                dimless=params.dimless,
                edge_batch=graph_edge.batch,
                params=params,
            )  # repeat for two way edge_attr

        node_decode_attr = self.fvgn(
            graph_node, graph_edge, graph_cell, graph_Index, params=params
        )

        predicted_node_uvp = self.enforce_BC(node_decode_attr=node_decode_attr,
                                             graph_node=graph_node
                                             ,graph_Index=graph_Index,
                                             params=params)
        
        # explicit / implicit / IMEX integration schemes
        if params.integrator == "explicit":
            fluid_uv_node_hat = uv_node_old[:, 0:2]

        if params.integrator == "implicit":
            fluid_uv_node_hat = predicted_node_uvp[:, 0:2]

        if params.integrator == "imex":
            fluid_uv_node_hat = (
                uv_node_old[:, 0:2] + predicted_node_uvp[:, 0:2]
            ) / 2.0

        # Intergrate all flux at every edge`s of all cells
        (
            loss_cont,
            loss_momtentum_x,
            loss_momtentum_y,
            loss_press,
            uvp_new
        ) = self.integrator(
            predicted_node_uvp=predicted_node_uvp,
            uv_node_hat=fluid_uv_node_hat,
            uv_node_old=uv_node_old,
            graph_node=graph_node,
            graph_node_x=graph_node_x,
            graph_edge=graph_edge,
            graph_cell=graph_cell,
            graph_Index=graph_Index,
            params=params,
            inteplote=inteplote,
            device=self._device,
        )

        # reverse dimless for storing
        uvp_new = (
            uvp_new * graph_Index.uvp_dim[graph_node.batch]
        )
        
        return (
            uvp_new,
            loss_cont,
            loss_momtentum_x,
            loss_momtentum_y,
            loss_press
        )

    def load_checkpoint(
        self, optimizer=None, scheduler=None, ckpdir=None, device=None, is_training=True
    ):
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir, map_location=device)
        self.load_state_dict(dicts["model"])
        keys = list(dicts.keys())
        keys.remove("model")
        if optimizer is not None:
            if type(optimizer) is not list:
                optimizer = [optimizer]
            for i, o in enumerate(optimizer):
                o.load_state_dict(dicts["optimizer{}".format(i)])
                keys.remove("optimizer{}".format(i))

        if scheduler is not None:
            if type(scheduler) is not list:
                scheduler = [scheduler]
            for i, s in enumerate(scheduler):
                s.load_state_dict(dicts["scheduler{}".format(i)])
                keys.remove("scheduler{}".format(i))

        if not is_training:
            for key in keys.copy():
                if key.find("optimizer") >= 0:
                    keys.remove(key)
                elif key.find("scheduler") >= 0:
                    keys.remove(key)

        print("Simulator model and optimizer/scheduler loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, path=None, optimizer=None, scheduler=None):
        if path is None:
            path = self.model_dir

        model = self.state_dict()

        to_save = {"model": model}

        if type(optimizer) is not list:
            optimizer = [optimizer]
        for i, o in enumerate(optimizer):
            to_save.update({"optimizer{}".format(i): o.state_dict()})

        if type(scheduler) is not list:
            scheduler = [scheduler]
        for i, s in enumerate(scheduler):
            # to_save.update({"scheduler{}".format(i): s.get_variable()})
            to_save.update({"scheduler{}".format(i): s.state_dict()})
            
        torch.save(to_save, path)
        print("Simulator model saved at %s" % path)
