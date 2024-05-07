import sys
import os

cwd = os.getcwd()
sys.path.append(cwd + "/repos-py/FVM/my_FVNN/FVMmodel")
sys.path.append(cwd + "my_FVNN-pde-predict-on-cell/FVMmodel/derivatives.py")
from .model import EncoderProcesserDecoder, Intergrator
import torch.nn as nn
import torch
from torch_geometric.data import Data
import enum
from torch_scatter import scatter, scatter_mean
from torch_geometric.nn import global_mean_pool
from utils.normalization import Normalizer


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


def manual_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
        else:
            m.train()


class GenFVGN(nn.Module):
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
        super(GenFVGN, self).__init__()
        self._device = device
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.cell_input_size = cell_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(
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
        ).to(device)
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

        # self.model = GraphUNetModel(in_channels = node_input_size, hidden_channels=hidden_size, out_channels=node_output_size, depth=message_passing_num, pool_ratios=0.2).to(device)

        self.integrator = Intergrator(
            edge_input_size, cell_input_size, cell_output_size
        )

        print("GenFVGN model initialized")

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

    """we do normlize EU and relative_mesh_pos,also normlize cell_differce_on_edge"""

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
            edge_feature.append(edge_attr[:, 3:])  # init edge velocity and edge_EU_RP
            face_type = types.clone().view(-1).to(torch.long)
            one_hot_feature = torch.nn.functional.one_hot(face_type, one_hot)
            edge_feature.append(one_hot_feature)
            edge_feature = torch.cat(edge_feature, dim=1)

            norm_edge_feature = self.edge_normlizer(
                edge_feature, accumulate=params.accumulated_flag
            )

            norm_edge_feature_ahead = self.normalize_graph_features(
                edge_attr[:, 0:3], edge_batch, dimless=dimless
            )

            # norm_edge_feature_ahead = edge_attr[:,0:3]

            norm_edge_feature = torch.cat(
                (norm_edge_feature_ahead, norm_edge_feature), dim=-1
            )

            return norm_edge_feature
        else:
            edge_feature = []
            edge_feature.append(edge_attr[:, 3:])  # init edge velocity and edge_EU_RP
            # norm_edge_feature = self.normalize_graph_features(torch.cat(edge_feature, dim=1), edge_batch, dimless=dimless)

            # norm_edge_feature = torch.cat((edge_attr[:,0:3],norm_edge_feature),dim=-1)

            edge_feature = torch.cat(edge_feature, dim=1)

            norm_edge_feature = self.edge_normlizer(
                edge_feature, accumulate=params.accumulated_flag
            )

            norm_edge_feature_ahead = self.normalize_graph_features(
                edge_attr[:, 0:3], edge_batch, dimless=dimless
            )

            # norm_edge_feature_ahead = edge_attr[:,0:3]

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
            node_feature.append(node_attr[:, 3:])  # init edge velocity and edge_EU_RP
            node_type = types.clone().view(-1).to(torch.long)
            one_hot_feature = torch.nn.functional.one_hot(node_type, one_hot)
            node_feature.append(one_hot_feature)
            node_feature = torch.cat(node_feature, dim=-1)
            norm_node_feature = self.node_normlizer(
                node_feature, accumulate=params.accumulated_flag
            )

            norm_node_feature_ahead = self.normalize_graph_features(
                node_attr[:, 0:3], node_batch, dimless=dimless
            )

            norm_node_feature = torch.cat(
                (norm_node_feature_ahead, norm_node_feature), dim=-1
            )

            return norm_node_feature
        else:
            node_feature = []
            node_feature.append(node_attr[:, 3:])  # init edge velocity and edge_EU_RP
            node_feature = torch.cat(node_feature, dim=-1)
            norm_node_feature = self.node_normlizer(
                node_feature, accumulate=params.accumulated_flag
            )

            norm_node_feature_ahead = self.normalize_graph_features(
                node_attr[:, 0:3], node_batch, dimless=dimless
            )

            norm_node_feature = torch.cat(
                (norm_node_feature_ahead, norm_node_feature), dim=-1
            )

            return norm_node_feature

    def interpolate_node_to_face_and_cell_avg(
        self, node_phi=None, cells_node=None, face_node=None, cell_factor=None
    ):
        return (node_phi[face_node[0]] + node_phi[face_node[1]]) / 2.0, (
            node_phi[cells_node[0]] + node_phi[cells_node[1]] + node_phi[cells_node[2]]
        ) / 3.0

    def interploate_edge_to_node_and_cell(
        self, predicted_edge_uvp=None, graph_node=None, graph_edge=None, graph_cell=None
    ):
        # mask_face_interior = ((graph_edge.x[:,0]==NodeType.NORMAL)).view(-1,1)
        # mask_face_boundary = torch.logical_not(mask_face_interior)

        # predicted_edge_uvp_interior = predicted_edge_uvp[mask_face_interior.view(-1)].clone()
        # predicted_edge_uvp_boundary = predicted_edge_uvp[mask_face_boundary.view(-1)].detach()

        # predicted_edge_uvp_for_eq = torch.zeros_like(predicted_edge_uvp)

        # predicted_edge_uvp_for_eq[mask_face_boundary.view(-1)] = predicted_edge_uvp_boundary
        # predicted_edge_uvp_for_eq[mask_face_interior.view(-1)] = predicted_edge_uvp_interior

        predicted_edge_uvp_for_eq = predicted_edge_uvp
        # scatter edge attr to node
        senders_node_idx, receivers_node_idx = graph_node.edge_index
        twoway_node_connections = torch.cat(
            [senders_node_idx, receivers_node_idx], dim=0
        )
        twoway_edge_attr = torch.cat(
            [predicted_edge_uvp_for_eq, predicted_edge_uvp_for_eq], dim=0
        )
        predicted_node_uvp = scatter_mean(
            twoway_edge_attr,
            twoway_node_connections,
            dim=0,
            dim_size=graph_node.num_nodes,
        )
        predicted_cell_uvp = (
            predicted_edge_uvp_for_eq[graph_edge.face[0]]
            + predicted_edge_uvp_for_eq[graph_edge.face[1]]
            + predicted_edge_uvp_for_eq[graph_edge.face[2]]
        ) / 3.0

        return predicted_node_uvp, predicted_cell_uvp

    def enforce_boundary_condition(
        self,
        predicted_cell_attr=None,
        predicted_edge_attr=None,
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

        # constant padding at inflow boundary
        predicted_cell_attr[edge_neighbour_index_inflow[1]] = (
            predicted_cell_attr[edge_neighbour_index_inflow[1]].clone().detach()
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
        predicted_cell_attr[edge_neighbour_index_wall[1]] = (
            predicted_cell_attr[edge_neighbour_index_wall[1]].clone().detach()
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

        # interior equal ghost cell at outflow
        predicted_cell_attr[edge_neighbour_index_outflow[1]] = (
            predicted_cell_attr[edge_neighbour_index_outflow[1]].clone().detach()
        )

        mask_face_ghost_inverse = (
            (face_type == NodeType.GHOST_INFLOW)
            | (face_type == NodeType.GHOST_WALL)
            | (face_type == NodeType.GHOST_OUTFLOW)
        )
        predicted_edge_attr[mask_face_ghost_inverse] = (
            predicted_edge_attr[mask_face_ghost_inverse].clone().detach()
        )

        return (
            predicted_edge_attr[:, 0:2],
            predicted_edge_attr[:, 0:2],
            predicted_edge_attr[:, 2:3],
            predicted_cell_attr[:, 0:2],
            predicted_cell_attr[:, 2:3],
        )

    def forward(
        self,
        graph_cell: Data = None,
        graph_node_x: Data = None,
        graph_edge: Data = None,
        graph_node: Data = None,
        graph_cell_x: Data = None,
        params=None,
        inteplote=False,
    ):
        if self.training:
            """perform *************************FORWARD*********************** at cell attr and edge attributes"""
            # forward model
            graph_node.x = self.update_node_attr(
                node_attr=graph_node.x,
                one_hot=params.node_one_hot,
                types=graph_node.node_type.view(-1),
                dimless=params.dimless,
                node_batch=graph_node.batch.cuda(),
                graph_node=graph_node,
                params=params,
            )

            graph_node.edge_attr = self.update_edge_attr(
                edge_attr=graph_node.edge_attr,
                one_hot=params.edge_one_hot,
                types=graph_edge.x[:, 1],
                dimless=params.dimless,
                edge_batch=graph_edge.batch.cuda(),
                params=params,
            )  # repeat for two way edge_attr

            predicted_node_uvp = self.model(
                graph_cell, graph_node, graph_edge, params=params
            )

            # Intergrate all flux at every edge`s of all cells
            (
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y,
                loss_pressure_outlet,
                smoothed_predicted_uvp_node,
                nabla_uv_node,
                projection_method,
            ) = self.integrator(
                predicted_node_uvp=predicted_node_uvp,
                graph_node=graph_node,
                graph_node_x=graph_node_x,
                graph_edge=graph_edge,
                graph_cell=graph_cell,
                graph_cell_x=graph_cell_x,
                params=params,
                device=self._device,
            )

            # reverse dimless for storing
            smoothed_predicted_uvp_node = (
                smoothed_predicted_uvp_node * graph_node.uvp_dim
            )

            return (
                smoothed_predicted_uvp_node,
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y,
                loss_pressure_outlet,
                projection_method,
            )

        else:
            # '''Rolling out results'''
            # forward model
            graph_node.x = self.update_node_attr(
                node_attr=graph_node.x,
                one_hot=params.node_one_hot,
                types=graph_node.node_type.view(-1),
                dimless=params.dimless,
                node_batch=graph_node.batch.cuda(),
                graph_node=graph_node,
                params=params,
            )

            graph_node.edge_attr = self.update_edge_attr(
                edge_attr=graph_node.edge_attr,
                one_hot=params.edge_one_hot,
                types=graph_edge.x[:, 1],
                dimless=params.dimless,
                edge_batch=graph_edge.batch.cuda(),
                params=params,
            )  # repeat for two way edge_attr

            predicted_node_uvp = self.model(
                graph_cell, graph_node, graph_edge, params=params
            )

            # Intergrate all flux at every edge`s of all cells
            (   
                smoothed_predicted_uvp_node,
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y
            ) = self.integrator(
                predicted_node_uvp=predicted_node_uvp,
                graph_node=graph_node,
                graph_node_x=graph_node_x,
                graph_edge=graph_edge,
                graph_cell=graph_cell,
                graph_cell_x=graph_cell_x,
                params=params,
                device=self._device,
            )

            smoothed_predicted_uvp_node = (
                smoothed_predicted_uvp_node * graph_node.uvp_dim
            )

            return (
                smoothed_predicted_uvp_node,
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y
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
                scheduler_dicts = dicts["scheduler{}".format(i)]
                for key, value in scheduler_dicts.items():
                    object = eval("s." + key)
                    if type(value) == torch.Tensor:
                        value = value.cpu().cuda(device)
                    setattr(s, key, value)
                keys.remove("scheduler{}".format(i))

        if not is_training:
            for key in keys.copy():
                if key.find("optimizer") >= 0:
                    keys.remove(key)
                elif key.find("scheduler") >= 0:
                    keys.remove(key)

        print("GenFVGN model and optimizer/scheduler loaded checkpoint %s" % ckpdir)

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
            to_save.update({"scheduler{}".format(i): s.get_variable()})

        torch.save(to_save, path)
        print("GenFVGN model saved at %s" % path)
