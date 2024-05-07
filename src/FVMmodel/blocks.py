from xmlrpc.client import Boolean, boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_softmax
from utils.utilities import (
    decompose_and_trans_node_attr_to_cell_attr_graph,
    calc_cell_centered_with_node_attr,
)
from torch_geometric.data import Data
import numpy as np


class NodeBlock(nn.Module):
    def __init__(
        self, input_size, attention_size, attention=True, MultiHead=1, custom_func=None
    ):
        super(NodeBlock, self).__init__()
        if attention_size % MultiHead > 0:
            raise ValueError("MultiHead must be the factor of attention_size")
        # self.Linear = nn.Sequential(nn.LazyLinear(1),nn.LeakyReLU(negative_slope=0.2))
        # self.Linear_projection = nn.ModuleList([self.Linear for i in range(MultiHead)])
        self.net = custom_func
        self.attention = attention
        self.scale = torch.sqrt(torch.tensor(input_size))

    def forward(self, graph_node, graph_cell):
        # Decompose graph
        (
            node_attr,
            edge_index,
            edge_attr,
            face,
            _,
            _,
        ) = decompose_and_trans_node_attr_to_cell_attr_graph(
            graph_node, has_changed_node_attr_to_cell_attr=True
        )

        # two step message passing algorithm

        senders_cell_idx, receivers_cell_idx = graph_cell.edge_index
        twoway_edge_attr = torch.cat((torch.chunk(edge_attr, 2, dim=-1)), dim=0)

        if self.attention:
            senders_node_idx, receivers_node_idx = graph_node.edge_index
            twoway_node_connections_outdegree = torch.cat(
                [receivers_node_idx, senders_node_idx], dim=0
            )
            attention_src = (
                torch.sum(
                    (twoway_edge_attr * node_attr[twoway_node_connections_outdegree]),
                    dim=1,
                    keepdim=True,
                )
                / self.scale
            )
            attention_factor = scatter_softmax(
                attention_src, twoway_node_connections_outdegree, dim=0
            )
            node_agg_received_edges = scatter_add(
                twoway_edge_attr * attention_factor,
                twoway_node_connections_outdegree,
                dim=0,
            )

        else:
            twoway_cell_connections_indegree = torch.cat(
                [senders_cell_idx, receivers_cell_idx], dim=0
            )

            twoway_cell_connections_outdegree = torch.cat(
                [receivers_cell_idx, senders_cell_idx], dim=0
            )

            cell_agg_received_edges = scatter_add(
                twoway_edge_attr, twoway_cell_connections_indegree, dim=0
            )

            cell_agg_neighbour_cell = scatter_add(
                cell_agg_received_edges[twoway_cell_connections_indegree],
                twoway_cell_connections_outdegree,
                dim=0,
            )

            cells_node = graph_node.face[0]
            cells_index = graph_cell.face[0]
            cell_to_node = cell_agg_neighbour_cell[cells_index]
            node_agg_received_edges = scatter_mean(cell_to_node, cells_node, dim=0)

        # update node attr
        x = self.net(torch.cat((node_agg_received_edges, node_attr), dim=1))

        return Data(x=x, edge_attr=edge_attr, edge_index=edge_index, face=face)


class EdgeBlock(nn.Module):
    def __init__(self, custom_func=None):
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph_node, graph_cell):
        (
            node_attr,
            edge_index,
            edge_attr,
            face,
            _,
            _,
        ) = decompose_and_trans_node_attr_to_cell_attr_graph(
            graph_node, has_changed_node_attr_to_cell_attr=True
        )

        edges_to_collect = []

        """ >>> node to cell and concancentendate to edge >>> """
        # aggrate node attr to cell
        cells_node = graph_node.face
        cells_index = graph_cell.face
        cell_attr = calc_cell_centered_with_node_attr(
            node_attr=node_attr,
            cells_node=cells_node,
            cells_index=cells_index,
            reduce="sum",
            map=True,
        )

        # concancentendate cell attr to edge
        senders_cell_idx, receivers_cell_idx = graph_cell.edge_index
        # filter self-loop face
        mask = torch.logical_not(senders_cell_idx == receivers_cell_idx).unsqueeze(1)

        senders_attr = cell_attr[senders_cell_idx]
        receivers_attr = cell_attr[receivers_cell_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr * (mask.long()))
        edges_to_collect.append(edge_attr)
        """ <<< node to cell and concancentendate to edge <<< """

        """ >>> only node concancentendate to edge >>> """
        # senders_node_idx,receivers_node_idx_idx= edge_index
        # senders_attr = node_attr[senders_node_idx]
        # receivers_attr = node_attr[receivers_node_idx_idx]

        # edges_to_collect.append(senders_attr)
        # edges_to_collect.append(receivers_attr)
        # edges_to_collect.append(edge_attr)
        """ >>>> only node concancentendate to edge >>> """

        collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr_ = self.net(collected_edges)  # Update

        return Data(x=node_attr, edge_attr=edge_attr_, edge_index=edge_index, face=face)


class CellBlock(nn.Module):
    def __init__(
        self, input_size, attention_size, attention=True, MultiHead=1, custom_func=None
    ):
        super(CellBlock, self).__init__()
        if attention_size % MultiHead > 0:
            raise ValueError("MultiHead must be the factor of attention_size")
        # self.Linear = nn.Sequential(nn.LazyLinear(1),nn.LeakyReLU(negative_slope=0.2))
        # self.Linear_projection = nn.ModuleList([self.Linear for i in range(MultiHead)])
        self.net = custom_func
        self.attention = attention
        self.scale = torch.sqrt(torch.tensor(input_size))

    def forward(self, graph, graph_node, node_embedding):
        # Decompose graph
        (
            cell_attr,
            _,
            edge_attr,
            face,
            _,
            _,
        ) = decompose_and_trans_node_attr_to_cell_attr_graph(
            graph, has_changed_node_attr_to_cell_attr=True
        )

        """
        receivers_idx = graph.cells_face[0]
        num_nodes = graph.num_nodes #num_nodes stands for the number of cells
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
        """
        # senders_idx,receivers_idx = graph.edge_index
        # twoway_cell_connections = torch.cat([senders_idx,receivers_idx],dim=0)

        # two step message passing algorithm
        senders_node_idx, receivers_node_idx = graph_node.edge_index
        twoway_node_connections_indegree = torch.cat(
            [senders_node_idx, receivers_node_idx], dim=0
        )
        twoway_node_connections_outdegree = torch.cat(
            [receivers_node_idx, senders_node_idx], dim=0
        )
        twoway_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        if self.attention:
            attention_src = (
                torch.sum(
                    (
                        twoway_edge_attr
                        * node_embedding[twoway_node_connections_outdegree]
                    ),
                    dim=1,
                    keepdim=True,
                )
                / self.scale
            )
            attention_factor = scatter_softmax(
                attention_src, twoway_node_connections_outdegree, dim=0
            )
            node_agg_received_edges = scatter_add(
                twoway_edge_attr * attention_factor,
                twoway_node_connections_outdegree,
                dim=0,
                dim_size=graph_node.num_nodes,
            )

        else:
            node_agg_received_edges = scatter_add(
                twoway_edge_attr,
                twoway_node_connections_indegree,
                dim=0,
                dim_size=graph_node.num_nodes,
            )
            node_attr = None

        cell_agg_received_nodes = (
            torch.index_select(node_agg_received_edges, 0, graph_node.face[0])
            + torch.index_select(node_agg_received_edges, 0, graph_node.face[1])
            + torch.index_select(node_agg_received_edges, 0, graph_node.face[2])
        ) / 3.0

        # update information
        cell_attr_new = self.net(torch.cat((cell_attr, cell_agg_received_nodes), dim=1))

        # make sure no updating on ghost cell
        # x *= mask_cell_interior.view(-1,1).long()
        cells_node_index = torch.cat(
            (graph_node.face[0], graph_node.face[1], graph_node.face[2]), dim=0
        )
        node_attr = scatter_mean(
            torch.cat((cell_attr_new, cell_attr_new, cell_attr_new), dim=0),
            cells_node_index,
            dim=0,
        )

        return (
            Data(
                x=cell_attr_new,
                edge_attr=edge_attr,
                edge_index=graph.edge_index,
            ),
            node_attr,
        )
