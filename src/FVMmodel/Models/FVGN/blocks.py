import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data


class NodeBlock(nn.Module):
    def __init__(self, input_size, custom_func=None):
        super(NodeBlock, self).__init__()
        self.net = custom_func
        # self.proj = nn.Linear(input_size, 3*input_size)

    def forward(self, graph_node, graph_cell=None):

        # Decompose graph
        (
            node_attr,
            edge_index,
            edge_attr,
        ) = (graph_node.x, graph_node.edge_index, graph_node.edge_attr)

        """ node-based two step message passing algorithm"""

        senders_node_idx, receivers_node_idx = edge_index
        twoway_node_connections_indegree = torch.cat(
            [senders_node_idx, receivers_node_idx], dim=0
        )

        twoway_node_connections_outdegree = torch.cat(
            [receivers_node_idx, senders_node_idx], dim=0
        )

        # sum agg
        twoway_edge_attr = torch.cat((torch.chunk(edge_attr, 2, dim=-1)), dim=0)
        node_agg_received_edges = scatter_add(
            twoway_edge_attr,
            twoway_node_connections_indegree,
            dim=0,
            out=torch.zeros(
                (node_attr.size(0), twoway_edge_attr.size(1)), device=node_attr.device
            ),
        )

        node_avg_neighbour_node = scatter_mean(
            node_agg_received_edges[twoway_node_connections_outdegree],
            twoway_node_connections_indegree,
            dim=0,
            out=torch.zeros(
                (node_attr.size(0), twoway_edge_attr.size(1)), device=node_attr.device
            ),
        )

        # update node attr
        x = self.net(torch.cat((node_avg_neighbour_node, node_attr), dim=1))

        return Data(
            x=x,
            edge_attr=edge_attr,
            edge_index=edge_index,
            face=graph_node.face,
            num_graphs=graph_node.num_graphs,
            batch=graph_node.batch,
        )


class EdgeBlock(nn.Module):
    def __init__(self, input_size=None, custom_func=None):
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph_node, graph_cell=None):
        # Decompose graph
        (
            node_attr,
            edge_index,
            edge_attr,
        ) = (graph_node.x, graph_node.edge_index, graph_node.edge_attr)

        edges_to_collect = []

        """ >>> only node concancentendate to edge >>> """
        senders_node_idx, receivers_node_idx = edge_index

        twoway_node_connections_indegree = torch.cat(
            [senders_node_idx, receivers_node_idx], dim=0
        )

        twoway_node_connections_outdegree = torch.cat(
            [receivers_node_idx, senders_node_idx], dim=0
        )

        node_avg_neighbour_node = scatter_add(
            node_attr[twoway_node_connections_outdegree],
            twoway_node_connections_indegree,
            dim=0,
            out=torch.zeros(
                (node_attr.size(0), node_attr.size(1)), device=node_attr.device
            ),
        )

        senders_attr = node_avg_neighbour_node[senders_node_idx]
        receivers_attr = node_avg_neighbour_node[receivers_node_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)
        """ >>>> only node concancentendate to edge >>> """

        collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr_ = self.net(collected_edges)  # Update

        return Data(
            x=node_attr,
            edge_attr=edge_attr_,
            edge_index=edge_index,
            face=graph_node.face,
            num_graphs=graph_node.num_graphs,
            batch=graph_node.batch,
        )
