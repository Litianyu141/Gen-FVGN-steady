import torch.nn as nn
import torch

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add, scatter_mean, scatter_softmax, scatter_mul

from FVMmodel.Models.FVGN.blocks import EdgeBlock, NodeBlock

def build_mlp(
    in_size, hidden_size, out_size, drop_out=True, lay_norm=True, dropout_prob=0.2
):
    if drop_out:
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.GELU(),
            nn.Linear(hidden_size, out_size),
        )
    else:
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_size),
        )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


def build_mlp_from_num_layer(
    in_size,
    hidden_size,
    out_size,
    drop_out=False,
    lay_norm=True,
    dropout_prob=0.2,
    num_layer=2,
):
    layers = []
    layers.append(nn.Linear(in_size, hidden_size))
    if drop_out:
        layers.append(nn.Dropout(p=dropout_prob))
    layers.append(nn.GELU())

    # Add specified number of hidden layers
    for i in range(num_layer - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if drop_out:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.GELU())

    layers.append(nn.Linear(hidden_size, out_size))

    if lay_norm:
        layers.append(nn.LayerNorm(normalized_shape=out_size))

    return nn.Sequential(*layers)


class GraphSCA3D(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()

        self.channel_excitation = nn.Sequential(
            nn.Linear(channel, int(channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel),
        )
        self.spatial_se = GCNConv(
            in_channels=channel,
            out_channels=1,
        )

    def forward(self, x, batch, edge_index):
        BN, C = x.size()
        chn_se = scatter_mean(x, index=batch, dim=0).view(-1, C)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se))
        chn_se = x * chn_se[batch]
        spa_se = torch.sigmoid(self.spatial_se(x, edge_index))
        spa_se = x * spa_se
        net_out = spa_se + x + chn_se
        return net_out


class Encoder(nn.Module):
    def __init__(
        self,
        node_input_size=128,
        edge_input_size=128,
        hidden_size=128,
    ):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(
            edge_input_size, hidden_size, int(hidden_size), drop_out=False
        )
        self.nb_encoder = build_mlp(
            node_input_size, hidden_size, int(hidden_size), drop_out=False
        )
        # self.c2b_mixer = build_mlp(
        #     int(1.5*hidden_size), hidden_size, int(hidden_size), drop_out=False
        # )
        
    def tranform_to_node_attr(self,cell_attr, cells_index, cells_node):
        node_attr = scatter_mean(
            src=cell_attr[cells_index], index=cells_node, dim=0
        )
        return node_attr
    
    def forward(self, graph_node, graph_cell=None):

        node_ = self.nb_encoder(graph_node.x)
        edge_ = self.eb_encoder(graph_node.edge_attr)

        # # message passing from cell to node
        # node_attr = scatter_add(
        #     src=cell_[graph_cell.face], index=graph_node.face, dim=0
        # )
        
        # # 以下这个过程类似FVGN，将cell的face上特征看作node与node边上的特征
        # senders_node_idx, receivers_node_idx = graph_node.edge_index
        # twoway_node_connections_indegree = torch.cat(
        #     [senders_node_idx, receivers_node_idx], dim=0
        # )
        # # sum agg
        # twoway_edge_attr = torch.cat((torch.chunk(edge_, 2, dim=-1)), dim=0)
        # node_agg_received_edges = scatter_add(
        #     twoway_edge_attr,
        #     twoway_node_connections_indegree,
        #     dim=0,
        #     out=torch.zeros(
        #         (node_attr.size(0), twoway_edge_attr.size(1)), device=node_attr.device
        #     ),
        # )
        # node_= self.c2b_mixer(torch.cat((node_attr,node_agg_received_edges),dim=-1)) + node_attr
        
        return (
            Data(
                x=node_,
                edge_attr=edge_,
                edge_index=graph_node.edge_index,
                face=graph_node.face,
                num_graphs=graph_node.num_graphs,
                batch=graph_node.batch,
            ),
            node_,
        )


class GnBlock(nn.Module):
    def __init__(self, hidden_size=128, drop_out=False):
        super(GnBlock, self).__init__()

        eb_input_dim = int(3 * (hidden_size))
        nb_input_dim = int(hidden_size + (hidden_size // 2.0))

        self.nb_module = NodeBlock(
            hidden_size,
            custom_func=build_mlp(
                nb_input_dim, hidden_size, int(hidden_size), drop_out=drop_out
            ),
        )

        self.eb_module = EdgeBlock(
            input_size=hidden_size,
            custom_func=build_mlp(
                eb_input_dim, hidden_size, int(hidden_size), drop_out=drop_out
            ),
        )

    def forward(self, graph_node):

        # conv
        graph_node_node_update = self.eb_module(graph_node)

        graph_node_edge_update = self.nb_module(graph_node_node_update)

        # resdiual connection
        x = graph_node.x + graph_node_edge_update.x
        edge_attr = graph_node.edge_attr + graph_node_edge_update.edge_attr

        return Data(
            x=x,
            edge_attr=edge_attr,
            edge_index=graph_node.edge_index,
            face=graph_node.face,
            num_graphs=graph_node.num_graphs,
            batch=graph_node.batch,
        )


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_sze=128,
        node_output_size=3,
    ):
        super(Decoder, self).__init__()

        self.node_decode_module = build_mlp_from_num_layer(
            hidden_sze,
            hidden_sze,
            node_output_size,
            drop_out=False,
            lay_norm=False,
            num_layer=2,
        )

    def forward(self, latent_graph_node=None):

        node_decode_attr = self.node_decode_module(latent_graph_node.x)

        return node_decode_attr


class EncoderProcesserDecoder(nn.Module):
    def __init__(
        self,
        message_passing_num,
        edge_input_size,
        node_input_size,
        node_output_size,
        drop_out=False,
        hidden_size=128,
        params=None,
    ):
        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            hidden_size=hidden_size,
        )

        GN_block_list = []
        for _ in range(message_passing_num):
            GN_block_list.append(
                GnBlock(
                    hidden_size=hidden_size,
                    drop_out=drop_out,
                )
            )
        self.GN_block_list = nn.ModuleList(GN_block_list)
        
        self.decoder = Decoder(
            hidden_sze=hidden_size,
            node_output_size=node_output_size,
        )

    def forward(
        self,
        graph_node=None,
        graph_cell=None,
    ):

        latent_graph_node,_ = self.encoder(graph_node)

        for idx,model in enumerate(self.GN_block_list):

            latent_graph_node = model(latent_graph_node)
        
        pred_node = self.decoder(latent_graph_node)

        return pred_node
