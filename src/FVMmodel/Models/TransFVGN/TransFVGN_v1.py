import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import torch
import torch.nn as nn
from FVMmodel.Models.FVGN.EPD import Encoder, Decoder, GnBlock
from FVMmodel.Models.GraphTransolver.GraphTransolver import Transolver_block

class Simulator(nn.Module):
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
        super(Simulator, self).__init__()

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
        
        self.TransBlock = Transolver_block(
            num_heads=8,
            hidden_dim=hidden_size,
            dropout=0,
            act="gelu",
            mlp_ratio=2,
            slice_num=32
        )
        
        self.decoder = Decoder(
            hidden_sze=hidden_size,
            node_output_size=node_output_size,
        )
        
    # @torch.compile
    def forward(
        self,
        graph_node=None,
        graph_edge=None,
        graph_cell=None,
    ):

        latent_graph_cell, cell_embedding = self.encoder(graph_cell)

        for _, model in enumerate(self.GN_block_list):

            latent_graph_cell = model(latent_graph_cell)
            
        latent_graph_cell.x = self.TransBlock(
            latent_graph_cell.x+cell_embedding,
            graph_cell.batch,
        )
        
        pred_cell = self.decoder(latent_graph_cell)

        return pred_cell