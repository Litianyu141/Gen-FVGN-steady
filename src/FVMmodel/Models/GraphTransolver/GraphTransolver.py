import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add
from einops import rearrange

import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class Graph_Physics_Attention_1D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.graph_temperature = nn.Parameter(torch.ones([1, heads, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def graph_forward(self, x, batch):
        # x: [B*N, C], batch: [B*N, 1]
        total_nodes = x.size(0)
        batch_size = torch.max(batch) + 1
        
        ### (1) Slice
        fx_mid = self.in_project_fx(x).view(
            total_nodes, self.heads, self.dim_head
        )  # [B*N, H, D]
        x_mid = self.in_project_x(x).view(
            total_nodes, self.heads, self.dim_head
        )  # [B*N, H, D]

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.graph_temperature
        )  # [B*N, H, G]
        slice_norm = scatter_add(
            slice_weights, batch, dim=0, dim_size=batch_size
        )  # [B, H, G]

        slice_token = scatter_add(
            fx_mid.unsqueeze(-2) * slice_weights.unsqueeze(-1), 
            batch, 
            dim=0,
            dim_size=batch_size
        )  # [B, H, G, C]
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        # attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_slice_token_expanded = out_slice_token[batch.squeeze()]  # [B*N, H, G, D]
        slice_weights_expanded = slice_weights  # [B*N, H, G]

        out_x = torch.sum(
            out_slice_token_expanded * slice_weights_expanded.unsqueeze(-1), dim=-2
        )  # [B*N, H, D]

        out_x = rearrange(out_x, "n h d -> n (h d)")

        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, hidden_size, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.hidden_size = hidden_size
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, hidden_size), act())
        self.linear_post = nn.Linear(hidden_size, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size, hidden_size), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        slice_num=32,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Graph_Physics_Attention_1D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )

    def forward(self, fx, batch, in_layernorm=False):
        if in_layernorm: # in_layernorm这个参数用于指定是来自TransFVGN还是原始Transolver的,TransFVGN的MLP在最后一层已经加入了LayerNorm
            fx = self.Attn.graph_forward(self.ln_1(fx), batch) + fx
        else:
            fx = self.Attn.graph_forward(fx, batch) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx
    


class Transolver(nn.Module):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        hidden_size=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
    ):
        super(Transolver, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(
                fun_dim + self.ref * self.ref * self.ref,
                hidden_size * 2,
                hidden_size,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = MLP(
                fun_dim + space_dim,
                hidden_size * 2,
                hidden_size,
                n_layers=0,
                res=False,
                act=act,
            )

        self.hidden_size = hidden_size
        self.space_dim = space_dim

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=hidden_size,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=False,
                )
                for _ in range(n_layers)
            ]
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, out_dim)
        )

        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / (hidden_size)) * torch.rand(hidden_size, dtype=torch.float)
        )

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

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(
            [batchsize, 1, self.ref, self.ref, 1]
        )
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(
            [batchsize, self.ref, 1, self.ref, 1]
        )
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(
            [batchsize, self.ref, self.ref, 1, 1]
        )
        grid_ref = (
            torch.cat((gridx, gridy, gridz), dim=-1)
            .cuda()
            .reshape(batchsize, self.ref**3, 3)
        )  # B 4 4 4 3

        pos = (
            torch.sqrt(
                torch.sum(
                    (my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1
                )
            )
            .reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, x, batch, params=None):


        # cfd_data, geom_data = data
        # x, fx, T = cfd_data.x, None, None
        # x  : node features
        # pos: node pos

        if self.unified_pos:
            new_pos = self.get_grid(pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        pos, fx = None, None

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx, batch, in_layernorm=True)
            
        fx = self.decoder(fx)

        return fx

if __name__ == '__main__':
    B, N = 3, 4
    dim = 256
    x = torch.randn(B, N, dim).cuda() 
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]).cuda()
    x_graph = x.reshape(-1, dim).cuda() 
    
    transo_deco = Transolver(5, 256,dropout=0.0,n_head=8,act='gelu',out_dim=1, slice_num=2).cuda()
    
    cd = transo_deco(x.reshape(3*4, -1), batch)
    print(cd.shape)