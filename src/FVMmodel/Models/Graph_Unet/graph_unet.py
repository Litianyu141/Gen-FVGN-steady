import torch
import torch.nn as nn
from pooling_block import TopKPooling
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphUNet
from torch_scatter import scatter_add
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.repeat import repeat
from typing import Callable, List, Union

class NodeToNode(nn.Module):
    def __init__(self, input_size, hidden_size, outputsize):
        super(NodeToNode, self).__init__()
        
        self.messagepassing_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, outputsize),
            nn.LayerNorm(normalized_shape=outputsize),
        )

    def forward(self, x, edge_index):
        
        senders, receivers = edge_index
        two_way_senders = torch.cat([senders, receivers], dim=0)
        two_way_receivers = torch.cat([receivers, senders], dim=0)
            
        agg_x = scatter_add(x[two_way_senders], two_way_receivers, dim=0)
        
        update_x = self.messagepassing_layer(agg_x)
        
        return update_x
    
class MeshGraphUnet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, outputsize, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()

        # Creating down layers
        for _ in range(num_layers // 2):
            self.pools.append(TopKPooling(hidden_size, ratio=0.5))
            self.down_convs.append(NodeToNode(hidden_size, hidden_size, hidden_size))

        # Creating up layers
        for _ in range(num_layers // 2):
            self.up_convs.append(NodeToNode(hidden_size * 2, hidden_size, hidden_size))  # *2 for concatenation
            
        # self.output_conv = NodeToNode(hidden_size * 2, hidden_size, hidden_size)
        
    def forward(self, graph_node,graph_cell=None, batch=None):
        x, edge_index,edge_attr = graph_node.x, graph_node.edge_index,  graph_node.edge_attr
        skip_connections = []
        perm_list = []
        contracted_edge_index = []
        
        for pool, down_conv in zip(self.pools, self.down_convs):
            skip_connections.append(x.clone())
            contracted_edge_index.append(edge_index)
            x, edge_index, edge_attr, batch, perm, _ = pool(x, 
                                                            edge_index, 
                                                            edge_attr=edge_attr, 
                                                            neighbor_cell=graph_cell.edge_index,
                                                            cells_node=graph_node.face.mT,
                                                            cells_index=graph_cell.face.mT, 
                                                            batch=batch)
            perm_list.append(perm)
            x = down_conv(x, edge_index)

        # Unpooling and skip connections for upsampling
        for up_conv, perm in zip(reversed(self.up_convs), reversed(perm_list)):
            recovered_x = torch.zeros_like(skip_connections[-1])
            recovered_x[perm] = x
            x = torch.cat([recovered_x, skip_connections.pop()], dim=-1)
            edge_index = contracted_edge_index.pop()
            x = up_conv(x, edge_index)

        return x

class MeshGraphUnet2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = False,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(NodeToNode(in_channels, channels, channels))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(NodeToNode(channels, channels, channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(NodeToNode(in_channels, channels, channels))
        self.up_convs.append(NodeToNode(in_channels, out_channels, channels))

    def forward(self, graph_node,
                batch: OptTensor = None) :
        """"""
        x, edge_index = graph_node.x, graph_node.edge_index
        
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        xs = [x]
        edge_indices = [edge_index]
        perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, batch, perm, _ = self.pools[i - 1](
                x, edge_index, batch)

            x = self.down_convs[i](x, edge_index)
            
            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')

if __name__ == '__main__':
    
    from torch_geometric.datasets import Planetoid
    
    # Load the dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    # Initialize model and optimizer
    model = MeshGraphUnet(dataset, num_layers=3, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = dataset[0].to(device)

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = (pred == data.y).sum()
    accuracy = int(correct) / int(data.y.size(0))
    print(f'Accuracy: {accuracy:.4f}')
