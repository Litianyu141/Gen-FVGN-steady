import torch
from utils.utilities import NodeType


def get_v_noise_on_cell(graph, noise_std, outchannel, device):
    if noise_std == 0:
        return torch.zeros_like(graph.x[:, 1:3], device=device)
    else:
        velocity_sequence = graph.x[:, 1:3]
        # type = graph.x[:, 0]
        noise_v = torch.normal(
            std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], outchannel)
        ).to(device)
        # mask = type!=NodeType.NORMAL
        # noise[mask]=0
        return noise_v


def get_noise_on_edge(graph, noise_std, outchannel, device):
    velocity_sequence = graph.x.repeat(1, 2)
    type = graph.x[:, 0]
    noise_v = torch.normal(
        std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], outchannel)
    ).to(device)
    noise = noise_v
    mask = type != NodeType.NORMAL
    noise[mask] = 0
    return noise.to(device)


def get_v_noise_on_node(graph, noise_std, outchannel, device):
    if noise_std == 0:
        return torch.zeros_like(graph.x[:, 1:3], device=device)
    else:
        velocity_sequence = graph.x[:, 1:3]
        _type = graph.x[:, 0]
        noise_v = torch.normal(
            std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], outchannel)
        ).to(device)
        noise = noise_v
        mask = torch.logical_not(
            (_type == NodeType.NORMAL) | (_type == NodeType.OUTFLOW)
        )
        noise[mask] = 0
        return noise.to(device)
