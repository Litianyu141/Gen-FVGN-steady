import torch
import numpy as np
import math
from torch_scatter import scatter

def velocity_profile(
    inlet_node_pos=None,
    mean_u=None,
    aoa=None,
    inlet_type="uniform",
):
    """
    >>>>计算管道流入口边界条件分布>>>>

    参数：
    inlet_node_pos: torch.Tensor [N,2], y轴上的高度值
    max_speed: float, 入口处的最大法向流速度
    spec_velocity: float, 特定的入口处平均法向流速度
    boundary: tuple, 流入口的上下限 (y_min, y_max)

    返回：
    inflow_distribution: numpy array,入口处的速度分布
    """
    if 0. == inlet_node_pos.numel():
        return torch.zeros_like(inlet_node_pos), torch.zeros_like(inlet_node_pos)[:,0:1]
    
    if inlet_node_pos is None:
        raise ValueError("There`s no inlet boundary pos")
    
    inlet_field = torch.zeros_like(inlet_node_pos)
    pressure_field = torch.zeros((inlet_node_pos.shape[0],1))
    
    if "parabolic" == inlet_type:
        y_positions = inlet_node_pos[:, 1] - torch.min(inlet_node_pos[:, 1])

        max_y = torch.max(y_positions)
        min_y = torch.min(y_positions)

        inlet_field[:, 0] = (
            6
            * mean_u
            * y_positions
            * (((max_y - min_y) - y_positions) / (max_y - min_y) ** 2)
        )
        
    elif "uniform" == inlet_type:
        inlet_field[:, 0] = torch.full_like(inlet_field[:, 0], float(mean_u))
        
    elif "uniform_aoa" == inlet_type:
        inlet_field[:, 0:2] = mean_u*torch.tensor(
            [math.cos(math.radians(aoa)), math.sin(math.radians(aoa))]
        )
        
    elif "Taylor_Green" == inlet_type:
        x = inlet_node_pos[:, 0]
        y = inlet_node_pos[:, 1]
        # 若需要在[0,1]区间实现波数为1的Taylor-Green，可乘以2π：
        inlet_field[:, 0] = mean_u*torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)
        inlet_field[:, 1] = -mean_u*torch.cos(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
        pressure_field = (-(1/4) * mean_u* (torch.cos(4 * torch.pi * x) + torch.cos(4 * torch.pi * y)))[:,None]

    elif None==inlet_type: # wave equation has no inlet velocity
        inlet_field[:, 0:2] = 0.
        Warning("No inlet velocity type is specified")
        
    return inlet_field.to(torch.float32), pressure_field.to(torch.float32)

def generate_pressure_source(
    mesh_pos, 
    batch:torch.Tensor, 
    source_frequency, 
    source_strength,
    dt:torch.Tensor,
    time_index:torch.Tensor
):
    """
    Generates a pressure source Sp at the center of the cavity based on the given parameters.

    Parameters:
    - mesh_pos: torch.Tensor of shape [1500, 2] containing mesh coordinates.
    - source_frequency: frequency of the source in Hz.
    - source_strength: amplitude of the source.
    - dt: time step for simulation.
    - time_index: current time step index. And it must be more than 1

    Returns:
    - Sp: torch.Tensor of shape [1500] representing the pressure source at each mesh point.
    """
    if (time_index < 1).any():
        raise ValueError(
            "When solving wave equation, time index should be more than 1"
        )

    # Calculate the center of the mesh_pos
    center = scatter(
        mesh_pos, batch, dim=0, reduce="mean"
    )
    
    # Calculate the source signal at the current time step
    current_time = (dt * time_index).view(-1,1)

    signal_magitude = torch.exp(
        -((mesh_pos - center[batch])[:, 0:1] ** 2 + (mesh_pos - center[batch])[:, 1:2] ** 2)
        * source_strength.view(-1,1)[batch]
        * 1000
    )

    source_signal = (
        torch.sin(source_frequency.view(-1,1) * torch.pi * current_time)[batch]
        * signal_magitude
    )

    return source_signal