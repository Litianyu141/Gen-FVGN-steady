"""Utility functions for reading the datasets."""
import sys
import os
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import InMemoryDataset
from tfrecord.torch.dataset import TFRecordDataset
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import matplotlib

matplotlib.use("Agg")
# import networkx as nx
import matplotlib.pyplot as plt
from threading import Lock, Thread

cwd = os.getcwd()
sys.path.append(cwd + "/repos-py/FVM/my_FVNN")
sys.path.append(cwd + "/meshgraphnets/migration_utilities/")
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as torch_geometric_DataLoader
from torch.utils.data import RandomSampler, Sampler
from matplotlib import tri as mtri
import h5py
import enum
from Extract_mesh.write_tec import write_tecplotzone,write_tecplotzone_in_process
import math
from utils import utilities, get_param
from utils.utilities import (
    extract_cylinder_boundary_only_training,
    flow_type_mapping,
    calc_cell_centered_with_node_attr,
    calc_node_centered_with_cell_attr,
)
import datetime
from collections import deque
from torch_scatter import scatter, scatter_add
import multiprocessing

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


c_NORMAL_max = 0
c_OBSTACLE_max = 0
c_AIRFOIL_max = 0
c_HANDLE_max = 0
c_INFLOW_max = 0
c_OUTFLOW_max = 0
c_WALL_BOUNDARY_max = 0
c_SIZE_max = 0

c_NORMAL_min = 1000
c_OBSTACLE_min = 1000
c_AIRFOIL_min = 1000
c_HANDLE_min = 1000
c_INFLOW_min = 1000
c_OUTFLOW_min = 1000
c_WALL_BOUNDARY_min = 1000
c_SIZE_min = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_meta = False
shapes = {}
dtypes = {}
types = {}
steps = None
dataset_dir = ""
batch_size = 1000
add_target = False


def stastic(frame):
    flatten = frame[:, 0]
    global c_NORMAL_max
    global c_OBSTACLE_max
    global c_AIRFOIL_max
    global c_HANDLE_max
    global c_INFLOW_max
    global c_OUTFLOW_max
    global c_WALL_BOUNDARY_max
    global c_SIZE_max

    global c_NORMAL_min
    global c_OBSTACLE_min
    global c_AIRFOIL_min
    global c_HANDLE_min
    global c_INFLOW_min
    global c_OUTFLOW_min
    global c_WALL_BOUNDARY_min
    global c_SIZE_min

    c_NORMAL = 0
    c_OBSTACLE = 0
    c_AIRFOIL = 0
    c_HANDLE = 0
    c_INFLOW = 0
    c_OUTFLOW = 0
    c_WALL_BOUNDARY = 0
    c_SIZE = 0

    for i in range(flatten.shape[0]):
        if flatten[i] == NodeType.NORMAL:
            c_NORMAL += 1
            c_NORMAL_max = max(c_NORMAL_max, c_NORMAL)
            c_NORMAL_min = min(c_NORMAL_min, c_NORMAL)
        elif flatten[i] == NodeType.OBSTACLE:
            c_OBSTACLE += 1
            c_OBSTACLE_max = max(c_OBSTACLE_max, c_OBSTACLE)
            c_OBSTACLE_min = min(c_OBSTACLE_min, c_OBSTACLE)
        elif flatten[i] == NodeType.AIRFOIL:
            c_AIRFOIL += 1
            c_AIRFOIL_max = max(c_AIRFOIL_max, c_AIRFOIL)
            c_OBSTACLE_min = min(c_AIRFOIL_min, c_AIRFOIL)
        elif flatten[i] == NodeType.HANDLE:
            c_HANDLE += 1
            c_HANDLE_max = max(c_HANDLE_max, c_HANDLE)
            c_HANDLE_min = min(c_HANDLE_min, c_HANDLE)
        elif (flatten[i] == NodeType.INFLOW) | (flatten[i] == NodeType.IN_WALL):
            c_INFLOW += 1
            c_INFLOW_max = max(c_INFLOW_max, c_INFLOW)
            c_INFLOW_min = min(c_INFLOW_min, c_INFLOW)
        elif flatten[i] == NodeType.OUTFLOW:
            c_OUTFLOW += 1
            c_OUTFLOW_max = max(c_OUTFLOW_max, c_OUTFLOW)
            c_OUTFLOW_min = min(c_OUTFLOW_min, c_OUTFLOW)
        elif flatten[i] == NodeType.WALL_BOUNDARY:
            c_WALL_BOUNDARY += 1
            c_WALL_BOUNDARY_max = max(c_WALL_BOUNDARY_max, c_WALL_BOUNDARY)
            c_WALL_BOUNDARY_min = min(c_WALL_BOUNDARY_min, c_WALL_BOUNDARY)
        elif flatten[i] == NodeType.SIZE:
            c_SIZE += 1
            c_SIZE_max = max(c_SIZE_max, c_SIZE)
            c_SIZE_min = min(c_SIZE_min, c_SIZE)


class TimeStepSequence:
    def __init__(self, init_data):
        assert len(init_data) == 2, "Initial data should contain exactly 2 elements"
        self.data = deque(init_data, maxlen=2)

    def get_timestep_at_t(self):
        return self.data[1]  # 现在，这个方法返回的是最新的时间步数据

    def get_timestep_at_t_1(self):
        return self.data[0]  # 现在，这个方法返回的是上一时间步的数据

    def get_timestep_at(self, index):
        assert index in [0, 1], "Index should be 0 or 1"
        return self.data[index]

    def update(self, new_data):
        self.data.append(new_data)

    def __getitem__(self, key):
        if key == "velocity_on_node":
            return self.get_timestep_at_t()
        else:
            raise KeyError(f"Unsupported key: {key}")


g__count = 0


class CFDdatasetBase:
    # Base class for CFDdataset with process_trajectory method
    @staticmethod
    def calc_mean_u(fluid_property_comb=None):
        mean_U, rho, mu, source, aoa, dt = random.choice(fluid_property_comb)

        return mean_U, rho, mu, source, aoa, dt

    @staticmethod
    def cal_charactisc_length(trajectory, params):
        """prepare data for cal_relonyds_number"""
        node_type = trajectory["node_type"][0].view(-1)
        ghosted_mesh_pos = trajectory["mesh_pos"][0]
        mesh_pos = trajectory["mesh_pos"][0][
            (node_type != NodeType.GHOST_INFLOW)
            & (node_type != NodeType.GHOST_OUTFLOW)
            & (node_type != NodeType.GHOST_WALL)
        ]

        top = torch.max(mesh_pos[:, 1]).numpy()
        bottom = torch.min(mesh_pos[:, 1]).numpy()
        left = torch.min(mesh_pos[:, 0]).numpy()
        right = torch.max(mesh_pos[:, 0]).numpy()

        """cal cylinder diameter"""
        if trajectory["flow_type"] == "pipe_flow":
            boundary_pos = ghosted_mesh_pos[node_type == NodeType.WALL_BOUNDARY].numpy()
            cylinder_mask = (
                torch.full((boundary_pos.shape[0], 1), True).view(-1).numpy()
            )
            cylinder_not_mask = np.logical_not(cylinder_mask)
            cylinder_mask = np.where(
                (
                    (boundary_pos[:, 1] > bottom)
                    & (boundary_pos[:, 1] < top)
                    & (boundary_pos[:, 0] < right)
                    & (boundary_pos[:, 0] > left)
                ),
                cylinder_mask,
                cylinder_not_mask,
            )
            obstcale_pos = torch.from_numpy(boundary_pos[cylinder_mask])

            _, left = torch.min(obstcale_pos[:, 0], dim=0)
            _, right = torch.max(obstcale_pos[:, 0], dim=0)

            trajectory["cylinder_diameter"] = torch.norm(
                obstcale_pos[left] - obstcale_pos[right], dim=0, keepdim=True
            )

        elif "cavity" in trajectory["flow_type"]:
            boundary_pos = ghosted_mesh_pos[
                (node_type == NodeType.INFLOW) | (node_type == NodeType.IN_WALL)
            ]
            L0 = boundary_pos[:, 0].max() - boundary_pos[:, 0].min()
            trajectory["cylinder_diameter"] = torch.tensor([L0])
        else:
            raise ValueError("wrong flow type")

        return trajectory

    @staticmethod
    def create_mask(mesh_pos, aoa):
        # 计算圆心（所有点的均值）
        center = mesh_pos.mean(dim=0)

        # 将坐标转换到以圆心为原点的坐标系
        centered_pos = mesh_pos - center

        # 将角度转换为弧度
        aoa_rad = math.radians(aoa)

        # 创建一个表示夹角方向的单位向量
        normal_vector = torch.tensor([math.cos(aoa_rad), -math.sin(aoa_rad)])

        # 计算每个点与法向量的点积
        dot_products = torch.matmul(centered_pos, normal_vector)

        # 创建一个mask，表示每个点是否位于直线的左侧
        mask = dot_products > 0

        # 调整 mask 的形状为 [num_nodes, 1]
        return mask.view(-1)

    @staticmethod
    def velocity_profile(
        trajectory=None,
        init_field=None,
        node_pos=None,
        mean_velocity=None,
        dimless=False,
        flow_type="pipe_flow",
        inflow_bc_type="parabolic_velocity_field",
    ):
        """
        >>>>计算管道流入口边界条件分布>>>>

        参数：
        node_pos: torch.Tensor, y轴上的高度值
        max_speed: float, 入口处的最大法向流速度
        spec_velocity: float, 特定的入口处平均法向流速度
        boundary: tuple, 流入口的上下限 (y_min, y_max)

        返回：
        inflow_distribution: numpy array,入口处的速度分布
        """
        if flow_type == "pipe_flow":
            if inflow_bc_type == "parabolic_velocity_field":
                y_positions = node_pos[:, 1]

                max_velocity = mean_velocity

                max_y = torch.max(y_positions)
                min_y = torch.min(y_positions)

                inflow_node_flow_v_x_dir = (
                    4
                    * max_velocity
                    * y_positions
                    * (((max_y - min_y) - y_positions) / (max_y - min_y) ** 2)
                )

                init_field[:, 0] = inflow_node_flow_v_x_dir
            else:
                init_field[:, 0] = torch.full_like(
                    init_field[:, 0], float(mean_velocity)
                )

        elif flow_type == "cavity_flow":
            init_field[:, 0] = torch.full_like(init_field[:, 0], float(mean_velocity))

        elif flow_type == "cavity_possion":
            init_field[:, 0] = torch.full_like(init_field[:, 0], float(mean_velocity))

        elif flow_type == "circular-possion":
            init_field[:, 0] = torch.full_like(init_field[:, 0], float(mean_velocity))

        elif flow_type == "farfield-circular":
            init_field[:, 0] = torch.full_like(init_field[:, 0], mean_velocity)

        elif flow_type == "farfield-square":
            init_field[:, 0] = torch.full_like(init_field[:, 0], float(mean_velocity))

        elif flow_type == "farfield-half-circular-square":
            init_field[:, 0] = torch.full_like(init_field[:, 0], mean_velocity)

        if dimless:
            init_field = init_field / mean_velocity

        return init_field, trajectory

    @staticmethod
    def init_env(
        trajectory,
        mean_velocity=None,
        dimless=False,
        flow_type="pipe_flow",
        inflow_bc_type="parabolic_velocity_field",
    ):
        # init node uvp
        attr_on_node = torch.zeros(
            (trajectory["mesh_pos"].shape[0], trajectory["mesh_pos"].shape[1], 3)
        )

        node_type = trajectory["node_type"][0].to(torch.long).view(-1)
        node_pos = trajectory["mesh_pos"][0]

        # INLET
        wall_mask = node_type == utilities.NodeType.WALL_BOUNDARY

        init_field, trajectory = CFDdatasetBase.velocity_profile(
            trajectory=trajectory,
            init_field=attr_on_node[0],
            node_pos=node_pos,
            mean_velocity=mean_velocity,
            dimless=dimless,
            flow_type=flow_type,
            inflow_bc_type=inflow_bc_type,
        )
        init_field[wall_mask] = 0

        attr_on_node[0] = init_field

        # store target node
        trajectory["target|velocity_on_node"] = attr_on_node[0:1, :, 0:2].clone()
        trajectory["target|pressure_on_node"] = attr_on_node[0:1, :, 2:3].clone()

        # store init value at node,edge,cell
        if dimless:
            trajectory["velocity_on_node"] = (
                trajectory["target|velocity_on_node"].clone() * mean_velocity
            )
        else:
            trajectory["velocity_on_node"] = trajectory[
                "target|velocity_on_node"
            ].clone()

        trajectory["pressure_on_node"] = torch.zeros_like(
            trajectory["target|pressure_on_node"][:, :, 0:1]
        )

        # # store target edge for enforcing inlet boundary condition
        # face_node = trajectory['face'][0].long()
        # trajectory['target|velocity_on_edge'] = ((trajectory['target|velocity_on_node'][0][face_node[0]]+trajectory['target|velocity_on_node'][0][face_node[1]])/2.).unsqueeze(0)
        # trajectory['velocity_on_edge'] = ((trajectory['velocity_on_node'][0][face_node[0]]+trajectory['velocity_on_node'][0][face_node[1]])/2.).unsqueeze(0)

        return trajectory

    def cal_edge_2_cell_weight(self, trajectory_data):
        trajectory_data["face_center_pos"] = (
            torch.index_select(
                trajectory_data["mesh_pos"],
                1,
                trajectory_data["face"][0].to(torch.long)[0],
            )
            + torch.index_select(
                trajectory_data["mesh_pos"],
                1,
                trajectory_data["face"][0].to(torch.long)[1],
            )
        ) / 2.0
        centroid = trajectory_data["centroid"][0]
        mesh_pos = trajectory_data["mesh_pos"][0]
        cells_face = trajectory_data["cells_face"][0].long()
        face_node = trajectory_data["face"][0].long()
        cells_area = trajectory_data["cells_area"][0]
        area_list = []
        for i in range(cells_face.shape[1]):
            areai = self.calc_triangle_area(
                mesh_pos1=centroid,
                mesh_pos2=mesh_pos[face_node[0, cells_face[:, i]]],
                mesh_pos3=mesh_pos[face_node[1, cells_face[:, i]]],
            )
            area_list.append(areai)
        interior_cell_area = torch.stack(area_list, dim=1)
        edge_2_cell_weight = interior_cell_area / cells_area
        trajectory_data["edge_2_cell_weight"] = edge_2_cell_weight.unsqueeze(0)
        return trajectory_data

    @staticmethod
    def calc_triangle_area(mesh_pos1, mesh_pos2, mesh_pos3):
        a = torch.norm(mesh_pos1 - mesh_pos2, dim=1)
        b = torch.norm(mesh_pos2 - mesh_pos3, dim=1)
        c = torch.norm(mesh_pos3 - mesh_pos1, dim=1)
        s = (a + b + c) / 2.0
        area = torch.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    @staticmethod
    def enforce_cell_boundary_condition(trajectory=None):
        target_on_cell = trajectory["target_on_cell"][0, :, 0:2]
        bc = trajectory["target_on_edge"][0]
        edge_neighbour_index = trajectory["neighbour_cell"][0].long()
        cells_type = trajectory["cells_type"][0].view(-1)
        face_type = trajectory["face_type"][0].view(-1)
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
        target_on_cell[edge_neighbour_index_inflow[1]] = bc[mask_face_inflow, 0:2]
        target_on_cell[edge_neighbour_index_inflow[0]] = bc[mask_face_inflow, 0:2]

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
        target_on_cell[edge_neighbour_index_wall[1]] = 0.0
        target_on_cell[edge_neighbour_index_wall[0]] = 0.0

        # OUTFLOW
        mask_face_outflow = face_type == NodeType.OUTFLOW

        edge_neighbour_index_l = edge_neighbour_index[0]
        edge_neighbour_index_r = edge_neighbour_index[1]

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

        # constant padding at outflow boundary
        target_on_cell[edge_neighbour_index_outflow[1]] = bc[mask_face_outflow, 0:2]
        target_on_cell[edge_neighbour_index_outflow[0]] = bc[mask_face_outflow, 0:2]

        trajectory["target_on_cell"][0, :, 0:2] = target_on_cell.unsqueeze(0)

        return trajectory

    @staticmethod
    def makedimless(
        trajectory, params, fluid_property_comb=None, spec_u_rho_mu_comb=None
    ):
        if spec_u_rho_mu_comb is None:
            mean_velocity, rho, mu, source, aoa, dt = CFDdatasetBase.calc_mean_u(
                fluid_property_comb
            )
        else:
            mean_velocity, rho, mu, source, aoa, dt = spec_u_rho_mu_comb

        trajectory["mean_u"] = torch.tensor(mean_velocity, dtype=torch.float32)
        trajectory["rho"] = torch.tensor(rho, dtype=torch.float32)
        trajectory["mu"] = torch.tensor(mu, dtype=torch.float32)
        trajectory["source"] = torch.tensor(source, dtype=torch.float32)
        trajectory["aoa"] = torch.tensor(aoa, dtype=torch.float32)

        cells_area = trajectory["cells_area"][0]
        mesh_pos = trajectory["mesh_pos"][0]

        if params.dimless:
            U0 = mean_velocity

            if "possion" in trajectory["flow_type"]:
                continutiy_eq_coefficent = 0.0

                convection_coefficent = 0.0

                grad_p_coefficent = 0.0

                diffusion_coefficent = mu  # In NS equation, it was negtive laplace U

                source_term = source / mean_velocity

                pde_theta_node = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                            mean_velocity,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(
                    mesh_pos.shape[0], 1
                )  # node is for encoding

                pde_theta_cell = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(cells_area.shape[0], 1)

                neural_network_output_mask = (
                    torch.tensor([1, 0, 0], device=mesh_pos.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(mesh_pos.shape[0], 1)
                )

                trajectory["uvp_dim"] = torch.tensor(
                    [[[mean_velocity, 1.0, 1.0]]],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(1, mesh_pos.shape[0], 1)

                trajectory["uvp_dim_cell"] = torch.tensor(
                    [[[mean_velocity, 1.0, 1.0]]],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(1, cells_area.shape[0], 1)

                trajectory["dt"] = torch.full_like(
                    trajectory["mesh_pos"][:, :, 0:1],
                    dt * mean_velocity,
                    dtype=torch.float32,
                )

            else:
                continutiy_eq_coefficent = 1.0

                convection_coefficent = 1.0

                try:
                    grad_p_coefficent = 1.0 / rho
                except ZeroDivisionError:
                    grad_p_coefficent = 0.0
                try:
                    diffusion_coefficent = mu / (rho * mean_velocity)
                except ZeroDivisionError:
                    diffusion_coefficent=mu
                    
                source_term = source

                pde_theta_node = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                            mean_velocity,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(mesh_pos.shape[0], 1)

                pde_theta_cell = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(cells_area.shape[0], 1)

                neural_network_output_mask = (
                    torch.tensor([1, 1, 1], device=mesh_pos.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(mesh_pos.shape[0], 1)
                )

                trajectory["uvp_dim"] = torch.tensor(
                    [[[U0, U0, (U0**2)]]], device=mesh_pos.device, dtype=torch.float32
                ).repeat(1, mesh_pos.shape[0], 1)

                trajectory["uvp_dim_cell"] = torch.tensor(
                    [[[U0, U0, (U0**2)]]], device=mesh_pos.device, dtype=torch.float32
                ).repeat(1, cells_area.shape[0], 1)

                trajectory["dt"] = torch.full_like(
                    trajectory["mesh_pos"][:, :, 0:1],
                    (dt * mean_velocity),
                    dtype=torch.float32,
                )

        else:
            U0 = mean_velocity

            if "possion" in trajectory["flow_type"]:
                continutiy_eq_coefficent = 0.0

                convection_coefficent = 0.0

                grad_p_coefficent = 0.0

                diffusion_coefficent = mu

                source_term = source

                pde_theta_node = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                            mean_velocity,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(
                    mesh_pos.shape[0], 1
                )  # node is for encoding
                pde_theta_cell = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(cells_area.shape[0], 1)

                neural_network_output_mask = (
                    torch.tensor([1, 0, 0], device=mesh_pos.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(mesh_pos.shape[0], 1)
                )

                trajectory["uvp_dim"] = (
                    torch.tensor([1, 0, 0], device=mesh_pos.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(mesh_pos.shape[0], 1)
                )

                trajectory["uvp_dim_cell"] = torch.tensor(
                    [[[1, 1, 1]]], device=mesh_pos.device, dtype=torch.float32
                ).repeat(1, cells_area.shape[0], 1)

                trajectory["dt"] = torch.full_like(
                    trajectory["mesh_pos"][:, :, 0:1], dt, dtype=torch.float32
                )

            else:
                continutiy_eq_coefficent = 1.0

                convection_coefficent = 1.0

                grad_p_coefficent = 0.0

                grad_p_coefficent = 1.0 / rho

                diffusion_coefficent = mu / rho

                source_term = source

                pde_theta_node = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                            mean_velocity,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(mesh_pos.shape[0], 1)

                pde_theta_cell = torch.tensor(
                    [
                        [
                            continutiy_eq_coefficent,
                            convection_coefficent,
                            grad_p_coefficent,
                            diffusion_coefficent,
                            source_term,
                        ]
                    ],
                    device=mesh_pos.device,
                    dtype=torch.float32,
                ).repeat(cells_area.shape[0], 1)

                neural_network_output_mask = (
                    torch.tensor([1, 1, 1], device=mesh_pos.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(mesh_pos.shape[0], 1)
                )

                trajectory["uvp_dim"] = (
                    torch.tensor([1, 1, 1], device=mesh_pos.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(mesh_pos.shape[0], 1)
                )

                trajectory["uvp_dim_cell"] = torch.tensor(
                    [[[1, 1, 1]]], device=mesh_pos.device, dtype=torch.float32
                ).repeat(1, cells_area.shape[0], 1)

                trajectory["dt"] = torch.full_like(
                    trajectory["mesh_pos"][:, :, 0:1], dt, dtype=torch.float32
                )

        trajectory["pde_theta_node"] = pde_theta_node.unsqueeze(0)
        trajectory["pde_theta_cell"] = pde_theta_cell.unsqueeze(0)
        trajectory["neural_network_output_mask"] = neural_network_output_mask.unsqueeze(
            0
        )

        return trajectory, mean_velocity

    @staticmethod
    def calc_unv_dimless(trajectory, mesh_pos_dimless, dimless):
        face_node = trajectory["face"][0].long()
        edge_neighbour_cell = trajectory["neighbour_cell"][0].long()
        mesh_pos = trajectory["mesh_pos"][0]
        centroid = trajectory["centroid"][0]
        node_type = trajectory["node_type"][0].long()

        senders_node, recivers_node = face_node[0], face_node[1]
        senders_cell, recivers_cell = edge_neighbour_cell[0], edge_neighbour_cell[1]

        twoway_face_node_indegree = torch.cat((senders_node, recivers_node), dim=0)
        twoway_face_node_outdegree = torch.cat((recivers_node, senders_node), dim=0)

        twoway_neighbour_cell_indegree = torch.cat((senders_cell, recivers_cell), dim=0)
        twoway_neighbour_cell_outdegree = torch.cat(
            (recivers_cell, senders_cell), dim=0
        )

        mesh_pos_diff = (
            mesh_pos[twoway_face_node_indegree] - mesh_pos[twoway_face_node_outdegree]
        )
        centroid_diff = (
            centroid[twoway_neighbour_cell_indegree]
            - centroid[twoway_neighbour_cell_outdegree]
        )

        node_control_volume_normal_vector = torch.cat(
            (-centroid_diff[:, 1:2], centroid_diff[:, 0:1]), dim=1
        )

        mask = (
            torch.sum(
                mesh_pos_diff * node_control_volume_normal_vector, dim=1, keepdim=True
            )
            > 0
        ).repeat(1, 2)

        node_control_volume_normal_vector_dirction_biased = torch.where(
            mask, node_control_volume_normal_vector, -node_control_volume_normal_vector
        )
        node_control_volume_normal_vector_length = torch.norm(
            node_control_volume_normal_vector_dirction_biased, dim=1, keepdim=True
        )
        # because there`s self-loop at boundary cell, so the length of normal vector is zero, we set it to 1
        node_control_volume_normal_vector_length = torch.where(
            node_control_volume_normal_vector_length == 0,
            torch.tensor([1.0]),
            node_control_volume_normal_vector_length,
        )

        node_control_volume_unit_normal_vector = (
            node_control_volume_normal_vector_dirction_biased
            / node_control_volume_normal_vector_length
        )

        cells_node = trajectory["cells_node"][0].to(torch.long).T
        cell_3_node_pos = torch.stack(
            (mesh_pos[cells_node[0]], mesh_pos[cells_node[1]], mesh_pos[cells_node[2]]),
            dim=1,
        )
        dist_cell_3_node_pos_to_centroid = torch.norm(
            cell_3_node_pos - centroid.unsqueeze(1), dim=2, keepdim=True
        )
        total_dist_cell_3_node_pos_to_centroid = torch.sum(
            dist_cell_3_node_pos_to_centroid, dim=1
        ).unsqueeze(2)
        cell_factor = (
            dist_cell_3_node_pos_to_centroid / total_dist_cell_3_node_pos_to_centroid
        ).squeeze(2)
        trajectory["cell_factor"] = cell_factor.unsqueeze(0)
        # valid_mask = (node_type==NodeType.NORMAL).view(-1)
        # valid_unv_calc = (scatter_add(node_control_volume_unit_normal_vector*node_control_volume_normal_vector_length,twoway_face_node_outdegree,dim=0))[valid_mask]

        # valid_face_area_in, valid_face_area_out= torch.chunk(node_control_volume_normal_vector_length,2,dim=0)
        # node_control_volume_unit_normal_vector_in,node_control_volume_unit_normal_vector_out = torch.chunk(node_control_volume_unit_normal_vector,2,dim=0)
        # node_control_volume_unit_normal_vector_test = torch.cat((-node_control_volume_unit_normal_vector_in[:,1:2],node_control_volume_unit_normal_vector_in[:,0:1]),dim=1)

        # if (torch.abs(valid_unv_calc)<1e-8).all() and dimless:
        #     trajectory['unit_norm_v'] = node_control_volume_unit_normal_vector.unsqueeze(0)
        #     trajectory["face_length"] = node_control_volume_normal_vector_length.unsqueeze(0)
        # elif (torch.abs(valid_unv_calc)<1e-8).all():
        #     trajectory['unit_norm_v'] = node_control_volume_unit_normal_vector.unsqueeze(0)
        #     trajectory["face_length"] = node_control_volume_normal_vector_length.unsqueeze(0)

        # # plot unv
        # graph_cell = Data(edge_index = edge_neighbour_cell,pos = centroid)
        # G = to_networkx(graph_cell)
        # fig, (ax1) = plt.subplots(1, 1, figsize=(16, 9))
        # ax1.set_title('test_node_control_volume_field')
        # ax1.set_aspect('equal')
        # nx.draw(G, pos=centroid.numpy(), with_labels=False, node_color="skyblue", alpha=0.7, node_size=1,arrows=False)
        # unv_start = mesh_pos[twoway_face_node_outdegree]
        # ax1.quiver(unv_start[:,0].numpy(),unv_start[:,1].numpy(),node_control_volume_unit_normal_vector[:,0].numpy(),node_control_volume_unit_normal_vector[:,1].numpy(),units='height',color="red", angles='xy',scale_units='xy', scale=50,width=0.0025, headlength=3, headwidth=2, headaxislength=4.5)
        # plt.savefig("test_node_control_volume_field.png",dpi=400)

        # edge_vector = mesh_pos_dimless[face_node[0]]-mesh_pos_dimless[face_node[1]]
        # normal_vector = torch.cat((-edge_vector[:,1:2],edge_vector[:,0:1]),dim=1)
        # unit_normal_vector_uncorrect_edge = normal_vector/torch.norm(edge_vector,dim=1,keepdim=True)
        # unit_normal_vector_uncorrect_cell = torch.stack((unit_normal_vector_uncorrect_edge[cells_face[0]],unit_normal_vector_uncorrect_edge[cells_face[1]],unit_normal_vector_uncorrect_edge[cells_face[2]]),dim=1)

        # centroid = (mesh_pos_dimless[cells_node[0]]+mesh_pos_dimless[cells_node[1]]+mesh_pos_dimless[cells_node[2]])/3.
        # face_center_pos = (mesh_pos_dimless[face_node[0]]+mesh_pos_dimless[face_node[1]])/2.
        # face_center_pos_cell = torch.stack((face_center_pos[cells_face[0]],face_center_pos[cells_face[1]],face_center_pos[cells_face[2]]),dim=1)

        # centroid_to_face_vec = face_center_pos_cell-centroid.unsqueeze(1).repeat(1,3,1)
        # unit_normal_vector = torch.where((((torch.sum(unit_normal_vector_uncorrect_cell*centroid_to_face_vec,dim=2,keepdim=True)>0).repeat(1,1,2))),unit_normal_vector_uncorrect_cell,-unit_normal_vector_uncorrect_cell)

        # edge_length_per_cell = torch.stack((torch.norm(edge_vector,dim=1,keepdim=True)[cells_face[0]],torch.norm(edge_vector,dim=1,keepdim=True)[cells_face[1]],torch.norm(edge_vector,dim=1,keepdim=True)[cells_face[2]]),dim=1)

        # valid_unv_calc = torch.sum(edge_length_per_cell*unit_normal_vector,dim=1)
        # valid_unv_data = torch.sum(edge_length_per_cell*trajectory['unit_norm_v'][0],dim=1)

        # else:
        #     raise ValueError("wrong calculation of unit normal vector at cell face")
        # fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(16, 27))
        # # 设置三角剖分
        # # mesh_pos=to_numpy(current_dataset['mesh_pos'][0])
        # # cells_node=to_numpy(current_dataset['cells_node'][0])
        # triang = mtri.Triangulation(mesh_pos_dimless[:, 0].numpy(), mesh_pos_dimless[:, 1].numpy(),cells_node.T.numpy())
        # ax1.set_title('cell_center_field')
        # ax1.set_aspect('equal')
        # ax1.scatter(centroid[:,0].numpy(),centroid[:,1].numpy(),s=0.5)
        # ax1.triplot(triang, 'ko-', ms=0.5, lw=0.3, zorder=1)

        # ax2.set_title('face_center_field')
        # ax2.set_aspect('equal')
        # ax2.triplot(triang, 'ko-', ms=0.5, lw=0.3, zorder=1)
        # ax2.scatter(face_center_pos[:,0].numpy(),face_center_pos[:,1].numpy(),s=0.5)
        # # trajectory["unit_norm_v"] = unit_normal_vector.unsqueeze(0)

        # ax3.set_title('unit_norm_vector')
        # ax3.set_aspect('equal')
        # ax3.triplot(triang, 'ko-', ms=0.5, lw=0.3, zorder=1)

        # ax3.quiver(centroid[:,0].numpy(),centroid[:,1].numpy(),unit_normal_vector[:,0,0].numpy(),unit_normal_vector[:,0,1].numpy(),units='height',color="red", angles='xy',scale_units='xy', scale=50,width=0.0025, headlength=3, headwidth=2, headaxislength=4.5)

        # ax3.quiver(centroid[:,0].numpy(),centroid[:,1].numpy(),unit_normal_vector[:,1,0].numpy(),unit_normal_vector[:,1,1].numpy(),units='height',color="blue", angles='xy',scale_units='xy', scale=50,width=0.0025, headlength=3, headwidth=2, headaxislength=4.5)

        # ax3.quiver(centroid[:,0].numpy(),centroid[:,1].numpy(),unit_normal_vector[:,2,0].numpy(),unit_normal_vector[:,2,1].numpy(),units='height',color="green", angles='xy',scale_units='xy', scale=50,width=0.0025, headlength=3, headwidth=2, headaxislength=4.5)

        # plt.savefig("test_field.png",dpi=400)
        return trajectory

    @staticmethod
    def randbool(*size, device=None):
        """Returns 50% channce of True of False"""
        return torch.randint(2, size, device=device) == torch.randint(
            2, size, device=device
        )

    @staticmethod
    def redirect_edge(trajectory=None):
        face_node = trajectory["face"][0].to(torch.long)
        edge_neighbour_cell = trajectory["neighbour_cell"][0].to(torch.long)

        senders_node, receivers_node = face_node
        senders_cell, receivers_cell = edge_neighbour_cell
        random_mask = Data_Pool.randbool(
            1, senders_node.shape[0], device=senders_node.device
        ).repeat(2, 1)

        random_direction_face_node = torch.where(
            random_mask,
            torch.stack((senders_node, receivers_node), dim=0),
            torch.stack((receivers_node, senders_node), dim=0),
        )

        random_direction_neighbour_cell = torch.where(
            random_mask,
            torch.stack((senders_cell, receivers_cell), dim=0),
            torch.stack((receivers_cell, senders_cell), dim=0),
        )

        trajectory["face"] = random_direction_face_node.unsqueeze(0)
        trajectory["neighbour_cell"] = random_direction_neighbour_cell.unsqueeze(0)

        return trajectory

    @staticmethod
    def plot_all_boundary_sate(trajectory):
        mesh_pos = trajectory["mesh_pos"][0]
        edge_index = trajectory["face"][0].to(torch.long)
        face_node_x = trajectory["face_node_x"][0].to(torch.long)
        cells_index = trajectory["cells_index"][0].to(torch.long)

        node_type = trajectory["node_type"][0].long().view(-1)
        face_type = trajectory["face_type"][0].long().view(-1)
        face_center_pos = trajectory["face_center_pos"][0]
        unit_normal_vector = trajectory["unit_norm_v"][0]
        centroid = trajectory["centroid"][0]

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.cla()
        ax.set_aspect("equal")

        # 通过索引获取每一条边的两个点的坐标
        point1 = mesh_pos[face_node_x[0]]
        point2 = mesh_pos[face_node_x[1]]

        # 将每一对点的坐标合并，方便绘图
        lines = np.hstack([point1, point2]).reshape(-1, 2, 2)

        # 使用plot绘制所有的边
        plt.plot(lines[:, :, 0].T, lines[:, :, 1].T, "k-", lw=1, alpha=0.2)

        # 绘制点
        node_size = 5
        plt.scatter(
            mesh_pos[node_type == NodeType.NORMAL, 0],
            mesh_pos[node_type == NodeType.NORMAL, 1],
            c="red",
            linewidths=1,
            s=node_size,
        )
        plt.scatter(
            mesh_pos[node_type == NodeType.WALL_BOUNDARY, 0],
            mesh_pos[node_type == NodeType.WALL_BOUNDARY, 1],
            c="green",
            linewidths=1,
            s=node_size,
        )
        plt.scatter(
            mesh_pos[node_type == NodeType.OUTFLOW, 0],
            mesh_pos[node_type == NodeType.OUTFLOW, 1],
            c="orange",
            linewidths=1,
            s=node_size,
        )
        plt.scatter(
            mesh_pos[node_type == NodeType.INFLOW, 0],
            mesh_pos[node_type == NodeType.INFLOW, 1],
            c="blue",
            linewidths=1,
            s=node_size,
        )

        # plot face center type
        plt.scatter(
            face_center_pos[face_type == NodeType.NORMAL, 0],
            face_center_pos[face_type == NodeType.NORMAL, 1],
            c="red",
            linewidths=1,
            s=node_size,
        )
        plt.scatter(
            face_center_pos[face_type == NodeType.WALL_BOUNDARY, 0],
            face_center_pos[face_type == NodeType.WALL_BOUNDARY, 1],
            c="green",
            linewidths=1,
            s=node_size,
        )
        plt.scatter(
            face_center_pos[face_type == NodeType.OUTFLOW, 0],
            face_center_pos[face_type == NodeType.OUTFLOW, 1],
            c="orange",
            linewidths=1,
            s=node_size,
        )
        plt.scatter(
            face_center_pos[face_type == NodeType.INFLOW, 0],
            face_center_pos[face_type == NodeType.INFLOW, 1],
            c="blue",
            linewidths=1,
            s=node_size,
        )

        # plot unv
        if unit_normal_vector is not None:
            ax.quiver(
                centroid[cells_index, 0],
                centroid[cells_index, 1],
                unit_normal_vector[:, 0],
                unit_normal_vector[:, 1],
                units="height",
                color="red",
                angles="xy",
                scale_units="xy",
                scale=75,
                width=0.01,
                headlength=3,
                headwidth=2,
                headaxislength=3.5,
            )

        # if other_cell_centered_vector is not None:
        # ax.quiver(centroid[:,0],centroid[:,1],other_cell_centered_vector[:,0],other_cell_centered_vector[:,1],units='height',color="cyan", angles='xy',scale_units='xy', scale=75,width=0.01, headlength=3, headwidth=2, headaxislength=3.5)

        # 显示图形
        plt.show()
        plt.close()

    @staticmethod
    def set_boundary_face_normal_vector(trajectory, path):
        unv_cell = trajectory["unit_norm_v"][0]
        face_type = trajectory["face_type"][0].long()
        cells_face = trajectory["cells_face"][0].long()
        face_center_pos = trajectory["face_center_pos"][0]
        boundary_face_normal_vector = torch.full_like(face_center_pos, 0.0)

        cell_face_type = face_type[cells_face]
        mask_cell_face_type = (
            (cell_face_type == NodeType.INFLOW)
            | (cell_face_type == NodeType.WALL_BOUNDARY)
            | (cell_face_type == NodeType.OUTFLOW)
        ).squeeze(2)

        boundary_face_normal_vector[cells_face[mask_cell_face_type]] = unv_cell[
            mask_cell_face_type
        ]
        trajectory[
            "boundary_face_normal_vector"
        ] = boundary_face_normal_vector.unsqueeze(0)

        """plot"""

        # global g__count
        # g__count+=1
        # subdir = path
        # saving_dir = f"{subdir}/case{g__count}"
        # os.makedirs(saving_dir,exist_ok=True)
        # mesh_pos = trajectory["mesh_pos"][0]
        # cells_node  = trajectory["cells_node"][0].long()
        # # unit_normal_vector=mesh["unit_norm_v"][0]
        # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node.view(-1,3))
        # ax.triplot(triang, 'k-', ms=0.5, lw=0.3)
        # ax.set_title('boundary_unit_norm_vector')
        # ax.set_aspect('equal')
        # ax.triplot(triang, 'k-', ms=0.5, lw=0.3, zorder=1)

        # ax.quiver(face_center_pos[cells_face[mask_cell_face_type],0].numpy(),face_center_pos[cells_face[mask_cell_face_type],1].numpy(),boundary_face_normal_vector[cells_face[mask_cell_face_type],0].numpy(),boundary_face_normal_vector[cells_face[mask_cell_face_type],1].numpy(),units='height',color="red", angles='xy',scale_units='xy', scale=20,width=0.01, headlength=3, headwidth=2, headaxislength=4.5)

        # plt.savefig(f"{saving_dir}/boundary unit norm vector distribution.png")
        # plt.close()
        """plot"""

        return trajectory

    @staticmethod
    def fix_face_type(trajectory):
        face_type = trajectory["face_type"][0].long()
        neighbour_cell = trajectory["neighbour_cell"][0].long()
        senders_cell, recivers_cell = neighbour_cell

        # use self-loop to generate boundary mask
        boundary_mask_from_self_loop = (senders_cell == recivers_cell).view(-1)

        boundary_mask_from_face_type = (face_type != NodeType.NORMAL).view(-1)

        total_boundary_mask = torch.logical_and(
            boundary_mask_from_self_loop, boundary_mask_from_face_type
        )

        total_interior_mask = torch.logical_not(total_boundary_mask)

        face_type[total_interior_mask] = NodeType.NORMAL

        trajectory["face_type"] = face_type.unsqueeze(0)

        return trajectory

    @staticmethod
    def recover_mesh_and_node_type(trajectory):
        try:
            trajectory["node_type"] = trajectory["original_node_type"].clone()
            trajectory["mesh_pos"] = trajectory["original_mesh_pos"].clone()
            trajectory["face_center_pos"] = trajectory[
                "original_face_center_pos"
            ].clone()
            trajectory["centroid"] = trajectory["original_centroid"].clone()
            trajectory["unit_norm_v"] = trajectory["original_unit_norm_v"].clone()

        except:
            trajectory["original_node_type"] = trajectory["node_type"].clone()
            trajectory["original_mesh_pos"] = trajectory["mesh_pos"].clone()
            trajectory["original_face_center_pos"] = trajectory[
                "face_center_pos"
            ].clone()
            trajectory["original_centroid"] = trajectory["centroid"].clone()
            trajectory["original_unit_norm_v"] = trajectory["unit_norm_v"].clone()

        return trajectory

    @staticmethod
    def rotate_mesh(trajectory):
        aoa_rad = math.radians(trajectory["aoa"])

        mesh_pos = trajectory["mesh_pos"][0]
        face_center_pos = trajectory["face_center_pos"][0]
        centroid = trajectory["centroid"][0]
        unv = trajectory["unit_norm_v"][0]

        # 创建旋转矩阵
        rotation_matrix = torch.tensor(
            [
                [math.cos(aoa_rad), -math.sin(aoa_rad)],
                [math.sin(aoa_rad), math.cos(aoa_rad)],
            ],
            device=mesh_pos.device,
        )

        trajectory["mesh_pos"] = torch.matmul(mesh_pos, rotation_matrix).unsqueeze(0)
        trajectory["face_center_pos"] = torch.matmul(
            face_center_pos, rotation_matrix
        ).unsqueeze(0)
        trajectory["centroid"] = torch.matmul(centroid, rotation_matrix).unsqueeze(0)
        trajectory["unit_norm_v"] = torch.matmul(unv, rotation_matrix).unsqueeze(0)

        return trajectory

    @staticmethod
    def scale_mesh(trajectory, params, scale):
        if params.scale_mesh is not None and scale:
            mesh_pos = (trajectory["mesh_pos"]) * params.scale_mesh
            trajectory["mesh_pos"] = mesh_pos
            face_node = trajectory["face"][0].to(torch.long)
            cells_node = trajectory["cells_node"][0].to(torch.long)
            cells_face = trajectory["cells_face"][0].to(torch.long)
            cells_index = trajectory["cells_index"][0].to(torch.long)
            face_center_pos = trajectory["face_center_pos"][0]
            cells_face_unv_bias = trajectory["unit_norm_v"][0]
            cells_face_length = trajectory["face_length"][0][cells_face]
            surface_vector = cells_face_unv_bias * cells_face_length

            # recalculating mesh properties
            face_length = torch.norm(
                mesh_pos[0][face_node[0]] - mesh_pos[0][face_node[1]],
                dim=1,
                keepdim=True,
            )
            trajectory["face_length"] = face_length.unsqueeze(0)
            trajectory["centroid"] = (
                (
                    mesh_pos[0][cells_node[0]]
                    + mesh_pos[0][cells_node[1]]
                    + mesh_pos[0][cells_node[2]]
                )
                / 3.0
            ).unsqueeze(0)

            surface_vector = surface_vector
            full_synataic_function = 0.5 * face_center_pos[cells_face.view(-1)]

            cells_area = calc_cell_centered_with_node_attr(
                node_attr=(full_synataic_function * surface_vector).sum(
                    dim=1, keepdim=True
                ),
                cells_node=cells_face,
                cells_index=cells_index,
                reduce="sum",
                map=False,
            )

            trajectory["cells_area"] = cells_area.unsqueeze(0)

        return trajectory

    @staticmethod
    def move_mesh_center(trajectory):
        if not "moved_mesh" in trajectory:
            mesh_pos = trajectory["mesh_pos"][0]

            center_mesh_pos = torch.mean(mesh_pos, dim=0, keepdim=True)

            moved_mesh_pos = mesh_pos - center_mesh_pos
            trajectory["mesh_pos"] = moved_mesh_pos.unsqueeze(0)

            face_node = trajectory["face"][0].to(torch.long)
            cells_index = trajectory["cells_index"][0].to(torch.long)
            cells_node = trajectory["cells_node"][0].to(torch.long)

            moved_face_center_pos = (
                mesh_pos[face_node[0]] + mesh_pos[face_node[1]]
            ) / 2.0

            moved_centroid = calc_cell_centered_with_node_attr(
                moved_mesh_pos, cells_node, cells_index, reduce="mean", map=True
            )
            trajectory["face_center_pos"] = moved_face_center_pos.unsqueeze(0)
            trajectory["centroid"] = moved_centroid.unsqueeze(0)
            trajectory["moved_mesh"] = True

        return trajectory

    @staticmethod
    def calc_cells_node_face_unv(trajectory):
        if not "cells_node_face_unv" in trajectory:
            cells_face_unv = trajectory["unit_norm_v"][0]

            cells_node_face = trajectory["cells_node_face"][0]

            cells_node_face_unv = 0.5 * (
                cells_face_unv[cells_node_face[:, 0]]
                + cells_face_unv[cells_node_face[:, 1]]
            )

            trajectory["cells_node_face_unv"] = cells_node_face_unv.unsqueeze(0)

        return trajectory

    @staticmethod
    def edge_direction_upwind_bias(trajectory):
        if not "edge_biased" in trajectory:
            face_node = trajectory["face"][0].long()
            mesh_pos = trajectory["mesh_pos"][0]

            # 获取边的起点和终点的坐标
            start_points = mesh_pos[face_node[0], :]
            end_points = mesh_pos[face_node[1], :]

            # 计算每条边的方向向量
            edge_directions = start_points - end_points

            # 检查每条边的方向是否指向第1或第2象限
            # 第1象限：x > 0, y >= 0; 第2象限：x < 0, y > 0
            not_first_or_second_quadrant = ~(
                (edge_directions[:, 0] > 0) & (edge_directions[:, 1] >= 0)
            ) & ~((edge_directions[:, 0] < 0) & (edge_directions[:, 1] > 0))

            # 获取需要交换节点索引的边
            swap_indices = not_first_or_second_quadrant.nonzero().squeeze()

            # 交换这些边的节点索引
            face_node[:, swap_indices] = face_node[:, swap_indices].flip(0)

            trajectory["face"] = face_node.unsqueeze(0)
            trajectory["edge_biased"] = True

            return trajectory
        else:
            return trajectory

    @staticmethod
    def calc_symmetric_ghost_pos(
        boundary_edge_pos_left=None,
        boundary_edge_pos_right=None,
        interior_centroid=None,
    ):
        """boundary_edge_pos:[num_edges,2,2]"""

        # Unpack the edge positions for clarity
        x1, y1 = boundary_edge_pos_left[:, 0], boundary_edge_pos_left[:, 1]
        x2, y2 = boundary_edge_pos_right[:, 0], boundary_edge_pos_right[:, 1]

        # Unpack the vertex positions for clarity
        x3, y3 = interior_centroid[:, 0], interior_centroid[:, 1]

        # Calculate vectors AB and AC
        AB = torch.stack((x2 - x1, y2 - y1), dim=1)
        AC = torch.stack((x3 - x1, y3 - y1), dim=1)

        # Calculate the projection of AC onto AB
        proj_len = (AC * AB).sum(dim=1, keepdim=True) / (AB * AB).sum(
            dim=1, keepdim=True
        )
        AP = AB * proj_len

        # Calculate the coordinates of the projection point P
        P = torch.stack((x1, y1), dim=1) + AP

        # Calculate the coordinates of the symmetric point D
        D = 2 * P - torch.stack((x3, y3), dim=1)

        return D

    @staticmethod
    def calc_WLSQ_A_B_normal_matrix(trajectory):
        
        if not "B_node_to_node" in trajectory.keys():
            """>>> compute WLSQ node to node left A matrix >>>"""
            mesh_pos = trajectory["mesh_pos"][0]
            centroid = trajectory["centroid"][0]
            cells_node = trajectory["cells_node"][0].to(torch.long).view(-1)
            cells_index = trajectory["cells_index"][0].to(torch.long).view(-1)
            face_node = trajectory["face"][0].to(torch.long)

            # node to node contribution
            senders_node, recivers_node = face_node[0], face_node[1]

            outdegree_node_index = torch.cat((senders_node, recivers_node), dim=0)

            indegree_node_index = torch.cat((recivers_node, senders_node), dim=0)

            mesh_pos_diff_on_edge = (
                mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
            ).unsqueeze(2)

            mesh_pos_diff_on_edge_T = mesh_pos_diff_on_edge.transpose(1, 2)

            weight_node_to_node = 1.0 / torch.norm(
                mesh_pos_diff_on_edge, dim=1, keepdim=True
            )

            left_on_edge_node_to_node = torch.matmul(
                mesh_pos_diff_on_edge * weight_node_to_node,
                mesh_pos_diff_on_edge_T * weight_node_to_node,
            )

            A_node_to_node = scatter(
                src=left_on_edge_node_to_node,
                index=indegree_node_index,
                dim=0,
                reduce="sum",
            )
 
            trajectory["A_node_to_node"]  = (A_node_to_node).unsqueeze(0).to(
                torch.float32
            )
            """ <<< compute WLSQ node to node left A matrix<<< """

            """ >>> compute WLSQ node to node right B matrix >>> """
            B_node_to_node = (weight_node_to_node**2) * mesh_pos_diff_on_edge

            trajectory["B_node_to_node"] = (
                torch.chunk(B_node_to_node, 2, dim=0)[0]
                .unsqueeze(0)
                .to(torch.float32)
            )
            """ <<< compute WLSQ node to node right B matrix<<< """

            
        if not "B_node_to_node_x" in trajectory.keys():
            """>>> compute WLSQ node to node left A matrix >>>"""
            mesh_pos = trajectory["mesh_pos"][0]
            centroid = trajectory["centroid"][0]
            cells_node = trajectory["cells_node"][0].to(torch.long).view(-1)
            cells_index = trajectory["cells_index"][0].to(torch.long).view(-1)
            face_node_x = trajectory["face_node_x"][0].to(torch.long)

            # node to node contribution
            senders_node_x, recivers_node_x = face_node_x[0], face_node_x[1]

            outdegree_node_index_x = torch.cat((senders_node_x, recivers_node_x), dim=0)

            indegree_node_index_x = torch.cat((recivers_node_x, senders_node_x), dim=0)

            mesh_pos_diff_on_edge_x = (
                mesh_pos[outdegree_node_index_x] - mesh_pos[indegree_node_index_x]
            ).unsqueeze(2)

            mesh_pos_diff_on_edge_x_T = mesh_pos_diff_on_edge_x.transpose(1, 2)

            weight_node_to_node_x = 1.0 / torch.norm(
                mesh_pos_diff_on_edge_x, dim=1, keepdim=True
            )

            left_on_edge_node_to_node_x = torch.matmul(
                mesh_pos_diff_on_edge_x * weight_node_to_node_x,
                mesh_pos_diff_on_edge_x_T * weight_node_to_node_x,
            )

            A_node_to_node_x = scatter(
                src=left_on_edge_node_to_node_x,
                index=indegree_node_index_x,
                dim=0,
                reduce="sum",
            )
 
            trajectory["A_node_to_node_x"]  = (A_node_to_node_x).unsqueeze(0).to(
                torch.float32
            )
            """ <<< compute WLSQ node to node left A matrix<<< """

            """ >>> compute WLSQ node to node right B matrix >>> """
            B_node_to_node_x = (weight_node_to_node_x**2) * mesh_pos_diff_on_edge_x

            trajectory["B_node_to_node_x"] = (
                torch.chunk(B_node_to_node_x, 2, dim=0)[0]
                .unsqueeze(0)
                .to(torch.float32)
            )
            """ <<< compute WLSQ node to node right B matrix<<< """

            
        if not "B_cell_to_node" in trajectory.keys():
            """ >>> cell to node contribution >>> """
            mesh_pos = trajectory["mesh_pos"][0]
            centroid = trajectory["centroid"][0]
            cells_node = trajectory["cells_node"][0].to(torch.long).view(-1)
            cells_index = trajectory["cells_index"][0].to(torch.long).view(-1)
            centriod_mesh_pos_diff = (
                centroid[cells_index] - mesh_pos[cells_node]
            ).unsqueeze(2)
            centriod_mesh_pos_diff_T = centriod_mesh_pos_diff.transpose(1, 2)
            weight_cell_node = 1.0 / torch.norm(
                centriod_mesh_pos_diff, dim=1, keepdim=True
            )
            left_on_edge_cell_to_node = torch.matmul(
                centriod_mesh_pos_diff * weight_cell_node,
                centriod_mesh_pos_diff_T * weight_cell_node,
            )
            A_cell_to_node = scatter_add(left_on_edge_cell_to_node, 
                                         cells_node, 
                                         dim=0)
            trajectory["A_cell_to_node"]  = (A_cell_to_node).unsqueeze(0).to(
                torch.float32
            )
            """ <<< cell to node contribution <<< """
            
            """ >>> compute WLSQ cell to node right B matrix >>> """
            B_cell_to_node = (weight_cell_node**2) * centriod_mesh_pos_diff

            trajectory["B_cell_to_node"] = (
                B_cell_to_node
                .unsqueeze(0)
                .to(torch.float32)
            )
            """ <<< compute WLSQ cell to node right B matrix<<< """
            
            
        if not "B_cell_to_cell_x" in trajectory.keys():
            """>>> compute WLSQ cell to cell left A matrix >>>"""
            mesh_pos = trajectory["mesh_pos"][0]
            centroid = trajectory["centroid"][0].to(torch.float64)
            face_node = trajectory["face"][0].to(torch.long)
            neighbour_cell = trajectory["neighbour_cell"][0].to(torch.long)
            neighbour_cell_x = trajectory["neighbour_cell_x"][0].to(torch.long)

            senders_cell, recivers_cell = neighbour_cell[0], neighbour_cell[1]
            twoway_face_node = torch.cat((face_node, face_node.flip(0)), dim=1)

            outdegree_cell = torch.cat((senders_cell, recivers_cell), dim=0)
            indegree_cell = torch.cat((recivers_cell, senders_cell), dim=0)

            mask = (indegree_cell == outdegree_cell).view(-1)
            in_centroid = centroid[indegree_cell].clone()

            boundary_edge_index = twoway_face_node[:, mask]
            boundary_edge_interior_centroid = in_centroid[mask]

            """>>>      ghost cell centroid       >>>"""
            symmetry_pos = CFDdatasetBase.calc_symmetric_ghost_pos(
                boundary_edge_pos_left=mesh_pos[boundary_edge_index[0]],
                boundary_edge_pos_right=mesh_pos[boundary_edge_index[1]],
                interior_centroid=boundary_edge_interior_centroid,
            )

            senders_cell_x, recivers_cell_x = neighbour_cell_x[0], neighbour_cell_x[1]
            outdegree_cell_x = torch.cat((senders_cell_x, recivers_cell_x), dim=0)
            indegree_cell_x = torch.cat((recivers_cell_x, senders_cell_x), dim=0)

            out_centroid_x = centroid[outdegree_cell_x].clone()
            in_centroid_x = centroid[indegree_cell_x].clone()

            mask_x = (indegree_cell_x == outdegree_cell_x).view(-1)
            out_centroid_x[mask_x] = symmetry_pos
            mask_interior = torch.logical_not(mask_x)
            """<<<      ghost cell centroid       <<<"""

            # cell to cell contributation
            centroid_diff_on_edge = ((out_centroid_x - in_centroid_x)).unsqueeze(2)

            centroid_diff_on_edge_T = centroid_diff_on_edge.transpose(1, 2)

            weight_cell_to_cell_x = 1.0 / torch.norm(
                centroid_diff_on_edge, dim=1, keepdim=True
            )

            left_on_edge_cell_to_cell = torch.matmul(
                centroid_diff_on_edge * weight_cell_to_cell_x,
                centroid_diff_on_edge_T * weight_cell_to_cell_x,
            )[mask_interior]

            A_cell_to_cell_x = scatter(
                left_on_edge_cell_to_cell,
                indegree_cell_x[mask_interior],
                dim=0,
                dim_size=centroid.shape[0],
            )
            # cell to cell contributation

            # node to cell contributation
            cells_node = cells_node
            cells_index = cells_index
            node_to_centroid_pos_diff = (
                mesh_pos[cells_node] - centroid[cells_index]
            ).unsqueeze(2)
            node_to_centroid_pos_diff_T = node_to_centroid_pos_diff.transpose(1, 2)

            weight_node_to_centroid = 1.0 / torch.norm(
                node_to_centroid_pos_diff, dim=1, keepdim=True
            )

            left_on_edge_node_to_centroid = torch.matmul(
                node_to_centroid_pos_diff * weight_node_to_centroid,
                node_to_centroid_pos_diff_T * weight_node_to_centroid,
            )

            A_node_to_cell = calc_cell_centered_with_node_attr(
                node_attr=left_on_edge_node_to_centroid,
                cells_node=cells_node,
                cells_index=cells_index,
                reduce="sum",
                map=False,
            )

            # node to cell contributation
            A_inv_cell_to_cell_x = torch.linalg.inv(A_cell_to_cell_x + A_node_to_cell)
            trajectory["A_inv_cell_to_cell_x"] = A_inv_cell_to_cell_x.unsqueeze(0).to(
                torch.float32
            )
            """ <<< compute WLSQ cell to cell left A matrix <<< """

            """ >>> compute WLSQ cell to cell right B matrix >>> """
            # compute WLSQ cell to cell left B matrix
            B_cell_to_cell_x = (weight_cell_to_cell_x**2) * centroid_diff_on_edge

            # node to cell contributation
            B_node_to_cell = (weight_node_to_centroid**2) * node_to_centroid_pos_diff
            """ <<< compute WLSQ cell to cell right B matrix<<< """

            trajectory["B_cell_to_cell_x"] = (
                torch.chunk(B_cell_to_cell_x, 2, dim=0)[0]
                .unsqueeze(0)
                .to(torch.float32)
            )
            trajectory["B_node_to_cell"] = B_node_to_cell.unsqueeze(0).to(torch.float32)

            return trajectory

        return trajectory
    
    @staticmethod
    def calc_WLSQ_A_B_matrix(trajectory):
        
        if not "R_inv_Q_t" in trajectory.keys():
            """>>> compute WLSQ node to node R_inv_Q_t matrix >>>"""
            mesh_pos = trajectory["mesh_pos"][0]
            node_neigbors = trajectory["node_neigbors"][0].to(torch.long)
            max_neighbors = trajectory["max_neighbors"]
                
            mask_fil = (node_neigbors!=-1).unsqueeze(2)
            neigbor_pos = mesh_pos[node_neigbors]*mask_fil
            moments_left = neigbor_pos-mesh_pos.unsqueeze(1)
            weight_unfiltered = 1./torch.norm(moments_left,dim=2,keepdim=True)
            weight = torch.where(torch.isfinite(weight_unfiltered),weight_unfiltered,0.)*mask_fil
            A = weight*moments_left
            
            Q,R = torch.linalg.qr(A)
            
            # # 创建同维度的单位矩阵
            # I = torch.eye(R.shape[1]).unsqueeze(0).repeat(R.shape[0],1,1)

            # # 使用torch.linalg.solve_triangular一次性求解整个逆矩阵
            # R_inv = torch.linalg.solve_triangular(R, I, upper=True)
            R_inv = torch.linalg.inv(R)
            
            R_inv_Q_t = torch.matmul(R_inv,Q.transpose(1,2))*(weight.transpose(1,2))
            
            if A.shape[1]<max_neighbors:
                R_inv_Q_t = torch.cat((R_inv_Q_t,torch.zeros((R_inv_Q_t.shape[0],R_inv_Q_t.shape[1],max_neighbors-R_inv_Q_t.shape[2]),device=R_inv_Q_t.device)),dim=2)
                node_neigbors = torch.cat((node_neigbors,torch.full((node_neigbors.shape[0],max_neighbors-A.shape[1]),-1,device=node_neigbors.device)),dim=1)
                mask_fil = torch.cat((mask_fil,torch.full((mask_fil.shape[0],max_neighbors-mask_fil.shape[1],mask_fil.shape[2]),False,device=mask_fil.device)),dim=1)
                mask_fil = torch.where(mask_fil,1.,0.)
                
            trajectory["R_inv_Q_t"] = R_inv_Q_t.unsqueeze(0).to(torch.float32)
            trajectory["node_neigbors"] = node_neigbors.unsqueeze(0)
            trajectory["mask_node_neigbors_fil"] = mask_fil.unsqueeze(0)
            
        return trajectory

    @staticmethod
    def random_boolean():
        return torch.rand(1).item() < 0.2

    @staticmethod
    def find_fluid_property_comb(fluid_property_comb_all, flow_type):
        if flow_type == "cavity_flow":
            return fluid_property_comb_all["cf"]
        elif flow_type == "pipe_flow":
            return fluid_property_comb_all["pf"]
        elif "farfield" in flow_type:
            return fluid_property_comb_all["ff"]
        elif "possion" in flow_type:
            return fluid_property_comb_all["p"]
        elif "cavity_wave" in flow_type:
            return fluid_property_comb_all["cw"]

    @staticmethod
    def transform_trajectory(
        trajectory,
        params,
        scale=False,
        is_training=False,
        spec_u_rho_mu_comb=None,
        inflow_bc_type=None,
        plot=False,
        fluid_property_comb_all=None,
    ):
        # we set default flow type=2, means for compact with old uniform 1200 dataset
        if not isinstance(trajectory["flow_type"], str):
            trajectory["flow_type"] = flow_type_mapping.get(
                str(trajectory["flow_type"][0, 0, 0].item())
            )

        if "cavity" in trajectory["flow_type"] and (
            "wave" not in trajectory["flow_type"]
        ):
            if ("possion" in params.equation_state) and (
                CFDdatasetBase.random_boolean()
            ):
                trajectory["flow_type"] = "cavity_possion"

            else:
                # let`s compose pressure constraint point boundary condition
                trajectory["flow_type"] = "cavity_flow"
                node_type = trajectory["node_type"][0]
                inflow_mask = (
                    (node_type == NodeType.INFLOW) | (node_type == NodeType.IN_WALL)
                ).view(-1)
                mesh_pos = trajectory["mesh_pos"][0]
                inflow_pos = mesh_pos[inflow_mask]

                # put pressure constraint point at the center of inlet
                median = torch.median(inflow_pos[:, 0:1])

                median_index = inflow_pos[:, 0] == median

                # _,left_pos_index = torch.min(inflow_pos[:,0:1],dim=0)
                inflow_node_type = node_type[inflow_mask]
                inflow_node_type[median_index] = NodeType.IN_WALL
                node_type[inflow_mask] = inflow_node_type
                trajectory["node_type"] = node_type.unsqueeze(0)

        trajectory["cell_factor"] = trajectory["cells_factor"]
        trajectory["time_steps"] = 1

        # trajectory = CFDdatasetBase.move_mesh_center(trajectory)

        trajectory = CFDdatasetBase.scale_mesh(trajectory, params, scale)

        trajectory = CFDdatasetBase.calc_cells_node_face_unv(trajectory)

        fluid_property_comb = CFDdatasetBase.find_fluid_property_comb(
            fluid_property_comb_all, trajectory["flow_type"]
        )

        trajectory, mean_velocity = CFDdatasetBase.makedimless(
            trajectory,
            params,
            fluid_property_comb=fluid_property_comb,
            spec_u_rho_mu_comb=spec_u_rho_mu_comb,
        )

        trajectory = CFDdatasetBase.init_env(
            trajectory,
            mean_velocity=mean_velocity,
            dimless=params.dimless,
            flow_type=trajectory["flow_type"],
            inflow_bc_type=inflow_bc_type,
        )

        # trajectory = CFDdatasetBase.edge_direction_upwind_bias(trajectory)

        trajectory = CFDdatasetBase.calc_WLSQ_A_B_normal_matrix(trajectory)

        if (
            ("boundary_zone" not in trajectory)
            and ("cavity" not in trajectory["flow_type"])
            and ("possion" not in trajectory["flow_type"])
        ):
            boundary_zone = extract_cylinder_boundary_only_training(
                dataset=trajectory,
                params=params,
                rho=trajectory["rho"].item(),
                mu=trajectory["mu"].item(),
                dt=trajectory["dt"][0, 0, 0].item(),
            )
            trajectory["boundary_zone"] = boundary_zone

        return trajectory


class CFDdatasetIt(IterableDataset):
    def __init__(
        self,
        params,
        path,
        split,
        max_epochs=600,
        is_training=False,
        dataset_type="tf",
        spec_u_rho_mu_comb=None,
        inflow_bc_type="parabolic_velocity_field",
        fluid_property_comb_all=None,
    ):
        super().__init__()
        self.is_training = is_training
        self.steps = max_epochs
        self.path = path
        self.split = split
        self.dataset_dir = path
        self.params = params
        self.fluid_property_comb_all = fluid_property_comb_all

        if dataset_type == "tf":
            self.tfrecord_path = os.path.join(path, split + ".tfrecord")

            # index is generated by tfrecord2idx
            self.index_path = os.path.join(path, split + ".idx")
            self.file_handle = TFRecordDataset(
                self.tfrecord_path, self.index_path, transform=self.process_trajectory
            )

        elif dataset_type == "h5":
            self.file_handle = h5py.File(self.dataset_dir + f"/{split}.h5", "r")

        self.load_meta = False

        # user specified boundary condition and equation
        self.spec_u_rho_mu_comb = spec_u_rho_mu_comb
        self.inflow_bc_type = inflow_bc_type

    def __iter__(self):
        return iter(self.file_handle)

    def process_trajectory(self, trajectory_data):
        self.shapes = {}
        self.dtypes = {}
        self.types = {}
        self.steps = 600
        if not loaded_meta:
            try:
                with open(os.path.join(self.dataset_dir, "meta.json"), "r") as fp:
                    meta = json.loads(fp.read())
                self.shapes = {}
                self.dtypes = {}
                self.types = {}
                self.steps = meta["trajectory_length"] - 2
                for key, field in meta["features"].items():
                    self.shapes[key] = field["shape"]
                    self.dtypes[key] = field["dtype"]
                    self.types[key] = field["type"]
            except FileNotFoundError as e:
                print(e)
                quit()
            self.load_meta = True
            
        trajectory = {}
        
        if "max_neighbors" in meta.keys():
            trajectory["max_neighbors"] = meta["max_neighbors"]
        
        # decode bytes into corresponding dtypes
        for key, field in meta["features"].items():
            raw_data = trajectory_data[key]
            mature_data = np.frombuffer(raw_data, dtype=getattr(np, field["dtype"]))
            mature_data = torch.from_numpy(mature_data.copy())
            
            if not "node_neigbors" in key:
                reshaped_data = torch.reshape(mature_data, field["shape"])
            else:
                trajectory[key] = mature_data
                continue
            
            if key == "face":
                pass
            else:
                if field["type"] == "static":
                    reshaped_data = torch.tile(reshaped_data, (1, 1, 1))
                elif field["type"] == "dynamic_varlen":
                    pass
                elif field["type"] == "dynamic":
                    pass
                elif field["type"] != "dynamic":
                    raise ValueError("invalid data format")
            trajectory[key] = reshaped_data
        
        if not trajectory:
            raise ValueError("trajectory is empty")
        
        if "node_neigbors" in trajectory.keys():
            trajectory["node_neigbors"] = torch.reshape(trajectory["node_neigbors"],
                                                        (trajectory["node_neigbors_shape"].numpy()[0],
                                                         trajectory["node_neigbors_shape"].numpy()[1])).unsqueeze(0)

        trajectory = CFDdatasetBase.transform_trajectory(
            trajectory,
            self.params,
            scale=False,
            is_training=self.is_training,
            fluid_property_comb_all=self.fluid_property_comb_all,
            spec_u_rho_mu_comb=self.spec_u_rho_mu_comb,
            inflow_bc_type=self.inflow_bc_type,
        )

        return trajectory


class CFDdatasetmap(Dataset):
    def __init__(
        self, params, path, split="train", dataset_type="h5", is_training=False
    ):
        super().__init__()

        self.path = path
        self.split = split
        self.dataset_dir = path
        self.params = params
        self.is_training = is_training
        if dataset_type == "h5":
            self.file_handle = h5py.File(self.dataset_dir + f"/{split}.h5", "r")
        else:
            raise ValueError("invalid data format")

    def __getitem__(self, index):
        trajectory_handle = self.file_handle[str(index)]
        trajectory = {}

        for key in trajectory_handle.keys():
            trajectory[key] = torch.from_numpy(trajectory_handle[key][:])

        trajectory = CFDdatasetBase.transform_trajectory(
            trajectory, self.params, scale=False, is_training=self.is_training
        )

        return trajectory

    def __len__(self):
        return len(self.file_handle)


class Data_Pool:
    def __init__(self, params=None, is_training=True, device=None, state_save_dir=None):
        self.params = params
        self.is_training = is_training
        self.lock = Lock()
        self.pool = []
        self.mbatch_graph_node = []
        self.mbatch_graph_node_x = []
        self.mbatch_graph_cell = []
        self.mbatch_graph_cell_x = []
        self.mbatch_graph_edge = []
        self.data_target_on_node = []
        self.origin_mesh_file_location = []
        self.has_boundary = []
        self.target = 0
        self.device = device

        try:
            if not (state_save_dir.find("traing_results") != -1):
                os.makedirs(f"{state_save_dir}/traing_results", exist_ok=True)
                self.state_save_dir = f"{state_save_dir}/traing_results"
        except:
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>Warning, no state_save_dir is specified, check if traing states is specified<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
        self.fluid_property_comb_all = {}
        # do not change this value
        self.epoch = 0
        self.forwardtrick = False

    def _set_epoch(self, epoch):
        self.epoch = epoch

    def _set_fetch_time_steps(self, epoch):
        self.train_epoch = epoch

    def _set_reset_env_flag(self, flag=False, rst_time=1):
        self.reset_env_flag = flag
        self.rst_time = rst_time

    def _set_plot_flag(self, _plot=False):
        self._plot_env = _plot

    def _set_status(self, is_training=False):
        self.is_training = is_training

    def _set_time_step_list(self, new_time_step_list: np.ndarray):
        self._time_step_list = new_time_step_list

    def _set_forwardtrick(self, using_pushforwardtrick=False):
        self.forwardtrick = using_pushforwardtrick

    @staticmethod
    def randbool(*size, device=None):
        """Returns 50% channce of True of False"""
        return torch.randint(2, size, device=device) == torch.randint(
            2, size, device=device
        )

    def _debug_time_steps(self):
        np.savetxt(
            "/home/litianyu/mycode/debug/time_steps_logger_"
            + str(self.train_epoch)
            + ".txt",
            self._time_step_list,
            fmt="%d",
            delimiter=",",
        )

    def random_time_steps(self, data_idx):
        choosed_time_steps = self._time_step_list[data_idx][
            self.train_epoch % self.params.train_traj_length
        ]
        return choosed_time_steps

    def collate(self, batch_list):
        return batch_list

    def load_mesh_to_cpu(
        self,
        split="train",
        dataset_dir=None,
        spec_u_rho_mu_comb=None,
        inflow_bc_type="parabolic_velocity_field",
    ):
        for trainging_flow_type in self.params.training_flow_type:
            prefix = trainging_flow_type
            U_range = getattr(self.params, f"{prefix}_inflow_range")
            rho_range = getattr(self.params, f"{prefix}_rho")
            mu_range = getattr(self.params, f"{prefix}_mu")
            Re_max = getattr(self.params, f"{prefix}_Re_max")
            Re_min = getattr(self.params, f"{prefix}_Re_min")
            source_range = getattr(self.params, f"{prefix}_source")
            aoa_range = getattr(self.params, f"{prefix}_aoa")
            dt = getattr(self.params, f"{prefix}_dt")
            L = getattr(self.params, f"{prefix}_L")

            self.fluid_property_comb_all[
                trainging_flow_type
            ] = get_param.generate_combinations(
                U_range,
                rho_range,
                mu_range,
                Re_max,
                Re_min,
                source_range,
                aoa_range,
                dt,
                L=L,
            )

        if self.params.dataset_type == "tf":
            test_ds = CFDdatasetIt(
                self.params,
                dataset_dir,
                split=split,
                dataset_type=self.params.dataset_type,
                is_training=self.is_training,
                spec_u_rho_mu_comb=spec_u_rho_mu_comb,
                inflow_bc_type=inflow_bc_type,
                fluid_property_comb_all=self.fluid_property_comb_all,
            )
            
        elif self.params.dataset_type == "h5":
            test_ds = CFDdatasetmap(
                params=self.params,
                path=dataset_dir,
                split=split,
                dataset_type="h5",
                is_training=self.is_training,
            )
            
        ds_loader = torch_DataLoader(
            test_ds,
            batch_size=25,
            num_workers=4,
            pin_memory=False,
            collate_fn=lambda x: x,
        )
        
        print("loading whole dataset to cpu")
        read_done=False
        read_round=0
        while True:
            for _, trajs in enumerate(ds_loader):
                tmp = list(trajs)
                for samples in tmp:
                    for k, v in samples.items():
                        if isinstance(v, torch.Tensor) and self.params.all_on_gpu:
                            samples[k] = v.to(self.device)
                        if k == "origin_mesh_path":
                            origin_mesh_dir = "".join(
                                [chr(int(f)) for f in v[0, :, 0].numpy()]
                            )
                            if (
                                "pipe_flow/pipe_flow_cylinder/mesh4/mesh-1.dat"
                                in origin_mesh_dir
                            ):
                                print(f"target mesh id:{len(self.pool)}")
                    if read_round<1:
                        self.pool.append(samples)
                    # after one round of reading, if pool is still not statisfied 
                    # params.datasetsize, we start another round of reading
                    elif (len(self.pool)<self.params.dataset_size) and (read_round>=1):
                        self.pool.append(samples)
                    else:
                        read_done=True
                        break
            # read done
            if read_done:
                break
            read_round+=1

        self.dataset_size = len(self.pool)
        self.params.dataset_size = self.dataset_size
        self.inflow_bc_type = inflow_bc_type
        self._time_step_list = (
            torch.from_numpy(
                np.random.permutation(self.params.train_traj_length).astype(np.int64)
            )
            .view(1, -1)
            .repeat(self.dataset_size, 1)
            .numpy()
        )

        # count for reseting environment
        self._reset_env_order = list(range(self.dataset_size))
        random.shuffle(self._reset_env_order)
        self.reset_env_flag = False
        self._plot_env = False
        self.i_count = 0
        self.p_count = 0

        # for manul dataloader
        self.__num_samples = self.dataset_size
        self.__indices = list(range(self.__num_samples))
        random.shuffle(self.__indices)

        return self.dataset_size, self.params

    @staticmethod
    def datapreprocessing(
        graph_node, graph_node_x, graph_edge, graph_cell, graph_cell_x, dimless=False
    ):
        # normliaze node uvp
        uvp_node = graph_node.x[:, 0:3]
        pde_theta_node = graph_node.pde_theta_node

        # permute edge direction
        senders, receivers = graph_node.edge_index
        single_way_senders = senders
        single_way_receivers = receivers

        releative_mesh_pos = (
            torch.index_select(graph_node.pos, 0, single_way_senders)
            - torch.index_select(graph_node.pos, 0, single_way_receivers)
        ).to(torch.float32)

        edge_length = torch.norm(releative_mesh_pos, dim=1, keepdim=True)

        releative_node_attr = torch.index_select(
            uvp_node, 0, single_way_senders
        ) - torch.index_select(uvp_node, 0, single_way_receivers)

        graph_node.x = torch.cat((uvp_node, pde_theta_node), dim=1)

        graph_node.edge_attr = torch.cat(
            (releative_node_attr, releative_mesh_pos, edge_length), dim=1
        )

        return [graph_node, graph_node_x, graph_edge, graph_cell, graph_cell_x]

    def require_target(self, rollout_index):
        target = self.pool[rollout_index]

        return torch.cat(
            (target["target|velocity_on_node"], target["target|pressure_on_node"]),
            dim=2,
        )

    def prob_return(self, p):
        # p is a probability between 0 and 1
        # returns True with probability p and False with probability 1-p
        assert 0 <= p <= 1, "p must be a valid probability"
        return random.random() < p

    def reset_env(self, idx, graph_node_old=None, synatic=False, plot=False):
        self.mbatch_graph_node.clear()
        self.mbatch_graph_node_x.clear()
        self.mbatch_graph_cell.clear()
        self.mbatch_graph_edge.clear()
        self.has_boundary.clear()

        # for plotting
        current_dataset = self.pool[idx]

        if graph_node_old is not None and (self.params.all_on_gpu or synatic):
            for k, v in current_dataset.items():
                if isinstance(v, torch.Tensor):
                    current_dataset[k] = v.cpu()

            current_dataset["velocity_on_node"] = graph_node_old.x[
                :, 0:2
            ].data.unsqueeze(0)
            current_dataset["pressure_on_node"] = graph_node_old.x[
                :, 2:3
            ].data.unsqueeze(0)
            current_dataset["velocity_gradient_on_node"] = graph_node_old.x[
                :, 3:
            ].data.unsqueeze(0)

            """>>>>>>>>>only for synatic function>>>>>>>>>"""
            if synatic:
                current_dataset["target_on_node"] = graph_node_old.y[
                    :, 0:3
                ].data.unsqueeze(0)
                current_dataset["target_gradient_on_node"] = graph_node_old.y[
                    :, 3:
                ].data.unsqueeze(0)
            """<<<<<<<<<only for synatic function<<<<<<<<<"""

        if plot:
            plot_dataset = self.pool[idx]
            self.plot_state(plot_dataset, idx, synatic=synatic)
            self.p_count = self.p_count + 1
            self._plot_env = False

        # start reseting env
        current_dataset = CFDdatasetBase.transform_trajectory(
            current_dataset,
            self.params,
            scale=False,
            is_training=self.is_training,
            plot=False,
            fluid_property_comb_all=self.fluid_property_comb_all,
            inflow_bc_type=self.inflow_bc_type,
        )

        self.pool[idx] = current_dataset
        # for k,v in current_dataset.items():
        #     if isinstance(v,torch.Tensor):
        #         current_dataset[k] = v.cuda()

        # # build graph from dict dataset
        # self._build_graph(current_dataset,graph_indices=idx,reset=True)
        # graph_cell_new = self.mbatch_graph_cell[0]
        # graph_node_new = self.mbatch_graph_node[0]
        # self.pool[idx] = current_dataset
        # print(f"env {idx} has been reset ")

        # return graph_cell_new,graph_node_new

    def export_to_tecplot(self, export_dataset, case_idx, synatic=False):
        to_numpy = lambda x: x.cpu().numpy() if x.is_cuda else x.numpy()
        write_dataset = []
        interior_zone_numpy = {}
        boundary_zone_numpy = {}

        for k, v in export_dataset.items():
            if (not isinstance(v, torch.Tensor)) and (
                not isinstance(v, TimeStepSequence)
            ):
                interior_zone_numpy[k] = v
            elif isinstance(v, TimeStepSequence):
                interior_zone_numpy[k] = to_numpy(
                    v.get_timestep_at_t().unsqueeze(0)
                )  # 待修改
            elif isinstance(v, torch.Tensor):
                interior_zone_numpy[k] = to_numpy(v)
            else:
                interior_zone_numpy[k] = v

        """ >>> interior zone >>> """
        interior_zone_numpy["velocity"] = interior_zone_numpy["velocity_on_node"]
        interior_zone_numpy["pressure"] = interior_zone_numpy["pressure_on_node"]
        if synatic:
            interior_zone_numpy["target|UVP"] = interior_zone_numpy["target_on_node"]
            interior_zone_numpy["target|gradient"] = interior_zone_numpy[
                "target_gradient_on_node"
            ]

        interior_zone_numpy["cells"] = interior_zone_numpy["cells_node"]
        interior_zone_numpy["face_node"] = interior_zone_numpy["face"]
        interior_zone_numpy["zonename"] = "Fluid"
        interior_zone_numpy["data_packing_type"] = ["node"]
        write_dataset.append(interior_zone_numpy)
        """ <<< interior zone <<< """

        """ >>> boundary zone >>> """
        if "boundary_zone" in export_dataset:
            boundary_zone = export_dataset["boundary_zone"]
            mask_node_boundary = boundary_zone["mask_node_boundary"]
            boundary_zone["velocity"] = export_dataset["velocity_on_node"][
                :, mask_node_boundary
            ]
            boundary_zone["pressure"] = export_dataset["pressure_on_node"][
                :, mask_node_boundary
            ]
            boundary_zone["data_packing_type"] = ["node"]
            for k, v in boundary_zone.items():
                if not isinstance(v, torch.Tensor):
                    continue
                boundary_zone_numpy[k] = to_numpy(v)

            write_dataset.append(boundary_zone)
        """ <<< boundary zone <<< """

        aoa = interior_zone_numpy["aoa"]
        mean_u = interior_zone_numpy["mean_u"]
        rho = interior_zone_numpy["rho"]
        mu = interior_zone_numpy["mu"]
        source = interior_zone_numpy["source"]
        dt = (
            interior_zone_numpy["dt"][0, 0, 0] / interior_zone_numpy["uvp_dim"][0, 0, 0]
        )
        flow_type = interior_zone_numpy["flow_type"]

        # 计算大的分块文件夹名
        folder_range = 50  # 每个大文件夹存储50个子文件夹
        parent_folder_name = f"{(int(self.p_count) // folder_range) * folder_range}-{((int(self.p_count) // folder_range) + 1) * folder_range - 1}"
        parent_folder_path = os.path.join(self.state_save_dir, parent_folder_name)

        # 创建大的分块文件夹
        os.makedirs(parent_folder_path, exist_ok=True)

        # 为当前 self.p_count 创建独立的子文件夹
        sub_folder_name = f"case_{int(self.p_count)}"
        sub_folder_path = os.path.join(parent_folder_path, sub_folder_name)
        os.makedirs(sub_folder_path, exist_ok=True)

        # 更新文件保存路径
        save_path = f"{sub_folder_path}/results_No.{int(self.p_count)}_type_{flow_type}_training_case_id_{case_idx}_mean_u{mean_u}_rho_{rho}_mu_{mu}_source{source}_dt_{dt}_aoa_{aoa}.dat"

        # 更新文本文件的保存路径
        text_file_path = f"{sub_folder_path}/case_dir_No.{int(self.p_count)}_type_{flow_type}_training_case_id_{case_idx}_mean_u{mean_u}_rho_{rho}_mu_{mu}_source{source}_dt_{dt}_aoa_{aoa}.txt"

        # 接下来是保存数据的代码...
        with open(text_file_path, "w") as mesh_path_file:
            floats = interior_zone_numpy["origin_mesh_path"][0, :, 0]
            mesh_path_file.write("".join([chr(int(f)) for f in floats]))

        # write_tec.write_tecplotzone(
        #     filename=save_path,
        #     datasets=write_dataset,
        #     time_step_length=1,
        #     has_cell_centered=True,
        #     synatic=synatic,
        # )
        process = multiprocessing.Process(target=write_tecplotzone_in_process, args=(save_path, write_dataset, 1, True, synatic))
        process.start()

    def plot_state(self, current_dataset, case_idx, synatic=False):
        # to_numpy = lambda x: x.cpu().numpy() if x.is_cuda else x.numpy()
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

        # write to tecplot file
        self.export_to_tecplot(current_dataset, case_idx, synatic=synatic)

        # # 设置三角剖分
        # mesh_pos=to_numpy(current_dataset['mesh_pos'][0])
        # cells_node=to_numpy(current_dataset['cells_node'][0])
        # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)

        # u = to_numpy(current_dataset['velocity_on_node'][0,:,0])
        # v = to_numpy(current_dataset['velocity_on_node'][0,:,1])
        # velocity_maginitude = np.abs(np.sqrt(u**2+v**2))

        # # 绘制 velocity maginitude 速度场
        # bb_min_u = (velocity_maginitude).min()
        # bb_max_u = (velocity_maginitude).max()
        # cntr1 = ax1.tripcolor(triang,velocity_maginitude, vmin=bb_min_u, vmax=bb_max_u)
        # ax1.set_title('velocity maginitude')
        # ax1.set_aspect('equal')
        # ax1.triplot(triang, 'k-', ms=0.5, lw=0.3, zorder=1)
        # plt.colorbar(cntr1, ax=ax1,orientation="horizontal",shrink=0.5)

        # # 绘制 v 速度场
        # bb_min_v = (to_numpy(current_dataset['velocity_on_node'][0,:,1])).min()
        # bb_max_v = (to_numpy(current_dataset['velocity_on_node'][0,:,1])).max()
        # cntr2 = ax2.tripcolor(triang,to_numpy(current_dataset['velocity_on_node'][0,:,1]), vmin=bb_min_v, vmax=bb_max_v)
        # ax2.set_title('V_field')
        # ax2.set_aspect('equal')
        # ax2.triplot(triang, 'k-', ms=0.5, lw=0.3, zorder=1)
        # plt.colorbar(cntr2, ax=ax2,orientation="horizontal",shrink=0.5)

        # # 绘制压力场
        # bb_min_p = (to_numpy(current_dataset['pressure_on_node'][0,:,0])).min()
        # bb_max_p = (to_numpy(current_dataset['pressure_on_node'][0,:,0])).max()
        # cntr3 = ax3.tripcolor(triang,to_numpy(current_dataset['pressure_on_node'][0,:,0]), vmin=bb_min_p, vmax=bb_max_p)
        # ax3.set_title('P_field')
        # ax3.set_aspect('equal')
        # ax3.triplot(triang, 'k-', ms=0.5, lw=0.3, zorder=1)
        # plt.colorbar(cntr3, ax=ax3,orientation="horizontal",shrink=0.5)

        # aoa = current_dataset["aoa"]
        # mean_u = current_dataset["mean_u"][0,0,0]
        # # 显示图形
        # plt.tight_layout()
        # plt.savefig(self.state_save_dir+f"/{int(self.p_count)}_velocity_during_training{case_idx}_mean_u{mean_u}_aoa_{aoa}"+".png")
        # plt.close()
        # plt.show()

    def payback(self, graph_node_new, graph_cell_new):
        graph_cell_sets = Batch.to_data_list(graph_cell_new)
        graph_node_sets = Batch.to_data_list(graph_node_new)

        minibatch_threadpool = []

        def decompose_graph_to_dict(graph_cell, graph_node):
            # set decomposed graph to data pool dict
            current_dataset = self.pool[graph_node_new.graph_index]

            current_dataset["velocity_on_node"] = graph_node.x[:, 0:2].unsqueeze(0).data
            current_dataset["pressure_on_node"] = graph_node.x[:, 2:3].unsqueeze(0).data

            self.pool[graph_node_new.graph_index] = current_dataset

        for graph_index in range(len(graph_cell_sets)):
            p = Thread(
                target=decompose_graph_to_dict,
                args=(
                    graph_cell_sets[graph_index],
                    graph_node_sets[graph_index],
                ),
            )
            minibatch_threadpool.append(p)
            p.start()

        for i in range(len(minibatch_threadpool)):
            minibatch_threadpool[i].join()

        # Usually, every 40 train steps will reset 1 env
        if self.reset_env_flag:
            for _ in range(self.rst_time):
                if not self._reset_env_order:  # If indices list is empty, re-shuffle
                    self._reset_env_order = list(range(self.params.dataset_size))
                    random.shuffle(self._reset_env_order)

                self.i_count = self._reset_env_order.pop()
                self.reset_env(int(self.i_count), plot=self._plot_env)
                self.reset_env_flag = False

    def payback_test(self, graph_node_new):
        def decompose_graph_to_dict(graph_node):
            # set decomposed graph to data pool dict
            current_dataset = self.pool[graph_node.graph_index]

            current_dataset["velocity_on_node"] = (
                graph_node.x[:, 0:2].unsqueeze(0).cpu().data
            )
            current_dataset["pressure_on_node"] = (
                graph_node.x[:, 2:3].unsqueeze(0).cpu().data
            )

            self.pool[graph_node.graph_index] = current_dataset

        decompose_graph_to_dict(graph_node_new)

        # Usually, every 40 train steps will reset 1 env
        if self.reset_env_flag:
            for _ in range(self.rst_time):
                if not self._reset_env_order:  # If indices list is empty, re-shuffle
                    self._reset_env_order = list(range(self.params.dataset_size))
                    random.shuffle(self._reset_env_order)

                self.i_count = self._reset_env_order.pop()
                self.reset_env(int(self.i_count), plot=self._plot_env)
                self.reset_env_flag = False
                
    def payback_sp(self, graph_node_new):
        
        def decompose_graph_to_dict(graph_node):
            # set decomposed graph to data pool dict
            current_dataset = self.pool[graph_node.graph_index]

            current_dataset["velocity_on_node"] = (
                graph_node.x[:, 0:2].unsqueeze(0).cpu().data
            )
            current_dataset["pressure_on_node"] = (
                graph_node.x[:, 2:3].unsqueeze(0).cpu().data
            )

            self.pool[graph_node.graph_index] = current_dataset

        decompose_graph_to_dict(graph_node_new)

        self.reset_env(int(graph_node_new.graph_index), plot=True)
                
    def create_next_graph(
        self,
        graph_node_new=None,
        graph_node_x_new=None,
        graph_edge_new=None,
        graph_cell_new=None,
        graph_cell_x_new=None,
        synatic=False,
    ):
        (
            graph_node_next,
            graph_node_x_next,
            graph_edge_next,
            graph_cell_next,
            graph_cell_x_next,
        ) = Data_Pool.datapreprocessing(
            graph_node_new,
            graph_node_x_new,
            graph_edge_new,
            graph_cell_new,
            graph_cell_x_new,
            dimless=self.params.dimless,
        )

        return (
            graph_node_next,
            graph_node_x_next,
            graph_edge_next,
            graph_cell_next,
            graph_cell_x_next,
        )


class GraphNodeDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
        """Optional node attr"""
        vp_on_node = torch.cat(
            (
                minibatch_data["velocity_on_node"][0],
                minibatch_data["pressure_on_node"][0],
            ),
            dim=1,
        )
        mesh_pos = minibatch_data["mesh_pos"][0].to(torch.float32)
        face_node = minibatch_data["face"][0].to(torch.long)
        cells_node = minibatch_data["cells_node"][0].to(torch.long)
        node_type = minibatch_data["node_type"][0]
        target_on_node = torch.cat(
            (
                minibatch_data["target|velocity_on_node"][0],
                minibatch_data["target|pressure_on_node"][0],
            ),
            dim=1,
        )
        pde_theta_node = minibatch_data["pde_theta_node"][0]
        neural_network_output_mask = minibatch_data["neural_network_output_mask"][0]
        uvp_dim = minibatch_data["uvp_dim"][0]

        graph_node = Data(
            x=vp_on_node,
            edge_index=face_node,
            face=cells_node.T,
            pde_theta_node=pde_theta_node,
            neural_network_output_mask=neural_network_output_mask,
            uvp_dim=uvp_dim,
            pos=mesh_pos,
            node_type=node_type,
            y=target_on_node,
            graph_index=torch.as_tensor([idx]),
        )

        return graph_node


class GraphNode_X_Dataset(InMemoryDataset):
    """This graph is undirected"""

    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
        """Optional node attr"""
        mesh_pos = minibatch_data["mesh_pos"][0].to(torch.float32)
        face_node_x = minibatch_data["face_node_x"][0].to(torch.long)
        A_node_to_node = minibatch_data["A_node_to_node"][0]
        A_node_to_node_x = minibatch_data["A_node_to_node_x"][0]
        A_cell_to_node = minibatch_data["A_cell_to_node"][0]
        B_node_to_node = minibatch_data["B_node_to_node"][0]
        B_node_to_node_x = minibatch_data["B_node_to_node_x"][0]
        B_cell_to_node = minibatch_data["B_cell_to_node"][0]
        
        graph_node_x = Data(
            edge_index=face_node_x,
            num_nodes=mesh_pos.shape[0],
            A_node_to_node = A_node_to_node,
            A_node_to_node_x = A_node_to_node_x,
            A_cell_to_node = A_cell_to_node,
            B_node_to_node = B_node_to_node,
            B_node_to_node_x = B_node_to_node_x,
            B_cell_to_node = B_cell_to_node,
            graph_index=torch.as_tensor([idx]),
        )
        
        # mesh_pos = minibatch_data["mesh_pos"][0].to(torch.float32)
        # R_inv_Q_t = minibatch_data["R_inv_Q_t"][0]
        # node_neigbors = minibatch_data["node_neigbors"][0].to(torch.long)
        # mask_node_neigbors_fil = minibatch_data["mask_node_neigbors_fil"][0]
        
        # graph_node_x = Data(
        #     num_nodes=mesh_pos.shape[0],
        #     R_inv_Q_t=R_inv_Q_t,
        #     mask_node_neigbors_fil=mask_node_neigbors_fil,
        #     face=node_neigbors.T,
        #     graph_index=torch.as_tensor([idx]),
        # )
        
        return graph_node_x


class GraphEdgeDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]

        # edge_attr
        face_length = minibatch_data["face_length"][0].to(torch.float32)
        face_types = minibatch_data["face_type"][0].to(torch.long)
        face_center_pos = minibatch_data["face_center_pos"][0].to(torch.float32)
        cells_face = minibatch_data["cells_face"][0].to(torch.long)

        graph_edge = Data(
            x=torch.cat((face_types, face_length), dim=1),
            face=cells_face.T,
            pos=face_center_pos,
            graph_index=torch.as_tensor([idx]),
        )

        return graph_edge


class GraphCellDataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
        # current_time_steps = torch.as_tensor([minibatch_data['time_steps']]).to(torch.long)

        # cell_attr
        edge_neighbour_cell = minibatch_data["neighbour_cell"][0].to(torch.long)
        cell_area = minibatch_data["cells_area"][0]
        centroid = minibatch_data["centroid"][0].to(torch.float32)
        cell_factor = minibatch_data["cell_factor"][0]
        cells_type = minibatch_data["cells_type"][0].to(torch.long)
        # unv = minibatch_data['unit_norm_v'][0] # unit norm vector
        cells_node_face_unv = minibatch_data["cells_node_face_unv"][
            0
        ]  # unit norm vector
        cells_node_surface_vector = minibatch_data["node_face_surface_vector"][0]

        cells_index = minibatch_data["cells_index"][0].to(torch.long)
        pde_theta_cell = minibatch_data["pde_theta_cell"][0]
        uvp_dim_cell = minibatch_data["uvp_dim_cell"][0]

        graph_cell = Data(
            edge_index=edge_neighbour_cell,
            cells_node_surface_vector=cells_node_surface_vector,
            cells_node_face_unv=cells_node_face_unv,
            cells_type=cells_type,
            cell_area=cell_area,
            pos=centroid,
            face=cells_index.T,
            graph_index=torch.as_tensor([idx]),
            pde_theta_cell=pde_theta_cell,
            cell_factor=cell_factor,
            uvp_dim_cell=uvp_dim_cell,
            mask_cell_interior=cells_type.view(-1),
        )

        return graph_cell


class GraphCell_X_Dataset(InMemoryDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    @property
    def pool(self):
        # 这里你可以根据需要从基类的pool中筛选出GraphNode的数据
        return self.base_dataset.pool

    def len(self):
        return len(self.pool)

    def get(self, idx):
        minibatch_data = self.pool[idx]
        """Optional node attr"""
        centroid = minibatch_data["centroid"][0].to(torch.float32)
        neighbour_cell_x = minibatch_data["neighbour_cell_x"][0].to(torch.long)

        # A_inv_cell_to_cell_x = minibatch_data["A_inv_cell_to_cell_x"][0]
        # B_cell_to_cell_x = minibatch_data["B_cell_to_cell_x"][0]
        # B_node_to_cell = minibatch_data["B_node_to_cell"][0]

        graph_cell_x = Data(
            # edge_index=neighbour_cell_x,
            num_nodes=centroid.shape[0],
            # A_inv_cell_to_cell_x=A_inv_cell_to_cell_x,
            # B_cell_to_cell_x=B_cell_to_cell_x,
            # B_node_to_cell=B_node_to_cell,
            graph_index=torch.as_tensor([idx]),
        )

        return graph_cell_x


# 在你的代码文件的开头添加以下类定义
class SharedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.epoch = 0
        self.specific_indices = None  # 用于存储特定的索引

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.specific_indices is not None:
            return iter(self.specific_indices)
        return iter(torch.randperm(len(self.data_source), generator=g).tolist())

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = int(datetime.datetime.now().timestamp())

    def set_specific_indices(self, indices):
        self.specific_indices = indices


# 修改CustomDataLoader以使用SharedSampler
class CustomDataLoader:
    def __init__(
        self,
        graph_node_dataset,
        graph_node_x_dataset,
        graph_edge_dataset,
        graph_cell_dataset,
        graph_cell_x_dataset,
        batch_size,
        sampler,
        num_workers=4,
        pin_memory=True,
    ):
        # 保存输入参数到实例变量
        self.graph_node_dataset = graph_node_dataset
        self.graph_node_x_dataset = graph_node_x_dataset
        self.graph_edge_dataset = graph_edge_dataset
        self.graph_cell_dataset = graph_cell_dataset
        self.graph_cell_x_dataset = graph_cell_x_dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # 初始化DataLoaders
        self.loader_A = torch_geometric_DataLoader(
            graph_node_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_B = torch_geometric_DataLoader(
            graph_node_x_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_C = torch_geometric_DataLoader(
            graph_edge_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_D = torch_geometric_DataLoader(
            graph_cell_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.loader_E = torch_geometric_DataLoader(
            graph_cell_x_dataset,
            batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self):
        return zip(
            self.loader_A, self.loader_B, self.loader_C, self.loader_D, self.loader_E
        )

    def __len__(self):
        return min(
            len(self.loader_A),
            len(self.loader_B),
            len(self.loader_C),
            len(self.loader_D),
            len(self.loader_E),
        )

    def get_specific_data(self, indices):
        # 设置Sampler的特定索引
        self.sampler.set_specific_indices(indices)

        # 重新创建DataLoaders来使用更新的Sampler
        self.loader_A = torch_geometric_DataLoader(
            self.graph_node_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_B = torch_geometric_DataLoader(
            self.graph_node_x_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_C = torch_geometric_DataLoader(
            self.graph_edge_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_D = torch_geometric_DataLoader(
            self.graph_cell_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.loader_E = torch_geometric_DataLoader(
            self.graph_cell_x_dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        graph_node, graph_node_x, graph_edge, graph_cell, graph_cell_x = next(
            iter(self)
        )

        minibatch_data = self.graph_node_dataset.pool[indices[0]]

        origin_mesh_path = "".join(
            [chr(int(f)) for f in minibatch_data["origin_mesh_path"][0, :, 0].numpy()]
        )

        flow_type = minibatch_data["flow_type"]
        if ("cavity" in flow_type) or ("possion" in flow_type):
            has_boundary = False
        else:
            has_boundary = True

        return (
            graph_node,
            graph_node_x,
            graph_edge,
            graph_cell,
            graph_cell_x,
            has_boundary,
            origin_mesh_path,
        )


# 修改DatasetFactory以创建SharedSampler并将其传递给CustomDataLoader
class DatasetFactory:
    def __init__(
        self,
        params=None,
        is_training=True,
        state_save_dir=None,
        split=None,
        dataset_dir=None,
        spec_u_rho_mu_comb=None,
        inflow_bc_type=None,
        device=None,
    ):
        self.base_dataset = Data_Pool(
            params=params,
            is_training=is_training,
            device=device,
            state_save_dir=state_save_dir,
        )

        self.dataset_size, self.params = self.base_dataset.load_mesh_to_cpu(
            split=split,
            dataset_dir=dataset_dir,
            inflow_bc_type=inflow_bc_type,
            spec_u_rho_mu_comb=spec_u_rho_mu_comb,
        )

    def create_datasets(self, batch_size=100, num_workers=4, pin_memory=True):
        graph_node_dataset = GraphNodeDataset(base_dataset=self.base_dataset)
        graph_node_x_dataset = GraphNode_X_Dataset(base_dataset=self.base_dataset)
        graph_edge_dataset = GraphEdgeDataset(base_dataset=self.base_dataset)
        graph_cell_dataset = GraphCellDataset(base_dataset=self.base_dataset)
        graph_cell_x_dataset = GraphCell_X_Dataset(base_dataset=self.base_dataset)

        # 创建SharedSampler并将其传递给CustomDataLoader

        sampler = SharedSampler(graph_node_dataset)

        loader = CustomDataLoader(
            graph_node_dataset,
            graph_node_x_dataset,
            graph_edge_dataset,
            graph_cell_dataset,
            graph_cell_x_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return self.base_dataset, loader, sampler


def make_data_loader(
    data_pool=None,
    batch_size=None,
    num_workers=8,
    prefetch_factor=10,
    pin_memory=False,
    shuffle=False,
):
    def collate_fn(batch_list):
        assert type(batch_list) == list, f"Error"
        batch_size = len(batch_list)
        print("batch_size: %d" % batch_size)
        return list(batch_list)

    """
    Create a BRACS data loader
    """
    dataset = data_pool
    dataloader = torch_geometric_DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return dataloader


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join("/home/litianyu/mycode/repos-py/FVM/my_FVNN/"))
    from utils.get_param import get_hyperparam
    from utils import get_param

    import time

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.82, 0)
    params = get_param.params()[0]
    split = "train"
    start = time.time()
    datasets = Data_Pool(params=params, is_training=True, device=device)
    datasets.load_mesh_to_cpu()
    datasets._set_status(is_training=False)
    noise_std = 2e-2
    end = time.time()
    print("traj has been loaded time consuming:{0}".format(end - start))

    def loss_function(x):
        return torch.pow(x, 2)

    avg_time_2 = []
    ds_loader = make_data_loader(
        datasets, batch_size=35, num_workers=8, prefetch_factor=10, pin_memory=False
    )
    print("loading whole dataset to cpu")
    last_time = time.time()
    # transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])
    loader = torch_geometric_DataLoader(
        datasets, batch_size=50, shuffle=True, num_workers=4, prefetch_factor=20
    )

    # training loop
    last_time = time.time()
    avg_time_1 = []
    re_num_list = []

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for epoch in range(10000):
        total_time_steps = torch.zeros(601)
        datasets._set_epoch(epoch % 300)
        start = time.time()
        for iner_step in range(params.iner_step - 1, -1, -1):
            for batch_index, graph_list in enumerate(loader):
                graph_cell = graph_list[2]
                # re_num,mean_u = cal_relonyds_number(graph_list[0],graph_list[1])
                # re_num_list.append(re_num)
                # print("Relonyds number:",re_num.numpy())
                time_steps = graph_cell.timesteps
                for i in range(len(time_steps)):
                    total_time_steps[time_steps[i]] += 1
                # print("time_steps",time_steps)

                const_time = time.time() - last_time
                avg_time_1.append(const_time)
                # print('time consuming:', const_time)
                last_time = time.time()
                datasets.payback(graph_cell)
        """ >>>         plot ghosted boundary cell center pos           >>>"""
        plt.bar(np.arange(total_time_steps.shape[0]), total_time_steps.numpy())
        plt.savefig("test.png")
        plt.clf()
        print("epoch:", epoch)
        """ <<<         plot ghosted boundary cell center pos           <<<"""
    re_num_list = np.array(re_num_list)
    re_num_list = np.sort(re_num_list, axis=0)
    y = np.zeros(re_num_list.shape[0])
    plt.scatter(re_num_list, y, edgecolors="red")
    plt.ylim(-1, 1)
    plt.show()
    plt.savefig("Relonyds_number.png")
    print("MAX Relonyds number:", np.max(re_num_list))
    print("avg_time_2:", np.average(np.asarray(avg_time_1)))
    print("done loading dataset")
