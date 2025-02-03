import sys
import os

file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)
import numpy as np
import concurrent.futures
import multiprocessing
import threading
import torch
import re
from Extract_mesh.parse_to_h5 import (
    extract_mesh_state,
    NodeType,
)
from Extract_mesh.parse_base import Basemanager
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import math
from Utils.utilities import filter_adj
from Post_process.to_vtk import write_vtp_file
import random
from contextlib import ExitStack
from math import ceil
import subprocess
import json
import vtk
from torch_scatter import scatter
import sys
import pyvista as pv

# 将输出缓冲区设置为0
sys.stdout.flush()


def string_to_floats(s):
    """将字符串转换为一组浮点数"""
    return np.asarray([float(ord(c)) for c in s])


def floats_to_string(floats):
    """将一组浮点数转换为字符串"""
    return "".join([chr(int(f)) for f in floats])


class TecplotMesh(Basemanager):
    """
    Tecplot .dat file is only supported with Tobias`s airfoil dataset ,No more data file supported
    """

    def __init__(
        self,
        mesh_file,
        data_file,
        file_dir=None,
        case_name=None,
        path=None,
    ):
        self.mesh_info = {
            "mesh_pos": None,
            "cells_index": None,
            "U": None,
            "V": None,
            "P": None,
        }
        self.boundary_mesh_info = {"mesh_pos": None, "face_node": None}
        self._parse_file_test(mesh_file)
        self.node_type = []
        self.path = path
        self.case_name = case_name
        self.file_dir = file_dir
        
    def read_title_and_variables(self, file_handle, input_line):
        variables_info = []
        key, value = input_line.strip().split("=")
        variables_info.append(value.strip().strip('"'))

        while True:
            line = next(file_handle).strip()
            if line.startswith("DATASETAUXDATA"):
                continue
            elif line.startswith("ZONE"):
                break
            else:
                variables_info.append(line.strip().strip('"'))

        return variables_info, line, file_handle

    def read_zone(self, file_handle, input_line):
        zone_info = {}

        # 清除zone_info并处理ZONE行
        zone_info.clear()
        Zone_name = input_line.split("=")[1].strip().strip('"')
        zone_info["ZONE"] = Zone_name

        while True:
            line = next(file_handle).strip()
            if line.startswith("DT"):
                break
            else:
                kv_pairs = line.split(",")
                for kv_pair in kv_pairs:
                    key, value = kv_pair.split("=")
                    zone_info[key.strip()] = value.strip()

        return zone_info, line, file_handle

    def read_interior_mesh_pos_and_index(self, file_handle, zone_info, variables_info):
        # 获取节点和单元数量
        num_nodes = int(zone_info.get("Nodes", 0))
        num_elements = int(zone_info.get("Elements", 0))

        # 获取数据点
        data_groups = []
        data_points = []
        while True:
            current_data_line = next(file_handle).strip().split()
            for data in current_data_line:
                data_points.append(float(data))

            if len(data_points) >= (num_nodes * len(variables_info)):
                break

        data_groups = np.split(np.asarray(data_points), len(variables_info))

        for data_index, variables in enumerate(variables_info):
            self.mesh_info[variables] = data_groups[data_index]

        # 获取单元
        while True:
            try:
                current_line = next(file_handle)
                # ... 处理current_line的逻辑 ...
                if current_line.startswith("ZONE"):
                    break
                if current_line.startswith("#"):
                    key = "_".join(current_line.strip().split("#")[1].strip().split())
                    setattr(self, key, [])
                    current_attr = getattr(self, key)
                    continue
                current_attr.extend(map(int, current_line.strip().split()))
            except StopIteration:
                break

        # 将数据点转换为NumPy数组并存储
        self.mesh_info["mesh_pos"] = np.stack(
            (self.mesh_info["X"], self.mesh_info["Y"]), axis=1
        )
        self.mesh_info["face_node"] = np.array(self.face_nodes).reshape(-1, 2) - 1
        self.mesh_info["face_center"] = torch.from_numpy(
            (
                self.mesh_info["mesh_pos"][self.mesh_info["face_node"][:, 0]]
                + self.mesh_info["mesh_pos"][self.mesh_info["face_node"][:, 1]]
            )
            / 2.0
        )

        # self.plot_state(self.mesh_info["mesh_pos"],self.mesh_info["face_node"])

        # tecplot user manul said boundary face`s outside is 0
        self.left_elements = np.array(self.left_elements)
        self.right_elements = np.array(self.right_elements)
        self.neighbour_cell = (
            np.stack((self.left_elements, self.right_elements), axis=1) - 1
        )  # outside become -1

        face_index = torch.from_numpy(
            np.arange(self.mesh_info["face_node"].shape[0])
        ).repeat(2)
        two_way_neighbour_cell = torch.from_numpy(
            np.concatenate(
                (self.neighbour_cell[:, 0], self.neighbour_cell[:, 1]), axis=0
            )
        )

        self.mesh_info["cells_node"] = []
        self.mesh_info["cells_face"] = []
        self.mesh_info["cells_index"] = []
        for cells_index in range(two_way_neighbour_cell.max() + 1):
            current_mask_cells_face = two_way_neighbour_cell == cells_index
            current_cells_index = torch.full_like(
                face_index[current_mask_cells_face], cells_index
            )
            current_cells_face = self.ensure_counterclockwise(
                (face_index[current_mask_cells_face].unsqueeze(0)).numpy(),
                mesh_pos=self.mesh_info["face_center"].numpy(),
            )
            current_cells_node = self.ensure_counterclockwise(
                np.unique(self.mesh_info["face_node"][current_cells_face]).reshape(
                    1, -1
                ),
                mesh_pos=self.mesh_info["mesh_pos"],
            )
            self.mesh_info["cells_node"].extend(torch.from_numpy(current_cells_node))
            self.mesh_info["cells_face"].extend(torch.from_numpy(current_cells_face))
            self.mesh_info["cells_index"].extend(current_cells_index)

        self.mesh_info["cells_node"] = torch.stack(self.mesh_info["cells_node"])
        self.mesh_info["cells_face"] = torch.stack(self.mesh_info["cells_face"])
        self.mesh_info["cells_index"] = torch.stack(self.mesh_info["cells_index"])
        self.mesh_info["cells_face_node"] = torch.from_numpy(
            self.mesh_info["face_node"]
        )[self.mesh_info["cells_face"]]

        # centroid = scatter(self.mesh_info["face_center"][self.mesh_info["cells_face"]],self.mesh_info["cells_index"],dim=0,reduce="mean")
        # self.plot_state(self.mesh_info["mesh_pos"],self.mesh_info["face_node"],self.mesh_info["face_center"][self.mesh_info["cells_face"]],centroid=centroid)

        left_cell = scatter(
            self.mesh_info["cells_index"],
            self.mesh_info["cells_face"],
            dim=0,
            reduce="max",
        )
        right_cell = scatter(
            self.mesh_info["cells_index"],
            self.mesh_info["cells_face"],
            dim=0,
            reduce="min",
        )

        valid_neighbour_cell = torch.stack((left_cell, right_cell), dim=1)

        # self.plot_state(centroid,valid_neighbour_cell,centroid=centroid)

        # form self loop at boundary face
        mask_left = self.left_elements == 0
        mask_right = self.right_elements == 0
        self.left_elements[mask_left] = self.right_elements[mask_left]
        self.right_elements[mask_right] = self.left_elements[mask_right]
        self.neighbour_cell = (
            np.stack((self.left_elements, self.right_elements), axis=1) - 1
        )

        neighbour_cell, _ = torch.sort(torch.from_numpy(self.neighbour_cell), dim=1)
        valid_neighbour_cell, _ = torch.sort(valid_neighbour_cell, dim=1)

        valid_mask = valid_neighbour_cell == neighbour_cell

        if valid_mask.all():
            print("good neighbour cell")
        else:
            raise ValueError("bad neighbour cell")

        return file_handle, current_line

    def read_boundary_mesh_pos_and_index(self, file_handle, zone_info, variables_info):
        # 获取节点和单元数量
        num_nodes = int(zone_info.get("Nodes", 0))
        num_elements = int(zone_info.get("Elements", 0))

        # 获取数据点
        data_groups = []
        data_points = []
        while True:
            current_data_line = next(file_handle).strip().split()
            for data in current_data_line:
                data_points.append(float(data))
            if len(data_points) >= (num_nodes * len(variables_info)):
                break

        data_groups = np.split(np.asarray(data_points), len(variables_info))

        for data_index, variables in enumerate(variables_info):
            self.boundary_mesh_info[variables] = data_groups[data_index]

        # 获取单元
        face_node = []
        while True:
            iter_flag = False
            try:
                current_line = next(file_handle)
                # ... 处理current_line的逻辑 ...
                if current_line.startswith("ZONE"):
                    break
                elif current_line.startswith("#"):
                    key = "_".join(current_line.strip().split("#")[1].strip().split())
                    setattr(self, key, [])
                    current_attr = getattr(self, key)
                    continue
                else:
                    try:
                        current_attr.extend(map(int, current_line.strip().split()))
                    except:
                        face_node.extend(map(int, current_line.strip().split()))
            except StopIteration:
                iter_flag = True
                break
        face_node = np.array(face_node) - 1

        try:
            face_node += self.boundary_mesh_info["face_node"].max()
            self.boundary_mesh_info["face_node"] = np.concatenate(
                (self.boundary_mesh_info["face_node"], face_node), axis=0
            )
        except:
            self.boundary_mesh_info["face_node"] = face_node

        # 将数据点转换为NumPy数组并存储
        try:
            self.boundary_mesh_info["mesh_pos"] = np.concatenate(
                (
                    self.boundary_mesh_info["mesh_pos"],
                    np.stack(
                        (self.boundary_mesh_info["X"], self.boundary_mesh_info["Y"]),
                        axis=1,
                    ),
                ),
                axis=0,
            )
        except:
            self.boundary_mesh_info["mesh_pos"] = np.stack(
                (self.boundary_mesh_info["X"], self.boundary_mesh_info["Y"]), axis=1
            )

        return file_handle, current_line, iter_flag

    def _parse_file_test(self, file_path):
        # 标记数据的开始
        data_section = False
        iter_flag = True
        self.Zones = {}

        with open(file_path, "r") as file:
            while True:
                if iter_flag:
                    try:
                        line = next(file).strip()
                        if line is None:
                            break
                    except StopIteration:
                        break
                if line.startswith("VARIABLES"):
                    variables_info, line, file = self.read_title_and_variables(
                        file, line
                    )
                    iter_flag = False
                elif line.startswith("ZONE"):
                    zone_info, line, file = self.read_zone(file, line)
                    if zone_info["ZONETYPE"].lower() in [
                        "fepolygon",
                        "fetriangle",
                        "fequadrilateral",
                        "fetetrahedron",
                        "febrick",
                        "fepolygon",
                        "fepolyhedron",
                    ]:
                        file, line = self.read_interior_mesh_pos_and_index(
                            file, zone_info, variables_info
                        )
                    else:
                        file, line, iter_flag = self.read_boundary_mesh_pos_and_index(
                            file, zone_info, variables_info
                        )

        self.mesh_pos = self.mesh_info["mesh_pos"]

    def _parse_file(self, file_path):
        # 标记数据的开始
        data_section = False

        # 解析文件
        with open(file_path, "r") as file:
            zone_info = {}
            variables_info = []
            variables_flag = False
            for line_index, line in enumerate(file):
                line = line.strip()

                if not line.startswith("ZONE") and line_index > 0:
                    variables_flag = True
                else:
                    variables_flag = False

                if line.startswith("DATASETAUXDATA"):
                    variables_flag = False

                if line.startswith("VARIABLES") or (variables_flag):
                    for item in line.split(","):
                        if "=" in item:
                            key, value = item.split("=")
                            variables_info.append(value.strip().strip('"'))
                        else:
                            variables_info.append(line.strip().strip('"'))
                            break

                # 解析区域信息
                if line.startswith("ZONE"):
                    # 清除zone_info并处理ZONE行
                    zone_info.clear()
                    if not line.startswith("DATAPACKING"):
                        for item in line.split(","):
                            if "=" in item:
                                key, value = item.split("=")
                                zone_info[key.strip()] = value.strip()
                    while True:
                        nextline = next(file).strip()
                        if not nextline.startswith("DATAPACKING"):
                            for item in nextline.split(","):
                                if "=" in item:
                                    key, value = item.split("=")
                                    zone_info[key.strip()] = value.strip()

                        else:
                            # 继续处理接下来的ZONE属性行
                            while True:
                                next_line = next(file).strip()
                                if "DT" in next_line:
                                    break  # 到达数据区域的开始
                                for item in next_line.split(","):
                                    if "=" in item:
                                        key, value = item.split("=")
                                        zone_info[key.strip()] = value.strip()
                            break

                    zone_info["num_variables"] = len(variables_info)

                    # 获取节点和单元数量
                    num_nodes = int(zone_info.get("Nodes", 0))
                    num_elements = int(zone_info.get("Elements", 0))

                    # 获取数据点
                    data_groups = []
                    data_points = []
                    while True:
                        current_data_line = next(file).strip().split()
                        for data in current_data_line:
                            data_points.append(float(data))

                        if len(data_points) >= (num_nodes * zone_info["num_variables"]):
                            break

                    data_groups = np.split(
                        np.asarray(data_points), zone_info["num_variables"]
                    )

                    for data_index, variables in enumerate(variables_info):
                        self.mesh_info[variables] = data_groups[data_index]

                    # 获取单元
                    cells = []
                    # try:
                    while True:
                        current_line = next(file)
                        if current_line.startswith("ZONE"):
                            break
                        if current_line.startswith("#"):
                            key = "_".join(
                                current_line.strip().split("#")[1].strip().split()
                            )
                            setattr(self, key, [])
                            current_attr = getattr(self, key)
                            continue
                        current_attr.extend(map(int, current_line.strip().split()))
                    # except:pass

                    # 将数据点转换为NumPy数组并存储

                    self.mesh_info["mesh_pos"] = np.stack(
                        (self.mesh_info["X"], self.mesh_info["Y"]), axis=1
                    )
                    self.face_node = np.array(self.face_nodes).reshape(-1, 2) - 1

                    self.face_center = torch.from_numpy(
                        (
                            self.mesh_info["mesh_pos"][self.face_node[:, 0]]
                            + self.mesh_info["mesh_pos"][self.face_node[:, 1]]
                        )
                        / 2.0
                    )

                    # self.plot_state(self.mesh_info["mesh_pos"],self.face_node)

                    # tecplot user manul said boundary face`s outside is 0
                    self.left_elements = np.array(self.left_elements)
                    self.right_elements = np.array(self.right_elements)
                    self.neighbour_cell = (
                        np.stack((self.left_elements, self.right_elements), axis=1) - 1
                    )  # outside become -1

                    face_index = torch.from_numpy(
                        np.arange(self.face_node.shape[0])
                    ).repeat(2)
                    two_way_neighbour_cell = torch.from_numpy(
                        np.concatenate(
                            (self.neighbour_cell[:, 0], self.neighbour_cell[:, 1]),
                            axis=0,
                        )
                    )

                    self.cells_node = []
                    self.cells_face = []
                    self.cells_index = []
                    for cells_index in range(two_way_neighbour_cell.max() + 1):
                        current_mask_cells_face = two_way_neighbour_cell == cells_index
                        current_cells_index = torch.full_like(
                            face_index[current_mask_cells_face], cells_index
                        )

                        current_cells_face = self.ensure_counterclockwise(
                            (face_index[current_mask_cells_face].unsqueeze(0)).numpy(),
                            mesh_pos=self.face_center.numpy(),
                        )

                        current_cells_node = self.ensure_counterclockwise(
                            np.unique(self.face_node[current_cells_face]).reshape(
                                1, -1
                            ),
                            mesh_pos=self.mesh_info["mesh_pos"],
                        )

                        self.cells_node.extend(torch.from_numpy(current_cells_node))
                        self.cells_face.extend(torch.from_numpy(current_cells_face))
                        self.cells_index.extend(current_cells_index)

                    self.cells_node = torch.stack(self.cells_node)
                    self.cells_face = torch.stack(self.cells_face)
                    self.cells_index = torch.stack(self.cells_index)

                    # centroid = scatter(self.face_center[self.cells_face],self.cells_index,dim=0,reduce="mean")
                    # self.plot_state(self.mesh_info["mesh_pos"],self.face_node,self.face_center[self.cells_face],centroid=centroid)

                    left_cell = scatter(
                        self.cells_index, self.cells_face, dim=0, reduce="max"
                    )
                    right_cell = scatter(
                        self.cells_index, self.cells_face, dim=0, reduce="min"
                    )

                    valid_neighbour_cell = torch.stack((left_cell, right_cell), dim=1)

                    # self.plot_state(centroid,valid_neighbour_cell,centroid=centroid)

                    # form self loop at boundary face
                    mask_left = self.left_elements == 0
                    mask_right = self.right_elements == 0
                    self.left_elements[mask_left] = self.right_elements[mask_left]
                    self.right_elements[mask_right] = self.left_elements[mask_right]
                    self.neighbour_cell = (
                        np.stack((self.left_elements, self.right_elements), axis=1) - 1
                    )

                    neighbour_cell, _ = torch.sort(
                        torch.from_numpy(self.neighbour_cell), dim=1
                    )
                    valid_neighbour_cell, _ = torch.sort(valid_neighbour_cell, dim=1)

                    valid_mask = valid_neighbour_cell == neighbour_cell

                    if valid_mask.all():
                        print("good neighbour cell")
                        break
                    else:
                        raise ValueError("bad neighbour cell")

                    # self.plot_state(centroid,neighbour_cell,centroid=centroid)

        self.mesh_pos = self.mesh_info["mesh_pos"]

    def extract_pipe_flow_boundary(self, mesh_boundary_pos=None):
        
        self.node_type = np.empty((self.mesh_pos.shape[0], 1))
        mesh_pos = self.mesh_pos.astype(dtype=np.float32) + np.abs(
            self.mesh_pos.astype(dtype=np.float32).min(axis=0)
        )
        mesh_boundary_pos = mesh_boundary_pos.astype(dtype=np.float32) + np.abs(
            self.mesh_pos.astype(dtype=np.float32).min(axis=0)
        )
        self.surf_mask = np.full((mesh_pos.shape[0],1), False)
        
        topwall = np.max(mesh_pos[:, 1])
        bottomwall = np.min(mesh_pos[:, 1])
        outlet = np.max(mesh_pos[:, 0])
        inlet = np.min(mesh_pos[:, 0])

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        for i in range(mesh_pos.shape[0]):
            current_coord = mesh_pos[i]
            if (
                (self.is_equal(current_coord[0], inlet))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topwall - (1e-12)))
            ):
                self.node_type[i] = NodeType.INFLOW
                self.INFLOW += 1
            elif (current_coord[1] >= topwall) or (current_coord[1] <= bottomwall):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1
            elif (
                (self.is_equal(current_coord[0], outlet))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topwall - (1e-12)))
            ):
                self.node_type[i] = NodeType.OUTFLOW
                self.OUTFLOW += 1
                OUTFLOW_list.append(current_coord)
            elif (
                (
                    np.logical_and(
                        (current_coord == mesh_boundary_pos)[:, 0],
                        (current_coord == mesh_boundary_pos)[:, 1],
                    ).any()
                )
                and (current_coord[0] > 0)
                and (current_coord[0] < (outlet - (1e-12)))
                and (current_coord[1] > 0)
                and (current_coord[1] < (topwall - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                self.surf_mask[i] = True
                self.WALL_BOUNDARY += 1
                self.OBSTACLE += 1
                OBSTACLE_list.append(current_coord)
            else:
                self.node_type[i] = NodeType.NORMAL
                self.NORMAL += 1

        print(
            "After readed data in file has NODE TYPE: NORMAL: {0} OBSTACLE: {1} AIRFOIL: {2} HANDLE: {3} INFLOW: {4} OUTFLOW: {5} WALL_BOUNDARY: {6} SIZE: {7}".format(
                self.NORMAL,
                self.OBSTACLE,
                0,
                0,
                self.INFLOW,
                self.OUTFLOW,
                self.WALL_BOUNDARY,
                0,
            )
        )

    def extract_mesh(self, mesh_only=True):
        self.INFLOW = 0
        self.WALL_BOUNDARY = 0
        self.OUTFLOW = 0
        self.OBSTACLE = 0
        self.NORMAL = 0

        if ("cylinder" in self.path["case_name"]):
            self.path["flow_type"] = "pipe_flow"
            self.extract_pipe_flow_boundary(
                mesh_boundary_pos=self.boundary_mesh_info["mesh_pos"]
            )
            
        self.save_to_vtu(
            mesh=self.mesh_info, 
            payload={"node|node_type": self.node_type}, 
            file_name=f"{self.file_dir}/node_type_with_mesh.vtu",
        )
        # self.plot_state(self.mesh_info["mesh_pos"],self.mesh_info["face_node"],node_type = self.node_type)

        if mesh_only:
            mesh = {
                "node|pos": torch.from_numpy(self.mesh_pos),
                "node|surf_mask": torch.from_numpy(self.surf_mask).bool().squeeze(),
                "node|node_type": torch.from_numpy(self.node_type).long().squeeze(),
                "face|face_node": torch.from_numpy(self.mesh_info["face_node"]).long().transpose(0, 1),
                "cells_node": self.mesh_info["cells_node"].long().squeeze(),
                "cells_index": self.mesh_info["cells_index"].long().squeeze(),
                "cells_face": self.mesh_info["cells_face"].long().squeeze(),
            }
        else:
            # velocity = torch.index_select(
            #     torch.from_numpy(self.data_velocity),
            #     1,
            #     torch.from_numpy(self.earrange_index).to(torch.long),
            # )
            # pressure = torch.index_select(
            #     torch.from_numpy(self.data_pressure),
            #     1,
            #     torch.from_numpy(self.rearrange_index).to(torch.long),
            # )
            # mesh = {
            #     "mesh_pos": torch.from_numpy(self.mesh_pos)
            #     .to(torch.float64)
            #     .unsqueeze(0)
            #     .repeat(1, 1, 1),
            #     "boundary": torch.from_numpy(self.mesh_boundary_index)
            #     .to(torch.long)
            #     .unsqueeze(0)
            #     .repeat(1, 1, 1),
            #     "cells_node": torch.from_numpy(self.cells_node)
            #     .to(torch.long)
            #     .unsqueeze(0)
            #     .repeat(1, 1, 1),
            #     "cells_index": torch.from_numpy(self.cells_index)
            #     .to(torch.long)
            #     .unsqueeze(0)
            #     .repeat(1, 1, 1),
            #     "cells_face_node": torch.from_numpy(self.cells_face_node)
            #     .to(torch.long)
            #     .unsqueeze(0)
            #     .repeat(1, 1, 1),
            #     "node_type": torch.from_numpy(self.node_type)
            #     .to(torch.long)
            #     .unsqueeze(0)
            #     .repeat(1, 1, 1),
            #     "velocity": velocity[0:600].astype(np.float64),
            #     "pressure": pressure[0:600].astype(np.float64),
            # }
            pass

        # There`s face_center_pos, centroid, face_type, neighbour_cell, face_node_x need to be resolved
        h5_dataset = extract_mesh_state(
            mesh,
            path=self.path,
        )

        return h5_dataset


# Define the processing function
def process_file(plot, file_path, path, queue):
    
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    case_name = os.path.basename(file_dir)
    path["file_dir"] = file_dir
    path["case_name"] = case_name
    path["file_name"] = file_name

    # start convert func
    if file_path.endswith(".dat"):
        data = TecplotMesh(
            mesh_file=file_path,
            data_file=None,
            file_dir=file_dir,
            case_name=case_name,
            path=path,
        )

    else:
        return None

    h5_dataset = data.extract_mesh(mesh_only=path["mesh_only"])

    # Put the results in the queue
    queue.put((h5_dataset, case_name, file_dir))


# Writer process function
def writer_process(queue, path):

    while True:

        # Get data from queue
        h5_data, case_name, file_dir = queue.get()
        
        # Break if None is received (sentinel value)
        if h5_data is None:
            break
        
        os.makedirs(file_dir, exist_ok=True)
        h5_writer = h5py.File(f"{file_dir}/{case_name}.h5", "w")

        # Write dataset key value
        group = h5_writer.create_group(case_name)
        for key, value in h5_data.items():
            if key in group:
                del group[key]
            group.create_dataset(key, data=value)

        print(f"{case_name} mesh has been writed")

    # 关闭所有的writer
    h5_writer.close()


if __name__ == "__main__":
    # for debugging

    debug_file_path = None
    # debug_file_path = "datasets/cylinder_flow/cylinder_flow_poly_Re=1-10/cylinder_poly.dat"

    case = 0  # 0 stands for 980/PM9A1
    if case == 0:
        path = {
            "simulator": "StarCCM+",
            "tecplot_dataset_path": "datasets/cylinder_flow",
            "mesh_only": True,
        }

    # stastic total number of data samples
    total_samples = 0
    file_paths = []
    for subdir, _, files in os.walk(path["tecplot_dataset_path"]):
        for data_name in files:
            if data_name.endswith(".mphtxt") or data_name.endswith(".dat"):
                file_paths.append(os.path.join(subdir, data_name))

    # 统计选中的文件总数
    total_samples = len(file_paths)
    print("total samples: ", total_samples)

    if debug_file_path is not None:
        multi_process = 1
    elif total_samples < multiprocessing.cpu_count():
        multi_process = total_samples
    else:
        multi_process = multiprocessing.cpu_count()

    # Start to convert data using multiprocessing
    global_data_index = 0
    with multiprocessing.Pool(multi_process) as pool:
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        # Start writer process
        writer_proc = multiprocessing.Process(target=writer_process, args=(queue, path))
        writer_proc.start()

        if debug_file_path is not None:
            # for debuging
            results = [
                pool.apply_async(
                    process_file,
                    args=(
                        0,
                        debug_file_path,
                        path,
                        queue,
                    ),
                )
            ]
        else:
            # Process files in parallel
            results = [
                pool.apply_async(
                    process_file,
                    args=(
                        file_index,
                        file_path,
                        path,
                        queue,
                    ),
                )
                for file_index, file_path in enumerate(file_paths)
            ]

        # Wait for all processing processes to finish
        for res in results:
            res.get()

        # Send sentinel value to terminate writer process
        queue.put((None, None, None))
        writer_proc.join()

    print("done")
