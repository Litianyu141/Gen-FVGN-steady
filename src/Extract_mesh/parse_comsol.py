import numpy as np
import concurrent.futures
import multiprocessing
import threading
import torch
import re
from parse_tfrecord_refactor import (
    extract_mesh_state,
    NodeType,
    write_dict_info_to_json,
    write_tfrecord_one_with_writer,
)
import tensorflow as tf
import os
import matplotlib

matplotlib.use("agg")
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation
from natsort import natsorted
import h5py
import math
from random import shuffle
import random
from contextlib import ExitStack
from math import ceil
import subprocess
import itertools
import vtk
from torch_scatter import scatter
import sys

# 将输出缓冲区设置为0
sys.stdout.flush()

# Initialize lock for file writing synchronization
lock = threading.Lock()


def build_cell(cur_cell):
    if len(cur_cell) == 3:
        v_cell = vtk.vtkTriangle()
        [
            v_cell.GetPointIds().SetId(i, point_id)
            for i, point_id in zip(range(3), cur_cell)
        ]
        return v_cell
    elif len(cur_cell) == 4:
        v_cell = vtk.vtkQuad()
        [
            v_cell.GetPointIds().SetId(i, point_id)
            for i, point_id in zip(range(4), cur_cell)
        ]
        return v_cell


def get_vtk_mesh(points, cells, cells_index, point_attrib=None, is_2d: bool = True):
    v_points = vtk.vtkPoints()
    v_cells = vtk.vtkCellArray()

    if is_2d:
        [v_points.InsertNextPoint(*point, 0.0) for point in points]
    else:
        [v_points.InsertNextPoint(*point) for point in points]

    cur: int = 0
    cur_cell = []

    for i in range(len(cells_index)):
        if cur != cells_index[i]:
            v_cell = build_cell(cur_cell)
            v_cells.InsertNextCell(v_cell)
            cur = cells_index[i]
            cur_cell.clear()

        cur_cell.append(cells[i])

    v_cells.InsertNextCell(build_cell(cur_cell))

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(v_points)
    mesh.SetPolys(v_cells)

    # 假设 scalar_data 是一个列表，其中每个元素是与点相关联的标量值
    scalar_data = vtk.vtkLongArray()
    scalar_data.SetName("node_type")  # 为标量数据指定名称

    for scalar_value in point_attrib:
        scalar_data.InsertNextValue(scalar_value)

    mesh.GetPointData().SetScalars(scalar_data)

    return mesh


class MeshPlot:
    def __init__(self):
        # 创建一个VTK Renderer 和 RenderWindow
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # 创建一个VTK RenderWindowInteractor
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

        # 设置交互风格为 Trackball
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.render_window_interactor.SetInteractorStyle(style)

        self.window_to_image_filter = vtk.vtkWindowToImageFilter()
        self.window_to_image_filter.SetInput(self.render_window)
        self.png_writer = vtk.vtkPNGWriter()

        self.mesh = None

    def save_img(self, file_with_path):
        self.window_to_image_filter.Update()
        self.png_writer.SetFileName(file_with_path)
        self.png_writer.SetInputConnection(self.window_to_image_filter.GetOutputPort())
        self.png_writer.Write()

    def set_data(self, points, cells, cell_index, point_attrib, is_2d: bool = True):
        self.mesh = get_vtk_mesh(points, cells, cell_index, point_attrib, is_2d)
        self.layer_set()

    def layer_set(self):
        # 网格图层
        mapper_mesh = vtk.vtkPolyDataMapper()
        mapper_mesh.SetInputData(self.mesh)
        mapper_mesh.ScalarVisibilityOff()

        actor_mesh = vtk.vtkActor()
        actor_mesh.SetMapper(mapper_mesh)

        actor_mesh.GetProperty().SetRepresentationToWireframe()
        actor_mesh.GetProperty().SetColor(1, 1, 1)
        actor_mesh.GetProperty().SetLineWidth(1)
        self.renderer.AddActor(actor_mesh)

        # 散点图层

        mapper_scatter = vtk.vtkPolyDataMapper()
        mapper_scatter.SetInputData(self.mesh)

        color_map = vtk.vtkDiscretizableColorTransferFunction()

        color_map.SetRange(0.0, 5.0)
        color_map.SetColorSpaceToLab()
        color_map.SetScaleToLinear()

        color_map.AddRGBPoint(0, 0, 0, 0)
        color_map.AddRGBPoint(1, 1, 0, 0)
        color_map.AddRGBPoint(2, 0, 1, 0)
        color_map.AddRGBPoint(3, 0, 0, 1)
        color_map.AddRGBPoint(4, 1, 1, 0)
        color_map.AddRGBPoint(5, 0, 1, 1)

        color_map.Build()

        mapper_scatter.SetLookupTable(color_map)
        mapper_scatter.Update()

        actor_scatter = vtk.vtkActor()
        actor_scatter.SetMapper(mapper_scatter)
        actor_scatter.GetProperty().SetRepresentationToPoints()
        actor_scatter.GetProperty().SetPointSize(7)
        self.renderer.AddActor(actor_scatter)

    def start(
        self,
        interaction: bool = True,
        file_name=None,
        resolution=(1920, 1080),
        background=(0.4, 0.4, 0.4),
    ):
        # 设置渲染器的背景颜色
        self.renderer.SetBackground(*background)
        self.render_window.SetSize(*resolution)
        # 渲染场景
        self.render_window.GetRenderers()
        self.render_window.Render()
        if file_name is not None:
            self.save_img(file_name)
        if interaction:
            self.render_window_interactor.Start()


class Basemanager:
    def far_field_boundary_split(self, dataset):
        pass

    def triangles_to_faces(self, faces, mesh_pos, deform=False):
        """Computes mesh edges from triangles."""
        mesh_pos = torch.from_numpy(mesh_pos)
        if not deform:
            # collect edges from triangles
            edges = torch.cat(
                (
                    faces[:, 0:2],
                    faces[:, 1:3],
                    torch.stack((faces[:, 2], faces[:, 0]), dim=1),
                ),
                dim=0,
            )
            # those edges are sometimes duplicated (within the mesh) and sometimes
            # single (at the mesh boundary).
            # sort & pack edges as single tf.int64
            receivers, _ = torch.min(edges, dim=1)
            senders, _ = torch.max(edges, dim=1)

            packed_edges = torch.stack((senders, receivers), dim=1)
            unique_edges = torch.unique(
                packed_edges, return_inverse=False, return_counts=False, dim=0
            )
            senders, receivers = torch.unbind(unique_edges, dim=1)
            senders = senders.to(torch.int64)
            receivers = receivers.to(torch.int64)

            two_way_connectivity = (
                torch.cat((senders, receivers), dim=0),
                torch.cat((receivers, senders), dim=0),
            )
            unique_edges = torch.stack((senders, receivers), dim=1)

            # plot_edge_direction(mesh_pos,unique_edges)

            # face_with_bias = reorder_face(mesh_pos,unique_edges,plot=True)
            # edge_with_bias = reorder_face(mesh_pos,packed_edges,plot=True)

            return {
                "two_way_connectivity": two_way_connectivity,
                "senders": senders,
                "receivers": receivers,
                "unique_edges": unique_edges,
                "face_with_bias": unique_edges,
                "edge_with_bias": packed_edges,
            }

        else:
            edges = torch.cat(
                (
                    faces[:, 0:2],
                    faces[:, 1:3],
                    faces[:, 2:4],
                    torch.stack((faces[:, 3], faces[:, 0]), dim=1),
                ),
                dim=0,
            )
            # those edges are sometimes duplicated (within the mesh) and sometimes
            # single (at the mesh boundary).
            # sort & pack edges as single tf.int64
            receivers, _ = torch.min(edges, dim=1)
            senders, _ = torch.max(edges, dim=1)

            packed_edges = torch.stack((senders, receivers), dim=1)
            unique_edges = torch.unique(
                packed_edges, return_inverse=False, return_counts=False, dim=0
            )
            senders, receivers = torch.unbind(unique_edges, dim=1)
            senders = senders.to(torch.int64)
            receivers = receivers.to(torch.int64)

            two_way_connectivity = (
                torch.cat((senders, receivers), dim=0),
                torch.cat((receivers, senders), dim=0),
            )
            return {
                "two_way_connectivity": two_way_connectivity,
                "senders": senders,
                "receivers": receivers,
            }

    def position_relative_to_line_pytorch(A, B, angle_c):
        # A是点的坐标，表示为(x, y)的元组
        # B是一个数组，shape为[nums, 2]，其中nums为参与判断的点数量，2为xy坐标
        # angle_c是与X轴的夹角，以角度为单位

        # 将输入转换为张量
        A = torch.tensor(A, dtype=torch.float64)
        B = torch.tensor(B, dtype=torch.float64)
        angle_c = torch.tensor(angle_c, dtype=torch.float64)

        # 计算直线的方向向量
        direction_vector = torch.tensor(
            [
                torch.cos(angle_c * math.pi / 180.0),
                torch.sin(angle_c * math.pi / 180.0),
            ],
            dtype=torch.float64,
        )

        # 计算向量AB
        vector_AB = B - A

        # 计算两个向量的叉积，注意这里使用广播
        cross_product = (
            direction_vector[0] * vector_AB[:, 1]
            - direction_vector[1] * vector_AB[:, 0]
        )

        # 判断每个点相对于直线的位置，返回一个mask
        mask = cross_product > 0
        return mask.view(-1, 1)  # 调整shape为[nums, 1]

    def plot_state(
        self, mesh_pos, edge_index, face_center_pos=None, centroid=None, node_type=None
    ):
        mesh_pos = np.array(mesh_pos)
        edge_index = np.array(edge_index)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.cla()
        ax.set_aspect("equal")

        # 通过索引获取每一条边的两个点的坐标
        point1 = mesh_pos[edge_index[:, 0]]
        point2 = mesh_pos[edge_index[:, 1]]

        # 将每一对点的坐标合并，方便绘图
        lines = np.hstack([point1, point2]).reshape(-1, 2, 2)

        # 使用plot绘制所有的边
        plt.plot(lines[:, :, 0].T, lines[:, :, 1].T, "k-", lw=1, alpha=0.2)

        node_size = 5
        if face_center_pos is not None:
            plt.scatter(
                face_center_pos[:, 0], face_center_pos[:, 1], c="red", linewidths=1, s=1
            )

        if centroid is not None:
            plt.scatter(centroid[:, 0], centroid[:, 1], c="blue", linewidths=1, s=1)

        if node_type is not None:
            try:
                node_type = node_type.view(-1)
            except:
                node_type = node_type.reshape(-1)
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

        plt.show()
        plt.close()

    def is_convex(self, polygon):
        """
        检查一个多边形是否是凸的。
        :param polygon: 多边形的顶点坐标，一个二维numpy数组。
        :return: 如果是凸的返回True，否则返回False。
        """
        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            c = polygon[(i + 2) % n]
            ba = a - b
            bc = c - b
            cross_product = np.cross(ba, bc)
            if cross_product < 0:
                return False
        return True

    def reorder_polygon(self, polygon):
        """
        重新排序多边形的顶点使其成为一个凸多边形。
        :param polygon: 多边形的顶点坐标，一个二维numpy数组。
        :return: 重新排序后的多边形的顶点坐标。
        """
        centroid = np.mean(polygon, axis=0)
        sorted_polygon = sorted(
            polygon, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        )
        return np.array(sorted_polygon)

    def ensure_counterclockwise(self, cells, mesh_pos):
        """
        确保每个单元的顶点是按逆时针顺序排列的，并且是凸的。
        :param cells: 单元的索引数组。
        :param mesh_pos: 顶点的坐标数组。
        :return: 调整后的cells数组。
        """
        for i, cell in enumerate(cells):
            vertices = mesh_pos[cell]
            if not self.is_convex(vertices):
                vertices = self.reorder_polygon(vertices)
                sorted_indices = sorted(
                    range(len(cell)),
                    key=lambda k: list(map(list, vertices)).index(
                        list(mesh_pos[cell][k])
                    ),
                )
                cells[i] = np.array(cell)[sorted_indices]
        return cell

    def is_equal(self, x, pivot):
        """
        Determine if a value x is between two other values a and b, and if x and pivot have the same sign.

        Parameters:
        - x (float or int): The value to check.
        - pivot (float or int): The pivot value for comparison.

        Returns:
        - (bool): True if x is between a and b (inclusive) and x and pivot have the same sign, False otherwise.
        """
        a = abs(pivot) - float(1e-8)
        b = abs(pivot) + float(1e-8)
        # Check if x is between a and b, inclusive, and if x and pivot have the same sign
        if a <= abs(x) <= b and math.copysign(1, x) == math.copysign(1, pivot):
            return True
        else:
            return False
        
    def is_coordinate_in_array(single_coord, array_coords):
        """
        判断单个坐标是否存在于坐标数组中

        :param single_coord: 单个坐标，维度为[1,2]
        :param array_coords: 坐标数组，维度为[N,2]
        :return: 布尔值，如果单个坐标存在于数组中，则为True，否则为False
        """
        # 使用numpy的广播机制和逻辑运算检查坐标是否匹配
        matches = np.all(array_coords == single_coord, axis=1)
        
        # 如果有任何匹配，则返回True
        return np.any(matches)

class TecplotMesh(Basemanager):
    """
    Tecplot .dat file is only supported with Tobias`s airfoil dataset ,No more data file supported
    """

    def __init__(
        self, file_path, tf_writer=None, h5_writer=None, origin_writer=None, path=None
    ):
        self.mesh_info = {
            "mesh_pos": None,
            "cells_index": None,
            "U": None,
            "V": None,
            "P": None,
        }
        self.boundary_mesh_info = {"mesh_pos": None, "face_node": None}
        self._parse_file_test(file_path)
        self.node_type = []
        self.tf_writer = tf_writer
        self.h5_writer = h5_writer
        self.origin_writer = origin_writer
        self.path = path

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

    def extract_air_foil_boundary(self):
        self.node_type = torch.from_numpy(self.mesh_info["node_type"]).view(-1, 1)
        airfoil_to_wall_mask = (self.node_type == NodeType.AIRFOIL).view(-1)
        self.node_type[airfoil_to_wall_mask] = NodeType.WALL_BOUNDARY
        if len(self.node_type.shape) > 1:
            flatten = self.node_type[:, 0]
        else:
            flatten = self.node_type

        self.c_NORMAL = flatten[flatten == NodeType.NORMAL].shape[0]
        self.c_OBSTACLE = flatten[flatten == NodeType.OBSTACLE].shape[0]
        self.c_AIRFOIL = flatten[flatten == NodeType.AIRFOIL].shape[0]
        self.c_HANDLE = flatten[flatten == NodeType.HANDLE].shape[0]
        self.c_INFLOW = flatten[flatten == NodeType.INFLOW].shape[0]
        self.c_OUTFLOW = flatten[flatten == NodeType.OUTFLOW].shape[0]
        self.c_WALL_BOUNDARY = flatten[flatten == NodeType.WALL_BOUNDARY].shape[0]
        self.c_SIZE = flatten[flatten == NodeType.SIZE].shape[0]
        self.c_GHOST_INFLOW = flatten[flatten == NodeType.GHOST_INFLOW].shape[0]
        self.c_GHOST_OUTFLOW = flatten[flatten == NodeType.GHOST_OUTFLOW].shape[0]
        self.c_GHOST_WALL_BOUNDARY = flatten[flatten == NodeType.GHOST_WALL].shape[0]
        self.c_GHOST_AIRFOIL = flatten[flatten == NodeType.GHOST_AIRFOIL].shape[0]

        print(
            "After readed data in file has NODE TYPE: NORMAL: {0} OBSTACLE: {1} AIRFOIL: {2} HANDLE: {3} INFLOW: {4} OUTFLOW: {5} WALL_BOUNDARY: {6} SIZE: {7}".format(
                self.c_NORMAL,
                self.c_OBSTACLE,
                self.c_AIRFOIL,
                0,
                self.c_INFLOW,
                self.c_OUTFLOW,
                self.c_WALL_BOUNDARY,
                0,
            )
        )
        self.node_type = self.node_type.numpy()
        self.path["flow_type"] = "farfield-circular"

    def extract_pipe_flow_boundary(self, mesh_boundary_pos=None):
        mesh_pos = self.mesh_pos
        topwall = np.max(mesh_pos[:, 1])
        bottomwall = np.min(mesh_pos[:, 1])
        outlet = np.max(mesh_pos[:, 0])
        inlet = np.min(mesh_pos[:, 0])

        x_mesh_boundary_pos = mesh_boundary_pos[:, 0]
        y_mesh_boundary_pos = mesh_boundary_pos[:, 1]

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
        
    def extract_cavity_flow_boundary(
        self, mesh_boundary_pos=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        topinflow = np.max(mesh_pos[:, 1])
        bottomwall = np.min(mesh_pos[:, 1])
        rightwall = np.max(mesh_pos[:, 0])
        leftwall = np.min(mesh_pos[:, 0])

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        for i in range(mesh_pos.shape[0]):
            current_coord = mesh_pos[i]
            # left wall
            if (
                (self.is_equal(current_coord[0], leftwall))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topinflow - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1
            # right wall
            elif (
                (self.is_equal(current_coord[0], rightwall))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topinflow - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1
            # top inflow
            elif (
                (self.is_equal(current_coord[1], topinflow))
                and (current_coord[0] >= leftwall)
                and (current_coord[0] <= rightwall)
            ):
                self.node_type[i] = NodeType.INFLOW
                self.INFLOW += 1
                INFLOW_list.append(current_coord)
            # bottom wall
            elif (
                (self.is_equal(current_coord[1], bottomwall))
                and (current_coord[0] >= leftwall)
                and (current_coord[0] <= rightwall)
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                self.WALL_BOUNDARY += 1
                WALL_BOUNDARY_list.append(current_coord)
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

    def extract_circular_boundary(
        self, mesh_boundary_pos=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        mesh_boundary_pos = mesh_boundary_pos.astype(dtype=np.float32)
        
        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        
        for i in range(mesh_pos.shape[0]):
            current_coord = mesh_pos[i:i+1]
            # left wall
            if (
                (self.is_coordinate_in_array(current_coord, mesh_boundary_pos))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1

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

        
        
    def extract_mesh(self, plot, data_index=None, mesh_only=True):
        self.INFLOW = 0
        self.WALL_BOUNDARY = 0
        self.OUTFLOW = 0
        self.OBSTACLE = 0
        self.NORMAL = 0
        self.node_type = np.empty((self.mesh_pos.shape[0], 1))

        if self.path["flow_type"] == "pipe_flow" or (
            "pipe" in self.path["mesh_file_path"]
        ):
            self.path["flow_type"] = "pipe_flow"
            self.extract_pipe_flow_boundary(
                mesh_boundary_pos=self.boundary_mesh_info["mesh_pos"]
            )
        elif self.path["flow_type"] == "cavity_flow" or (
            "cavity" in self.path["mesh_file_path"]
        ):
            self.path["flow_type"] = "cavity_flow"
            self.extract_cavity_flow_boundary(
                mesh_boundary_pos=self.boundary_mesh_info["mesh_pos"]
            )
        elif self.path["flow_type"] == "circular" or (
            "circular" in self.path["mesh_file_path"]
        ):
            self.path["flow_type"] = "circular-possion"
            self.extract_circular_boundary(
                mesh_boundary_pos=self.boundary_mesh_info["mesh_pos"]
            )
        # self.plot_state(self.mesh_info["mesh_pos"],self.mesh_info["face_node"],node_type = self.node_type)

        if mesh_only:
            mesh = {
                "mesh_pos": torch.from_numpy(self.mesh_info["mesh_pos"])
                .to(torch.float64)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "cells_node": self.mesh_info["cells_node"]
                .to(torch.long)
                .unsqueeze(0)
                .unsqueeze(2)
                .repeat(1, 1, 1),
                "cells_index": self.mesh_info["cells_index"]
                .to(torch.long)
                .unsqueeze(0)
                .unsqueeze(2)
                .repeat(1, 1, 1),
                "cells_face_node": self.mesh_info["cells_face_node"]
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "node_type": torch.from_numpy(self.node_type)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
            }

        # if plot:
        #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        #     ax.cla()
        #     ax.set_aspect('equal')
        #     mesh_pos = mesh["mesh_pos"][0]
        #     cells_node = mesh["cells_node"][0]
        #     node_type = mesh['node_type'][0,:,0]
        #     triang = mtri.Triangulation(mesh_pos[:,0], mesh_pos[:,1],cells_node)
        #     ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        #     plt.scatter(mesh_pos[node_type==NodeType.WALL_BOUNDARY,0],mesh_pos[node_type==NodeType.WALL_BOUNDARY,1],c='red',linewidths=1)
        #     plt.scatter(mesh_pos[node_type==NodeType.INFLOW,0],mesh_pos[node_type==NodeType.INFLOW,1],c='green',linewidths=1)
        #     plt.scatter(mesh_pos[node_type==NodeType.OUTFLOW,0],mesh_pos[node_type==NodeType.OUTFLOW,1],c='orange',linewidths=1)
        #     plt.show()
        #     plt.close()

        tf_dataset, origin_dataset, h5_dataset = extract_mesh_state(
            dataset=mesh,
            tf_writer=self.tf_writer,
            index=data_index,
            origin_writer=self.origin_writer,
            mode="cylinder_mesh",
            h5_writer=self.h5_writer,
            path=self.path,
            mesh_only=mesh_only,
            plot=plot,
        )

        return tf_dataset, origin_dataset, h5_dataset


class Cosmol_manager(Basemanager):
    def __init__(
        self,
        mesh_file,
        data_file,
        tf_writer=None,
        h5_writer=None,
        origin_writer=None,
        path=None,
    ):
        self.node_type = []
        self.tf_writer = tf_writer
        self.h5_writer = h5_writer
        self.origin_writer = origin_writer
        self.path = path

        self.mesh_file_handle = self.openreadtxt(mesh_file, mesh_reading=True)

        try:
            file_name = os.path.basename(mesh_file)

            """>>>load inflow BC mesh pos>>>"""
            BC_inflow_file = os.path.join(
                os.path.dirname(mesh_file), file_name.split(".")[0] + "-inflow.txt"
            )
            self.mesh_inflow_boundary_pos = self.readBCtxt(
                BC_inflow_file, data_reading=True
            ).astype(np.float32)
            """<<<load inflow BC mesh pos<<<"""

            """>>>load wall BC mesh pos>>>"""
            BC_wall_file = os.path.join(
                os.path.dirname(mesh_file), file_name.split(".")[0] + "-wall.txt"
            )
            self.mesh_wall_boundary_pos = self.readBCtxt(
                BC_wall_file, data_reading=True
            ).astype(np.float32)
            """<<<load wall BC mesh pos<<<"""

            """>>>load outflow BC mesh pos>>>"""
            BC_outflow_file = os.path.join(
                os.path.dirname(mesh_file), file_name.split(".")[0] + "-outflow.txt"
            )
            self.mesh_outflow_boundary_pos = self.readBCtxt(
                BC_outflow_file, data_reading=True
            ).astype(np.float32)
            """<<<load outflow BC mesh pos<<<"""

        except:
            self.mesh_inflow_boundary_pos = np.empty(0)
            self.mesh_wall_boundary_pos = np.empty(0)
            self.mesh_outflow_boundary_pos = np.empty(0)

        self.X = self.mesh_pos[:, 0:1]
        self.Y = self.mesh_pos[:, 1:2]

        if not path["mesh_only"]:
            self.data_file_handle = self.openreadtxt(data_file, data_reading=True)

        else:
            self.U = (
                torch.zeros_like(torch.from_numpy(self.X))
                .unsqueeze(0)
                .repeat(600, 1, 1)
                .numpy()
            )
            self.V = (
                torch.zeros_like(torch.from_numpy(self.X))
                .unsqueeze(0)
                .repeat(600, 1, 1)
                .numpy()
            )
            self.P = (
                torch.zeros_like(torch.from_numpy(self.X))
                .unsqueeze(0)
                .repeat(600, 1, 1)
                .numpy()
            )

        # if not self.mesh_file_handle or not self.data_file_handle:
        #     raise ValueError("")

    def openreadtxt(self, file_name, mesh_reading=False, data_reading=False):
        self.raw_txt_data = []
        file = open(file_name, "r")  # 打开文件
        file_data = file.readlines()  # 读取所有行
        count = 0
        index = 0
        self.X = []
        self.Y = []
        self.U = []
        self.V = []
        self.P = []
        self.mesh_boundary_index = []
        self.cells_face_node = []
        self.cells_node = np.array([], dtype=np.int64)
        self.cells_index = np.array([], dtype=np.int64)
        self.last_ele_index = -1
        self.total_num_of_elements = 0
        while index < len(file_data):
            count += 1
            row = file_data[index]
            if row.find("# number of mesh vertices\n") > 0:
                tmp_list = row.split(" ")  # 按‘ '切分每行的数据
                self.num_of_mesh_pos = int(tmp_list[0])
            elif row == "# Mesh vertex coordinates\n":
                index = self.read_mesh_pos(file_data, index + 1, self.num_of_mesh_pos)
            elif row == "3 edg # type name\n":
                index = self.read_index(file_data, index)
            elif row == "3 tri # type name\n":
                index = self.read_index(file_data, index)
            elif row == "4 quad # type name\n":
                index = self.read_index(file_data, index)
            elif row.find("x (m) @ t") > 0 or row.find("% x") == 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                x_tmp = [np.array(x, dtype=np.float64) for x in tmp_list]
                self.X.append(np.array(x_tmp))
            elif row.find("y (m) @ t") > 0 or row.find("% y") == 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                y_tmp = [np.array(y, dtype=np.float64) for y in tmp_list]
                self.Y.append(np.array(y_tmp))
            elif row.find("u (m/s) @") > 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                u_tmp = [np.array(u, dtype=np.float64) for u in tmp_list]
                self.U.append(np.array(u_tmp))
            elif row.find("v (m/s) @") > 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                v_tmp = [np.array(v, dtype=np.float64) for v in tmp_list]
                self.V.append(np.array(v_tmp))
            elif row.find("p (Pa) @") > 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                p_tmp = [np.array(p, dtype=np.float64) for p in tmp_list]
                self.P.append(np.array(p_tmp))

            """
            tmp_list = row.split(' ') #按‘，'切分每行的数据
            tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
            self.raw_txt_data.append(tmp_list) #将每行数据插入data中
            """
            index += 1
        try:
            if mesh_reading:
                self.cells_index = np.array(self.cells_index)
                self.cells_face_node = np.array(self.cells_face_node)
                if np.max(self.cells_index) != (self.total_num_of_elements - 1):
                    raise ValueError(
                        f"data{file_name} read faild, wrong element reading"
                    )
                print("mesh read done")
            elif data_reading and len(self.X) == 0:
                raise ValueError(f"data{file_name} read faild")
            elif data_reading and len(self.X) > 0:
                print("data read done")
            return True
        except:
            return False

    def readBCtxt(self, file_name, mesh_reading=False, data_reading=False):
        self.raw_txt_data = []
        file = open(file_name, "r")  # 打开文件
        file_data = file.readlines()  # 读取所有行
        index = 0
        mesh_wall_boundary_pos = []

        while index < len(file_data):
            row = file_data[index]
            if "% x" in row:
                row = file_data[index + 1]
                boundary_pos_X = np.array([float(x) for x in row.strip().split() if x])

            elif "% y" in row:
                row = file_data[index + 1]
                boundary_pos_Y = np.array([float(y) for y in row.strip().split() if y])

            index += 1

        mesh_wall_boundary_pos = np.stack((boundary_pos_X, boundary_pos_Y), axis=1)

        return mesh_wall_boundary_pos

    def read_mesh_pos(self, input, start, end):
        self.mesh_pos = []
        for index in range(start, end + start):
            raw_data = input[index].split(" ")
            raw_data[-1] = raw_data[-1].replace("\n", ",")
            raw_x = np.array(raw_data[0], dtype=np.float64)
            raw_y = np.array(raw_data[1], dtype=np.float64)
            raw_pos = np.array([raw_x, raw_y])
            self.mesh_pos.append(raw_pos)
        self.mesh_pos = np.array(self.mesh_pos)
        return index

    def read_header(self, input, start, end):
        self.mesh_pos.append(input, start, end)

    def read_index(self, input, start):
        while start < len(input):
            if input[start].find("# number of elements\n") > 0:
                raw_data = input[start].split(" ")
                raw_data[-1] = raw_data[-1].replace("\n", ",")
                num_of_elements = int(raw_data[0])

            elif input[start].find("# number of vertices per element\n") > 0:
                raw_data = input[start].split(" ")
                raw_data[-1] = raw_data[-1].replace("\n", ",")
                num_of_vertices_per_elements = int(raw_data[0])

            elif input[start] == "# Elements\n":
                current_ele_list = []
                for ele_index, start in enumerate(
                    range(start + 1, start + num_of_elements + 1)
                ):
                    raw_cells_node_index_list = []
                    ele_index_list = []
                    raw_data = list(map(int, input[start].strip().split(" ")))
                    raw_data = self.ensure_counterclockwise(
                        np.array([raw_data]), self.mesh_pos
                    )
                    for vertex_index in range(num_of_vertices_per_elements):
                        raw_index = np.array(int(raw_data[vertex_index]))
                        raw_cells_node_index_list.append(raw_index)
                        ele_index_list.append(ele_index + 1 + self.last_ele_index)

                    raw_cells_node_index_list = np.array(raw_cells_node_index_list)
                    current_ele_list.append(raw_cells_node_index_list)

                    if num_of_vertices_per_elements > 2:
                        self.cells_index = np.concatenate(
                            (self.cells_index, np.array(ele_index_list).reshape(-1)),
                            axis=0,
                        )
                        for edge_num in range(num_of_vertices_per_elements - 1):
                            self.cells_face_node.append(
                                [
                                    raw_cells_node_index_list[edge_num],
                                    raw_cells_node_index_list[edge_num + 1],
                                ]
                            )
                        self.cells_face_node.append(
                            [
                                raw_cells_node_index_list[-1],
                                raw_cells_node_index_list[0],
                            ]
                        )

                if num_of_vertices_per_elements == 2:
                    self.mesh_boundary_index = np.array(current_ele_list)

                elif num_of_vertices_per_elements > 2:
                    self.cells_node = np.concatenate(
                        (self.cells_node, np.array(current_ele_list).reshape(-1)),
                        axis=0,
                    )

                break

            elif (
                input[start] == "3 vtx # type name\n"
                or "number of geometric entity indices" in input[start]
            ):
                break
            start += 1

        if num_of_vertices_per_elements > 2:
            self.last_ele_index = self.cells_index[-1]
            self.total_num_of_elements += num_of_elements

        return start

    def extract_pipe_flow_boundary(
        self, mesh_boundary_index=None, rearrange_index=None, rearrange_pos_dict=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
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
                (i in mesh_boundary_index)
                and (current_coord[0] > 0)
                and (current_coord[0] < (outlet - (1e-12)))
                and (current_coord[1] > 0)
                and (current_coord[1] < (topwall - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                self.WALL_BOUNDARY += 1
                self.OBSTACLE += 1
                OBSTACLE_list.append(current_coord)
            else:
                self.node_type[i] = NodeType.NORMAL
                self.NORMAL += 1
            rearrange_index[i] = rearrange_pos_dict[str(current_coord)]

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

        return rearrange_index

    def extract_cavity_wave_boundary(
        self, mesh_boundary_index=None, rearrange_index=None, rearrange_pos_dict=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        topinflow = np.max(mesh_pos[:, 1])
        bottomwall = np.min(mesh_pos[:, 1])
        rightwall = np.max(mesh_pos[:, 0])
        leftwall = np.min(mesh_pos[:, 0])

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        for i in range(mesh_pos.shape[0]):
            current_coord = mesh_pos[i]
            # left wall
            if (
                (self.is_equal(current_coord[0], leftwall))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topinflow - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1
            # right wall
            elif (
                (self.is_equal(current_coord[0], rightwall))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topinflow - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1
            # top inflow
            elif (
                (self.is_equal(current_coord[1], topinflow))
                and (current_coord[0] >= leftwall)
                and (current_coord[0] <= rightwall)
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                self.WALL_BOUNDARY += 1
                WALL_BOUNDARY_list.append(current_coord)
            # bottom wall
            elif (
                (self.is_equal(current_coord[1], bottomwall))
                and (current_coord[0] >= leftwall)
                and (current_coord[0] <= rightwall)
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                self.WALL_BOUNDARY += 1
                WALL_BOUNDARY_list.append(current_coord)
            else:
                self.node_type[i] = NodeType.NORMAL
                self.NORMAL += 1
            rearrange_index[i] = rearrange_pos_dict[str(current_coord)]
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

        return rearrange_index

    def extract_cavity_flow_boundary(
        self, mesh_boundary_index=None, rearrange_index=None, rearrange_pos_dict=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        topinflow = np.max(mesh_pos[:, 1])
        bottomwall = np.min(mesh_pos[:, 1])
        rightwall = np.max(mesh_pos[:, 0])
        leftwall = np.min(mesh_pos[:, 0])

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        for i in range(mesh_pos.shape[0]):
            current_coord = mesh_pos[i]
            # left wall
            if (
                (self.is_equal(current_coord[0], leftwall))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topinflow - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1
            # right wall
            elif (
                (self.is_equal(current_coord[0], rightwall))
                and (current_coord[1] > (bottomwall + (1e-12)))
                and (current_coord[1] < (topinflow - (1e-12)))
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                self.WALL_BOUNDARY += 1
            # top inflow
            elif (
                (self.is_equal(current_coord[1], topinflow))
                and (current_coord[0] >= leftwall)
                and (current_coord[0] <= rightwall)
            ):
                self.node_type[i] = NodeType.INFLOW
                self.INFLOW += 1
                INFLOW_list.append(current_coord)
            # bottom wall
            elif (
                (self.is_equal(current_coord[1], bottomwall))
                and (current_coord[0] >= leftwall)
                and (current_coord[0] <= rightwall)
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                self.WALL_BOUNDARY += 1
                WALL_BOUNDARY_list.append(current_coord)
            else:
                self.node_type[i] = NodeType.NORMAL
                self.NORMAL += 1
            rearrange_index[i] = rearrange_pos_dict[str(current_coord)]
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

        return rearrange_index

    def extract_circular_possion_boundary(
        self, mesh_boundary_index=None, rearrange_index=None, rearrange_pos_dict=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        average_length = torch.mean(
            torch.norm(torch.from_numpy(mesh_pos), dim=1, keepdim=True)
        )
        topoutflow = np.max(mesh_pos[:, 1])
        bottomoutflow = np.min(mesh_pos[:, 1])
        rightoutflow = np.max(mesh_pos[:, 0])
        leftinflow = np.min(mesh_pos[:, 0])
        radius = np.minimum(topoutflow - bottomoutflow, rightoutflow - leftinflow) / 2.0
        center_pos = (
            torch.tensor(
                [(rightoutflow + leftinflow) / 2.0, (topoutflow + bottomoutflow) / 2.0]
            )
            .view(1, 2)
            .numpy()
        )
        moved_to_center_pos = mesh_pos - center_pos

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        for i in range(mesh_pos.shape[0]):
            current_moved_coord = moved_to_center_pos[i]
            # current_coord_to_origin = torch.norm(torch.from_numpy(current_coord).view(-1,2))
            current_coord_to_origin = mesh_pos[i]
            # distance_to_origin = torch.norm(torch.from_numpy(current_moved_coord).view(-1, 2))
            if i in mesh_boundary_index:
                if (
                    (current_coord_to_origin[0] > (leftinflow + radius / 2.0))
                    and (current_coord_to_origin[0] < (rightoutflow - radius / 2.0))
                    and (current_coord_to_origin[1] > (bottomoutflow + radius / 2.0))
                    and (current_coord_to_origin[1] < (topoutflow - radius / 2.0))
                ):
                    self.node_type[i] = NodeType.WALL_BOUNDARY
                    WALL_BOUNDARY_list.append(current_coord_to_origin)
                    self.WALL_BOUNDARY += 1

                else:
                    self.node_type[i] = NodeType.INFLOW
                    INFLOW_list.append(current_coord_to_origin)
                    self.OUTFLOW += 1
            else:
                self.node_type[i] = NodeType.NORMAL
                self.NORMAL += 1

            rearrange_index[i] = rearrange_pos_dict[str(current_coord_to_origin)]
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

        return rearrange_index

    def extract_far_field_boundary(
        self, mesh_boundary_index=None, rearrange_index=None, rearrange_pos_dict=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        average_length = torch.mean(
            torch.norm(torch.from_numpy(mesh_pos), dim=1, keepdim=True)
        )
        topinflow = np.max(mesh_pos[:, 1])
        bottominflow = np.min(mesh_pos[:, 1])
        rightinflow = np.max(mesh_pos[:, 0])
        leftinflow = np.min(mesh_pos[:, 0])

        radius = ((topinflow - bottominflow) + (rightinflow - leftinflow)) / 4.0
        center_pos = (
            torch.tensor(
                [(rightinflow + leftinflow) / 2.0, (topinflow + bottominflow) / 2.0]
            )
            .view(1, 2)
            .numpy()
        )
        moved_to_center_pos = mesh_pos - center_pos

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        for i in range(mesh_pos.shape[0]):
            current_moved_coord = moved_to_center_pos[i]
            # current_coord_to_origin = torch.norm(torch.from_numpy(current_coord).view(-1,2))
            current_coord_to_origin = mesh_pos[i]
            distance_to_origin = torch.norm(
                torch.from_numpy(current_moved_coord).view(-1, 2)
            )
            if i in mesh_boundary_index:
                # if ((current_coord[0]<rightinflow) and (current_coord[0]>leftinflow)and (current_coord[1]>bottominflow) and (current_coord[1]<topinflow)):
                if not (np.abs(distance_to_origin - radius) < radius / 2.0):
                    self.node_type[i] = NodeType.WALL_BOUNDARY
                    WALL_BOUNDARY_list.append(current_coord_to_origin)
                    self.WALL_BOUNDARY += 1
                else:
                    self.node_type[i] = NodeType.INFLOW
                    INFLOW_list.append(current_coord_to_origin)
                    self.INFLOW += 1

            else:
                self.node_type[i] = NodeType.NORMAL
                self.NORMAL += 1

            rearrange_index[i] = rearrange_pos_dict[str(current_coord_to_origin)]
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

        return rearrange_index

    def extract_far_field_square_boundary(
        self, mesh_boundary_index=None, rearrange_index=None, rearrange_pos_dict=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        average_length = torch.mean(
            torch.norm(torch.from_numpy(mesh_pos), dim=1, keepdim=True)
        )
        topoutflow = np.max(mesh_pos[:, 1])
        bottomoutflow = np.min(mesh_pos[:, 1])
        rightoutflow = np.max(mesh_pos[:, 0])
        leftinflow = np.min(mesh_pos[:, 0])
        radius = np.minimum(topoutflow - bottomoutflow, rightoutflow - leftinflow) / 2.0
        center_pos = (
            torch.tensor(
                [(rightoutflow + leftinflow) / 2.0, (topoutflow + bottomoutflow) / 2.0]
            )
            .view(1, 2)
            .numpy()
        )
        moved_to_center_pos = mesh_pos - center_pos

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        for i in range(mesh_pos.shape[0]):
            current_moved_coord = moved_to_center_pos[i]
            # current_coord_to_origin = torch.norm(torch.from_numpy(current_coord).view(-1,2))
            current_coord_to_origin = mesh_pos[i]
            # distance_to_origin = torch.norm(torch.from_numpy(current_moved_coord).view(-1, 2))
            if i in mesh_boundary_index:
                if (
                    (current_coord_to_origin[0] > 0)
                    and (current_coord_to_origin[0] < (rightoutflow - (1e-12)))
                    and (current_coord_to_origin[1] > 0)
                    and (current_coord_to_origin[1] < (topoutflow - (1e-12)))
                ):
                    self.node_type[i] = NodeType.WALL_BOUNDARY
                    WALL_BOUNDARY_list.append(current_coord_to_origin)
                    self.WALL_BOUNDARY += 1

                elif (
                    (self.is_equal(current_coord_to_origin[0], leftinflow))
                    and (current_coord_to_origin[1] >= bottomoutflow)
                    and (current_coord_to_origin[1] <= topoutflow)
                ):
                    self.node_type[i] = NodeType.INFLOW
                    self.INFLOW += 1

                else:
                    self.node_type[i] = NodeType.OUTFLOW
                    INFLOW_list.append(current_coord_to_origin)
                    self.OUTFLOW += 1
            else:
                self.node_type[i] = NodeType.NORMAL
                self.NORMAL += 1

            rearrange_index[i] = rearrange_pos_dict[str(current_coord_to_origin)]
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

        return rearrange_index

    def extract_far_field_circular_square_boundary(
        self, mesh_boundary_index=None, rearrange_index=None, rearrange_pos_dict=None
    ):
        mesh_pos = self.mesh_pos.astype(dtype=np.float32)
        topoutflow = np.max(mesh_pos[:, 1])
        rightoutflow = np.max(mesh_pos[:, 0])
        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        INFLOW_list = []
        try:
            for i in range(mesh_pos.shape[0]):
                current_coord_to_origin = mesh_pos[i]

                if (
                    (current_coord_to_origin[0] == self.mesh_wall_boundary_pos[:, 0])
                    & (current_coord_to_origin[1] == self.mesh_wall_boundary_pos[:, 1])
                ).sum() > 0:
                    self.node_type[i] = NodeType.WALL_BOUNDARY
                    WALL_BOUNDARY_list.append(current_coord_to_origin)
                    self.WALL_BOUNDARY += 1

                elif (
                    (
                        (
                            current_coord_to_origin[0]
                            == self.mesh_inflow_boundary_pos[:, 0]
                        )
                        & (
                            current_coord_to_origin[1]
                            == self.mesh_inflow_boundary_pos[:, 1]
                        )
                    ).sum()
                    > 0
                ) or (
                    (self.is_equal(current_coord_to_origin[0], rightoutflow))
                    and (self.is_equal(current_coord_to_origin[1], topoutflow))
                ):
                    self.node_type[i] = NodeType.INFLOW
                    INFLOW_list.append(current_coord_to_origin)
                    self.INFLOW += 1

                elif (
                    (current_coord_to_origin[0] == self.mesh_outflow_boundary_pos[:, 0])
                    & (
                        current_coord_to_origin[1]
                        == self.mesh_outflow_boundary_pos[:, 1]
                    )
                ).sum() > 0:
                    self.node_type[i] = NodeType.OUTFLOW
                    self.OUTFLOW += 1

                else:
                    self.node_type[i] = NodeType.NORMAL
                    self.NORMAL += 1

                rearrange_index[i] = rearrange_pos_dict[str(current_coord_to_origin)]

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

        except:
            raise ValueError(
                f"Wrong node type conversion {self.path['mesh_file_path']}"
            )

        if (
            (not self.WALL_BOUNDARY == (self.mesh_wall_boundary_pos.shape[0]))
            or (not self.INFLOW == (self.mesh_inflow_boundary_pos.shape[0]))
            or (not self.OUTFLOW == (self.mesh_outflow_boundary_pos.shape[0] - 2))
        ):
            raise ValueError(
                f"Wrong node type conversion {self.path['mesh_file_path']}"
            )

        return rearrange_index

    def extract_mesh(self, plot, data_index=None, mesh_only=True):
        self.INFLOW = 0
        self.WALL_BOUNDARY = 0
        self.OUTFLOW = 0
        self.OBSTACLE = 0
        self.NORMAL = 0
        mesh_pos = torch.from_numpy(self.mesh_pos)
        mesh_boundary_index = torch.from_numpy(self.mesh_boundary_index).to(torch.long)
        boundary_pos_0 = torch.index_select(
            mesh_pos, 0, mesh_boundary_index[:, 0]
        ).numpy()
        boundary_pos_1 = torch.index_select(
            mesh_pos, 0, mesh_boundary_index[:, 1]
        ).numpy()
        self.node_type = np.empty((self.mesh_pos.shape[0], 1))

        # prepare for renumber data
        rearrange_pos_dict = {}
        data_pos = np.concatenate((np.array(self.X), np.array(self.Y)), axis=1)

        for index in range(self.mesh_pos.shape[0]):
            rearrange_pos_dict[str(data_pos[index].astype(np.float32))] = index
        data_velocity = np.concatenate((np.array(self.U), np.array(self.V)), axis=2)
        data_pressure = np.array(self.P)
        rearrange_index = np.zeros(data_velocity.shape[1])

        if self.path["flow_type"] == "pipe_flow" or (
            "pipe" in self.path["mesh_file_path"]
        ):
            self.path["flow_type"] = "pipe_flow"
            rearrange_index = self.extract_pipe_flow_boundary(
                mesh_boundary_index=mesh_boundary_index,
                rearrange_index=rearrange_index,
                rearrange_pos_dict=rearrange_pos_dict,
            )

        elif ("cavity" in self.path["mesh_file_path"]) and (
            "wave" in self.path["mesh_file_path"]
        ):
            self.path["flow_type"] = "cavity_wave"
            rearrange_index = self.extract_cavity_wave_boundary(
                mesh_boundary_index=mesh_boundary_index,
                rearrange_index=rearrange_index,
                rearrange_pos_dict=rearrange_pos_dict,
            )

        elif "cavity" in self.path["mesh_file_path"]:
            self.path["flow_type"] = "cavity_flow"
            rearrange_index = self.extract_cavity_flow_boundary(
                mesh_boundary_index=mesh_boundary_index,
                rearrange_index=rearrange_index,
                rearrange_pos_dict=rearrange_pos_dict,
            )

        elif ("circular-possion" in self.path["mesh_file_path"]) or (
            "circular" in self.path["mesh_file_path"]
        ):
            self.path["flow_type"] = "circular-possion"
            rearrange_index = self.extract_circular_possion_boundary(
                mesh_boundary_index=mesh_boundary_index,
                rearrange_index=rearrange_index,
                rearrange_pos_dict=rearrange_pos_dict,
            )

        elif "farfield-circular" in self.path["mesh_file_path"]:
            self.path["flow_type"] = "farfield-circular"
            rearrange_index = self.extract_far_field_boundary(
                mesh_boundary_index=mesh_boundary_index,
                rearrange_index=rearrange_index,
                rearrange_pos_dict=rearrange_pos_dict,
            )

        elif "farfield-square" in self.path["mesh_file_path"]:
            self.path["flow_type"] = "farfield-square"
            rearrange_index = self.extract_far_field_square_boundary(
                mesh_boundary_index=mesh_boundary_index,
                rearrange_index=rearrange_index,
                rearrange_pos_dict=rearrange_pos_dict,
            )

        elif "farfield-half-circular-square" in self.path["mesh_file_path"]:
            self.path["flow_type"] = "farfield-half-circular-square"
            rearrange_index = self.extract_far_field_circular_square_boundary(
                mesh_boundary_index=mesh_boundary_index,
                rearrange_index=rearrange_index,
                rearrange_pos_dict=rearrange_pos_dict,
            )

        if mesh_only:
            mesh = {
                "mesh_pos": torch.from_numpy(self.mesh_pos)
                .to(torch.float64)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "boundary": torch.from_numpy(self.mesh_boundary_index)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "cells_node": torch.from_numpy(self.cells_node)
                .to(torch.long)
                .unsqueeze(0)
                .unsqueeze(2)
                .repeat(1, 1, 1),
                "cells_index": torch.from_numpy(self.cells_index)
                .to(torch.long)
                .unsqueeze(0)
                .unsqueeze(2)
                .repeat(1, 1, 1),
                "cells_face_node": torch.from_numpy(self.cells_face_node)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "node_type": torch.from_numpy(self.node_type)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
            }
        else:
            velocity = torch.index_select(
                torch.from_numpy(data_velocity),
                1,
                torch.from_numpy(rearrange_index).to(torch.long),
            )
            pressure = torch.index_select(
                torch.from_numpy(data_pressure),
                1,
                torch.from_numpy(rearrange_index).to(torch.long),
            )
            mesh = {
                "mesh_pos": torch.from_numpy(self.mesh_pos)
                .to(torch.float64)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "boundary": torch.from_numpy(self.mesh_boundary_index)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "cells_node": torch.from_numpy(self.cells_node)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "cells_index": torch.from_numpy(self.cells_index)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "cells_face_node": torch.from_numpy(self.cells_face_node)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "node_type": torch.from_numpy(self.node_type)
                .to(torch.long)
                .unsqueeze(0)
                .repeat(1, 1, 1),
                "velocity": velocity[0:600].astype(np.float64),
                "pressure": pressure[0:600].astype(np.float64),
            }

        # if plot:
        #     mesh_pos = np.array(self.mesh_pos)
        #     edge_index = np.array(self.cells_face_node)
        #     node_type = self.node_type.reshape(-1)

        #     fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        #     ax.cla()
        #     ax.set_aspect('equal')

        #     # 通过索引获取每一条边的两个点的坐标
        #     point1 = mesh_pos[edge_index[:, 0]]
        #     point2 = mesh_pos[edge_index[:, 1]]

        #     # 将每一对点的坐标合并，方便绘图
        #     lines = np.hstack([point1, point2]).reshape(-1, 2, 2)

        #     # 使用plot绘制所有的边
        #     plt.plot(lines[:, :, 0].T, lines[:, :, 1].T, 'k-', lw=1, alpha=0.2)

        #     # 绘制点
        #     node_size=10
        #     plt.scatter(mesh_pos[node_type==NodeType.NORMAL,0],mesh_pos[node_type==NodeType.NORMAL,1],c='red',linewidths=1,s=node_size,edgecolor="none")
        #     plt.scatter(mesh_pos[node_type==NodeType.WALL_BOUNDARY,0],mesh_pos[node_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=node_size,edgecolor="none")
        #     plt.scatter(mesh_pos[node_type==NodeType.OUTFLOW,0],mesh_pos[node_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=node_size,edgecolor="none")
        #     plt.scatter(mesh_pos[node_type==NodeType.INFLOW,0],mesh_pos[node_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=node_size,edgecolor="none")
        #     plt.show()
        #     plt.close()

        tf_dataset, origin_dataset, h5_dataset = extract_mesh_state(
            dataset=mesh,
            tf_writer=self.tf_writer,
            index=data_index,
            origin_writer=self.origin_writer,
            mode="cylinder_mesh",
            h5_writer=self.h5_writer,
            path=self.path,
            mesh_only=mesh_only,
            plot=plot,
        )

        return tf_dataset, origin_dataset, h5_dataset


def random_samples_no_replacement(arr, num_samples, num_iterations):
    if num_samples * num_iterations > len(arr):
        raise ValueError(
            "Number of samples multiplied by iterations cannot be greater than the length of the array."
        )

    samples = []
    arr_copy = arr.copy()

    for _ in range(num_iterations):
        sample_indices = np.random.choice(len(arr_copy), num_samples, replace=False)
        sample = arr_copy[sample_indices]
        samples.append(sample)

        # 从 arr_copy 中移除已选样本
        arr_copy = np.delete(arr_copy, sample_indices)

    return samples, arr_copy


# Define the processing function
def process_file(plot, dataset_type, file_index, file_path, path, queue):
    # try:
    file_name = os.path.basename(file_path)
    subdir = os.path.dirname(file_path)
    mesh_name = file_name
    data_name = f"data{''.join(char for char in mesh_name if char.isdigit())}.txt"
    path["mesh_file_path"] = file_path
    path["data_file_path"] = f"{subdir}/{data_name}"

    # start convert func
    if path["mesh_file_path"].endswith(".mphtxt"):
        data = Cosmol_manager(
            mesh_file=path["mesh_file_path"],
            data_file=path["data_file_path"],
            path=path,
        )

    elif path["mesh_file_path"].endswith(".dat"):
        data = TecplotMesh(file_path=path["mesh_file_path"], path=path)

    tf_dataset, origin_dataset, h5_dataset = data.extract_mesh(
        plot=plot, data_index=file_index, mesh_only=path["mesh_only"]
    )

    # tf_dataset["origin_mesh_path"] = np.expand_dims(string_to_floats(path["mesh_file_path"]),axis=(0,2))
    # origin_dataset["origin_mesh_path"] = np.expand_dims(string_to_floats(path["mesh_file_path"]),axis=(0,2))
    # h5_dataset["origin_mesh_path"] = np.expand_dims(string_to_floats(path["mesh_file_path"]),axis=(0,2))

    # Put the results in the queue
    queue.put(([tf_dataset, origin_dataset, h5_dataset], file_index, dataset_type))


# except:
#     print(f"parsing {subdir}/{mesh_name} failed\n")


def string_to_floats(s):
    """将字符串转换为一组浮点数"""
    return np.asarray([float(ord(c)) for c in s])


def floats_to_string(floats):
    """将一组浮点数转换为字符串"""
    return "".join([chr(int(f)) for f in floats])


# Writer process function
def writer_process(queue, all_writers, path):
    while True:
        # Get data from queue
        data, file_index, dataset_type = queue.get()

        # Break if None is received (sentinel value)
        if data is None:
            break

        # Write data to file
        tf_dataset, origin_dataset, h5_dataset = data

        # Write dataset information
        write_dict_info_to_json(
            tf_dataset[0], os.path.dirname(path["tf_saving_path"]) + "/meta.json"
        )
        write_dict_info_to_json(
            origin_dataset[0],
            os.path.dirname(path["origin_saving_path"]) + "/meta.json",
        )

        # Write dataset key value
        tf_writer, origin_writer, h5_writer = all_writers[dataset_type]
        write_tfrecord_one_with_writer(tf_writer, tf_dataset[1], mode="cylinder_mesh")
        write_tfrecord_one_with_writer(
            origin_writer, origin_dataset[1], mode="cylinder_mesh"
        )
        current_traj = h5_writer.create_group(str(file_index))
        for key, value in h5_dataset[1].items():
            current_traj.create_dataset(key, data=value)

        print("{0}th mesh has been writed".format(file_index))

    # 关闭所有的writer
    for writers in all_writers.values():
        for writer in writers:
            if isinstance(writer, tf.io.TFRecordWriter):
                writer.close()
            elif isinstance(writer, h5py.File):
                writer.close()


def run_command(tfrecord_file, idx_file):
    subprocess.run(
        ["python", "-m", "tfrecord.tools.tfrecord2idx", tfrecord_file, idx_file],
        check=True,
    )


if __name__ == "__main__":
    # for debugging

    debug_file_path = None
    # debug_file_path="/mnt/dataset/work2/Dataset-GEP-FVGN-steady-with-poly/train_dataset/mesh_size=0.5.mphtxt"

    if debug_file_path is None or len(debug_file_path) <= 0:
        debug_file_path = None

    if debug_file_path is not None:
        multi_process = 1
    else:
        multi_process = 8

    flow_type_mapping = {
        "pipe_flow": 1,
        "farfield-circular": 2,
        "cavity_flow": 3,
        "farfield-square": 4,
        "farfield-half-circular-square": 5,
        "circular-possion": 6,
        "cavity_wave": 7,
    }

    case = 0  # 0 stands for 980/PM9A1
    if case == 0:
        path = {
            "simulator": "COMSOL",
            "dt": None,
            "rho": None,
            "mu": None,
            "features": None,
            "comsol_dataset_path": "/mnt/e/dataset/work2/Dataset-GEP-FVGN-steady-with-poly/train_dataset/",
            "h5_save_path": "/mnt/e/dataset/work2/Dataset-GEP-FVGN-steady-with-poly/converted_dataset/h5/",
            "tf_saving_path": "/mnt/e/dataset/work2/Dataset-GEP-FVGN-steady-with-poly/converted_dataset/tf/",
            "origin_saving_path": "/mnt/e/dataset/work2/Dataset-GEP-FVGN-steady-with-poly/converted_dataset/origin/",
            "flow_type": None,
            "mesh_only": True,
            "saving_tf": True,
            "saving_origin": True,
            "stastic": False,
            "saving_mesh": True,
            "saving_h5": True,
            "print_tf": False,
            "plot": True,
            "flow_type_mapping": flow_type_mapping,
        }

    # stastic total number of data samples
    total_samples = 0
    file_paths = []
    for subdir, _, files in os.walk(path["comsol_dataset_path"]):
        for data_name in files:
            if data_name.endswith(".mphtxt") or data_name.endswith(".dat"):
                file_paths.append(os.path.join(subdir, data_name))

    # 从列表中随机选择200个文件路径
    valid_selected_file_paths = random.sample(file_paths, min(1, len(file_paths)))
    test_selected_file_paths = random.sample(file_paths, min(1, len(file_paths)))

    # 统计选中的文件总数
    total_samples = len(file_paths)
    print("total samples: ", total_samples)

    # split data into train,valid and test
    train_samples = file_paths
    np.random.shuffle(train_samples)
    valid_samples = valid_selected_file_paths
    np.random.shuffle(valid_samples)
    test_samples = test_selected_file_paths
    np.random.shuffle(test_samples)

    # total_samples = [train_samples, valid_samples, test_samples]
    # transform_set = ["train", "valid", "test"]

    total_samples = [train_samples]
    transform_set = ['train']

    # start to convert data
    global_data_index = 0
    train_data_index = -1
    valid_data_index = -1
    test_data_index = -1

    # Using a process pool and a queue
    with multiprocessing.Pool(multi_process) as pool:
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        all_writers = {}

        for writers in transform_set:
            os.makedirs(path["tf_saving_path"], exist_ok=True)
            tf_writer = tf.io.TFRecordWriter(
                path["tf_saving_path"] + f"{writers}.tfrecord"
            )
            os.makedirs(path["origin_saving_path"], exist_ok=True)
            origin_writer = tf.io.TFRecordWriter(
                path["origin_saving_path"] + f"{writers}.tfrecord"
            )
            os.makedirs(path["h5_save_path"], exist_ok=True)
            h5_writer = h5py.File(path["h5_save_path"] + f"{writers}.h5", "w")
            writer_list = [tf_writer, origin_writer, h5_writer]
            all_writers[writers] = writer_list

        # Start writer process
        writer_proc = multiprocessing.Process(
            target=writer_process, args=(queue, all_writers, path)
        )
        writer_proc.start()

        # Process files in parallel
        for dataset_type, sample_list in zip(transform_set, total_samples):
            if debug_file_path is not None:
                # for debuging
                for file_index, file_path in enumerate(sample_list):
                    file_path = debug_file_path
                    process_file(
                        path["plot"], dataset_type, file_index, file_path, path, queue
                    )

                break

            # Start processing processes
            results = [
                pool.apply_async(
                    process_file,
                    args=(
                        path["plot"],
                        dataset_type,
                        file_index,
                        file_path,
                        path,
                        queue,
                    ),
                )
                for file_index, file_path in enumerate(sample_list)
            ]

            # Wait for all processing processes to finish
            for res in results:
                res.get()

        # Send sentinel value to terminate writer process
        queue.put((None, None, None))
        writer_proc.join()

    for subdir, _, files in os.walk(path["tf_saving_path"]):
        for data_name in files:
            if data_name.endswith(".tfrecord"):
                idx_file = f"{os.path.splitext(data_name)[0]}.idx"
                run_command(
                    os.path.join(subdir, data_name), os.path.join(subdir, idx_file)
                )

    print("done")
