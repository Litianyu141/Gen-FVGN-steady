import sys
import os
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)
import numpy as np
import torch
from Extract_mesh.parse_to_h5 import (
    NodeType,
)
import os
import math
import random
from contextlib import ExitStack
from math import ceil
from torch_scatter import scatter
import sys
import pyvista as pv
from Post_process.to_vtk import write_hybrid_mesh_to_vtu_2D,write_to_vtk,to_pv_cells_nodes_and_cell_types

# 将输出缓冲区设置为0
sys.stdout.flush()


def string_to_floats(s):
    """将字符串转换为一组浮点数"""
    return np.asarray([float(ord(c)) for c in s])


def floats_to_string(floats):
    """将一组浮点数转换为字符串"""
    return "".join([chr(int(f)) for f in floats])


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
        Determine if a value x is between two other values a and b.

        Parameters:
        - a (float or int): The lower bound.
        - b (float or int): The upper bound.
        - x (float or int): The value to check.

        Returns:
        - (bool): True if x is between a and b (inclusive), False otherwise.
        """
        a = abs(pivot) - float(1e-8)
        b = abs(pivot) + float(1e-8)
        # Check if x is between a and b, inclusive
        if a <= abs(x) <= b:
            return True
        else:
            return False
        
    def convert_to_tensors(self, input_dict: dict) -> dict:
        """Convert dictionary numeric values to PyTorch tensors.
        
        Args:
            input_dict: Dictionary with numpy arrays or numeric values
            
        Returns:
            Dictionary with numeric values converted to PyTorch tensors
        """
        output_dict = {}
        
        try:
            for key, value in input_dict.items():
                # Handle nested dictionaries
                if isinstance(value, dict):
                    output_dict[key] = self.convert_to_tensors(value)
                # Convert numpy arrays    
                elif isinstance(value, np.ndarray):
                    output_dict[key] = torch.from_numpy(value.copy())
                # Convert numeric types
                elif isinstance(value, (int, float, bool, list, tuple)):
                    output_dict[key] = torch.tensor(value)
                # Keep tensors as-is
                elif isinstance(value, torch.Tensor):
                    output_dict[key] = value.clone()
                # Keep strings and other types unchanged
                else:
                    output_dict[key] = value
                    
        except Exception as e:
            raise ValueError(f"Failed to convert to tensor: {str(e)}")

        return output_dict
    
    def convert_to_numpy(self, input_dict: dict) -> dict:
        """Convert dictionary values from PyTorch tensors/numeric types to numpy arrays.
        
        Args:
            input_dict: Dictionary containing PyTorch tensors and other values
            
        Returns:
            Dictionary with tensors and numeric values converted to numpy arrays
        """
        output_dict = {}
        
        try:
            for key, value in input_dict.items():
                # Handle nested dictionaries
                if isinstance(value, dict):
                    output_dict[key] = self.convert_to_numpy(value)
                # Convert PyTorch tensors
                elif isinstance(value, torch.Tensor):
                    output_dict[key] = value.detach().cpu().numpy()
                # Keep numpy arrays as-is    
                elif isinstance(value, np.ndarray):
                    output_dict[key] = value.copy()
                # Convert numeric types to numpy arrays
                elif isinstance(value, (int, float, bool, list, tuple)):
                    output_dict[key] = np.array(value)
                # Keep strings and other types unchanged
                else:
                    output_dict[key] = value
                    
        except Exception as e:
            raise ValueError(f"Failed to convert to numpy: {str(e)}")

        return output_dict
    
    def save_to_vtu(self, mesh:dict, payload:dict, file_name):
        """
        使用 PyVista 写入包含多个顶点和单元数据的 vtu 文件。

        参数:
        - mesh: 字典，包含网格数据，
        - payload: 字典，包含顶点或单元数据，键名以 'node|' 开头表示顶点数据，以 'cell|' 开头表示单元数据
                例如: {'node|temperature': [...], 'cell|pressure': [...]}
        - cells_node: 单元信息，格式为 [顶点数, 顶点1, 顶点2, ...]，例如 [4, 0,1,2,3, 3, 4,5,6]
        - filename: 保存的文件名，默认为 'output.vtu'
        """
        
        # first to tensor
        mesh = self.convert_to_tensors(mesh)
        
        # 暂时先写vtu来可视化
        mesh_pos = mesh["node|pos"].squeeze() if "node|pos" in mesh else mesh["mesh_pos"].squeeze()
        cells_node = mesh["cells_node"].long().squeeze()
        cells_face = mesh["cells_face"].long().squeeze()
        cells_index = mesh["cells_index"].long().squeeze()
         
        pv_cells_node,pv_cells_type = to_pv_cells_nodes_and_cell_types(
            cells_node=cells_node, cells_face=cells_face, cells_index=cells_index
        )
   
        write_hybrid_mesh_to_vtu_2D(
            mesh_pos=mesh_pos.cpu().numpy(), 
            data=payload, 
            cells_node=pv_cells_node.cpu().numpy(), 
            cells_type=pv_cells_type.cpu().numpy(),
            filename=file_name
        )

        print("Mesh saved as", file_name)

