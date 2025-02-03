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
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation
from natsort import natsorted
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


class Cosmol_manager(Basemanager):
    def __init__(
        self,
        mesh_file,
        data_file,
        file_dir=None,
        case_name=None,
        path=None,
    ):

        self.path = path
        self.case_name = case_name
        self.file_dir = file_dir
        self.mesh_file = self.read_mesh_file(mesh_file)

        with open(f"{file_dir}/BC.json", "r") as f:
            self.bc = json.load(f)
            
        for bc_type, bc_index_list in self.bc.items(): 
            if bc_index_list is None or not isinstance(bc_index_list, list):
                continue  # 跳过非列表类型的值

            # >>> 处理 bc_index_list，将范围字符串展开为整数列表 >>>
            def process_item(item):
                if isinstance(item, str) and '-' in item:
                    try:
                        start, end = map(int, item.split('-'))
                        return list(range(start, end + 1))
                    except ValueError:
                        raise ValueError(f"Invalid range format in '{item}' for '{bc_type}'")
                elif isinstance(item, list):
                    return [process_item(sub_item) for sub_item in item]
                else:
                    try:
                        return int(item)
                    except ValueError:
                        raise ValueError(f"Invalid item format in '{item}' for '{bc_type}'")

            processed_list = [process_item(item) for item in bc_index_list]

            self.bc[bc_type] = processed_list
            # <<< 新增代码结束 <<<
        
    def read_mesh_file(self, filename):
        global_dict = {}
        with open(filename, "r") as f:
            lines = f.readlines()

        # Remove leading and trailing whitespaces from each line
        lines = [line.strip() for line in lines]
        idx = 0

        # Find the starting point: '# --------- Object 0 ----------'
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("# --------- Object 0 ----------"):
                idx += 1
                break
            idx += 1
        else:
            raise ValueError("Start of the mesh data not found in the file.")

        # Skip empty lines
        while idx < len(lines) and not lines[idx]:
            idx += 1

        # Read '2 # sdim' to get spatial dimension
        while idx < len(lines):
            line = lines[idx]
            if line.endswith("# sdim"):
                sdim = int(line.split()[0])
                idx += 1
                break
            idx += 1
        else:
            raise ValueError("sdim not found in the file.")

        # Skip empty lines
        while idx < len(lines) and not lines[idx]:
            idx += 1

        # Read number of mesh vertices
        while idx < len(lines):
            line = lines[idx]
            if line.endswith("# number of mesh vertices"):
                num_vertices = int(line.split()[0])
                idx += 1
                break
            idx += 1
        else:
            raise ValueError("Number of mesh vertices not found in the file.")

        # Skip empty lines
        while idx < len(lines) and not lines[idx]:
            idx += 1

        # Read lowest mesh vertex index
        while idx < len(lines):
            line = lines[idx]
            if line.endswith("# lowest mesh vertex index"):
                lowest_vertex_index = int(line.split()[0])
                idx += 1
                break
            idx += 1
        else:
            raise ValueError("Lowest mesh vertex index not found in the file.")

        # Skip empty lines
        while idx < len(lines) and not lines[idx]:
            idx += 1

        # Read '# Mesh vertex coordinates'
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("# Mesh vertex coordinates"):
                idx += 1
                break
            idx += 1
        else:
            raise ValueError("Mesh vertex coordinates not found in the file.")

        # Read vertex coordinates
        vertices = []
        for _ in range(num_vertices):
            while idx < len(lines) and not lines[idx]:
                idx += 1
            if idx >= len(lines):
                raise ValueError("Not enough vertex coordinates in the file.")
            parts = []
            while len(parts) < sdim:
                line = lines[idx]
                idx += 1
                parts.extend(line.strip().split())
            coords = [float(x) for x in parts[:sdim]]
            vertices.append(coords)

        vertices = np.array(vertices)
        if vertices.shape != (num_vertices, sdim):
            raise ValueError("Vertices array has incorrect shape.")
        global_dict["vertices"] = vertices

        # Skip empty lines
        while idx < len(lines) and not lines[idx]:
            idx += 1

        # Read number of element types
        while idx < len(lines):
            line = lines[idx]
            if line.endswith("# number of element types"):
                num_element_types = int(line.split()[0])
                idx += 1
                break
            idx += 1
        else:
            raise ValueError("Number of element types not found in the file.")

        # Read each element type
        for _ in range(num_element_types):
            # Skip empty lines
            while idx < len(lines) and not lines[idx]:
                idx += 1

            # Read element type
            while idx < len(lines):
                line = lines[idx]
                if line.startswith("# Type #"):
                    idx += 1
                    break
                idx += 1
            else:
                raise ValueError("Element type not found in the file.")

            # Skip empty lines
            while idx < len(lines) and not lines[idx]:
                idx += 1

            # Read type name
            line = lines[idx]
            idx += 1
            parts = line.strip().split()
            if len(parts) < 2:
                raise ValueError("Element type name not found.")
            elem_type_name = parts[1]

            # Read number of vertices per element
            while idx < len(lines):
                line = lines[idx]
                if line.endswith("# number of vertices per element"):
                    num_vertices_per_element = int(line.split()[0])
                    idx += 1
                    break
                idx += 1
            else:
                raise ValueError("Number of vertices per element not found.")

            # Skip empty lines
            while idx < len(lines) and not lines[idx]:
                idx += 1

            # Read number of elements
            while idx < len(lines):
                line = lines[idx]
                if line.endswith("# number of elements"):
                    num_elements = int(line.split()[0])
                    idx += 1
                    break
                idx += 1
            else:
                raise ValueError("Number of elements not found.")

            # Skip empty lines and '# Elements' line
            while idx < len(lines) and (not lines[idx] or lines[idx].startswith("#")):
                idx += 1

            # Read elements
            elements = []
            for _ in range(num_elements):
                while idx < len(lines) and not lines[idx]:
                    idx += 1
                if idx >= len(lines):
                    raise ValueError("Not enough elements in the file.")
                parts = []
                while len(parts) < num_vertices_per_element:
                    line = lines[idx]
                    idx += 1
                    parts.extend(line.strip().split())
                # Adjust vertex indices
                element_vertices = [
                    int(x) - lowest_vertex_index
                    for x in parts[:num_vertices_per_element]
                ]

                if len(element_vertices) > 3:
                    # Ensure counter-clockwise ordering
                    element_coords = vertices[element_vertices, :]
                    centroid = np.mean(element_coords, axis=0)
                    vectors = element_coords - centroid
                    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                    sorted_indices = np.argsort(angles)
                    # Reorder element_vertices
                    element_vertices = list(np.array(element_vertices)[sorted_indices])

                elements.append(element_vertices)

            # Skip empty lines
            while idx < len(lines) and not lines[idx]:
                idx += 1

            # Read number of geometric entity indices
            while idx < len(lines):
                line = lines[idx]
                if line.endswith("# number of geometric entity indices"):
                    num_geometric_indices = int(line.split()[0])
                    idx += 1
                    break
                idx += 1
            else:
                raise ValueError("Number of geometric entity indices not found.")

            # Skip empty lines and '# Geometric entity indices' line
            while idx < len(lines) and (not lines[idx] or lines[idx].startswith("#")):
                idx += 1

            # Read geometric entity indices
            geometric_indices = []
            for _ in range(num_geometric_indices):
                while idx < len(lines) and not lines[idx]:
                    idx += 1
                if idx >= len(lines):
                    raise ValueError("Not enough geometric entity indices in the file.")
                line = lines[idx]
                idx += 1
                geometric_indices.append(int(line.strip()))

            # Store the data in the global dictionary
            elem_dict = {
                "Elements": np.array(elements),
                "Geometric entity indices": np.array(geometric_indices)
                + 1,  # offset because comsol GUI indices start from 1
            }
            global_dict[elem_type_name] = elem_dict

        return global_dict

    def set_node_type(
        self,
    ):
        pos = self.mesh_file["vertices"]
        node_type = np.full((pos.shape[0]), NodeType.NORMAL)
        surf_mask = np.full((pos.shape[0]), False)
        periodic_idx = None
        periodic_domain = np.zeros_like(node_type).astype(np.float64)
        
        edge_index = self.mesh_file["edg"]["Elements"]  # [E,2]
        edge_geo_index = self.mesh_file["edg"]["Geometric entity indices"]  # [E]

        for bc_type, bc_index_list in self.bc.items():
            
            if bc_index_list is None:
                continue
            
            if bc_type == "inflow":
                for bc_index in bc_index_list:
                    mask = edge_geo_index == bc_index
                    bc_edge_index = edge_index[mask]
                    node_type[bc_edge_index[:, 0]] = NodeType.INFLOW
                    node_type[bc_edge_index[:, 1]] = NodeType.INFLOW

            elif bc_type == "wall":

                for bc_index in bc_index_list:
                    mask = edge_geo_index == bc_index
                    bc_edge_index = edge_index[mask]
                    # 判断INFLOW和WALL的交界点
                    mask_in_wall_l = node_type[bc_edge_index[:, 0]] == NodeType.INFLOW
                    mask_in_wall_r = node_type[bc_edge_index[:, 1]] == NodeType.INFLOW

                    node_type[bc_edge_index[:, 0]] = NodeType.WALL_BOUNDARY
                    node_type[bc_edge_index[:, 1]] = NodeType.WALL_BOUNDARY
                    
                    node_type[bc_edge_index[mask_in_wall_l, 0]] = NodeType.IN_WALL
                    node_type[bc_edge_index[mask_in_wall_r, 1]] = NodeType.IN_WALL

            elif bc_type == "outflow":
                for bc_index in bc_index_list:
                    mask = edge_geo_index == bc_index
                    bc_edge_index = edge_index[mask]

                    # 判断OUTFLOW和WALL的交界点
                    mask_in_wall_l =node_type[bc_edge_index[:, 0]] == NodeType.WALL_BOUNDARY
                    mask_in_wall_r = node_type[bc_edge_index[:, 1]] == NodeType.WALL_BOUNDARY

                    # 判断INFLOW和OUTFLOW的交界点
                    mask_in_out_l = node_type[bc_edge_index[:, 0]] == NodeType.INFLOW
                    mask_in_out_r = node_type[bc_edge_index[:, 1]] == NodeType.INFLOW

                    node_type[bc_edge_index[:, 0]] = NodeType.OUTFLOW
                    node_type[bc_edge_index[:, 1]] = NodeType.OUTFLOW
                    
                    node_type[bc_edge_index[mask_in_wall_l, 0]] = NodeType.WALL_BOUNDARY
                    node_type[bc_edge_index[mask_in_wall_r, 1]] = NodeType.WALL_BOUNDARY

                    node_type[bc_edge_index[mask_in_out_l, 0]] = NodeType.INFLOW
                    node_type[bc_edge_index[mask_in_out_r, 1]] = NodeType.INFLOW
                    
            elif bc_type == "periodic": # 注意，这里不支持曲线的周期边界，仅支持为直线的周期边界
                for bc_index in bc_index_list:
                    mask_src = (edge_geo_index == bc_index[0])
                    mask_dst = (edge_geo_index == bc_index[1])
                    bc_edge_index_src = edge_index[mask_src]
                    bc_edge_index_dst = edge_index[mask_dst]
                    
                    node_idx_src = np.unique(
                        np.concatenate((bc_edge_index_src[:,0], bc_edge_index_src[:,1]))
                    )
                    node_idx_dst = np.unique(
                        np.concatenate((bc_edge_index_dst[:,0], bc_edge_index_dst[:,1]))
                    )
                    
                    assert len(node_idx_src) == len(node_idx_dst), "仅在节点数相等时支持周期边界"
                    
                    src_pos, dst_pos = pos[node_idx_src], pos[node_idx_dst]
                    
                    # 保证src和dst的一个端点位于直角坐标系原点，且剩余部分位于第一象限
                    src_dist = np.linalg.norm(src_pos, axis=1)
                    src_endpoint_idx = np.argmin(src_dist)
                    shift_src = src_pos[src_endpoint_idx].copy()
                    src_pos -= shift_src
                    src_pos = np.abs(src_pos)
                    
                    # 按照到原点的距离升序排序，并记录对应的原始索引
                    src_dist_all = np.linalg.norm(src_pos, axis=1)
                    src_sorted_idx = np.argsort(src_dist_all)
                    node_idx_src_ascend = node_idx_src[src_sorted_idx]
                    
                    dst_dist = np.linalg.norm(dst_pos, axis=1)
                    dst_endpoint_idx = np.argmin(dst_dist)
                    shift_dst = dst_pos[dst_endpoint_idx].copy()
                    dst_pos -= shift_dst
                    dst_pos = np.abs(dst_pos)
                    
                    dst_dist_all = np.linalg.norm(dst_pos, axis=1)
                    dst_sorted_idx = np.argsort(dst_dist_all)
                    node_idx_dst_ascend = node_idx_dst[dst_sorted_idx]

                    if periodic_idx is None:
                        periodic_idx = np.stack((node_idx_src_ascend, node_idx_dst_ascend), axis=0)
                    else:
                        periodic_idx = np.concatenate((
                            periodic_idx, np.stack((node_idx_src_ascend, node_idx_dst_ascend), axis=0)
                        ), axis=1)
                        
                    ''' >>> 测试周期边界的node_idx_src和node_idx_dst是否正确 >>> '''  
                    
                    src_value = np.sin(pos[periodic_idx[0],0])+np.cos(pos[periodic_idx[0],1])
                    periodic_domain[periodic_idx[0]] += src_value
                    
                    # 尝试将src_value赋值给dst
                    periodic_domain[periodic_idx[1]] = periodic_domain[periodic_idx[0]]
                     
                    ''' <<< 测试周期边界的node_idx_src和node_idx_dst是否正确 <<< ''' 
                    
                    
            elif bc_type == "pressure_point":
                vtk_index = self.mesh_file["vtx"]["Elements"]  # [E]
                vtk_geo_index = self.mesh_file["vtx"]["Geometric entity indices"]  # [E]
                for bc_index in bc_index_list:
                    mask = vtk_geo_index == bc_index
                    bc_node_index = vtk_index[mask]
                    node_type[bc_node_index] = NodeType.PRESS_POINT
                    
            elif bc_type == "surf":
                 for bc_index in bc_index_list:
                    mask = edge_geo_index == bc_index
                    bc_edge_index = edge_index[mask]
                    surf_mask[bc_edge_index[:,0]]=True
                    surf_mask[bc_edge_index[:,1]]=True
                    
        return node_type, surf_mask, periodic_idx, periodic_domain

    def element_to_faces(self, elements, mesh_pos=None):
        
        # # Ensure counter-clockwise ordering
        # element_coords = vertices[element_vertices, :]
        # centroid = np.mean(element_coords, axis=0)
        # vectors = element_coords - centroid
        # angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        # sorted_indices = np.argsort(angles)
        # # Reorder element_vertices
        # element_vertices = list(np.array(element_vertices)[sorted_indices])
        
        N_cells, N_cell_nodes = elements.shape
        edge_unordered = []
        for N_node in range(N_cell_nodes-1):
            edge_unordered.append(
                np.stack((elements[:,N_node],elements[:,N_node+1]),axis=1)
            )
        # add last loop
        edge_unordered.append(
            np.stack((elements[:,-1],elements[:,0]),axis=1)
        )
        edge_unordered = np.stack(edge_unordered,axis=1).reshape(-1,2)
        
        edge_sorted = np.sort(edge_unordered,axis=1).T

        unique_edge,cells_face = np.unique(edge_sorted,axis=1,return_inverse=True)
        
        return unique_edge,cells_face

    def save_mesh_as_vtu(self, global_dict, point_data_dict, filename):
        vertices = global_dict["vertices"]

        # 如果 vertices 是 2D 点，则扩展为 3D
        if vertices.shape[1] < 3:
            vertices = np.hstack(
                [vertices, np.zeros((vertices.shape[0], 3 - vertices.shape[1]))]
            )

        # 定义元素类型到 VTK 单元类型的映射
        elem_type_to_vtk = {"tri": pv.CellType.TRIANGLE, "quad": pv.CellType.QUAD}

        # 创建一个空的 UnstructuredGrid
        combined_grid = pv.UnstructuredGrid()

        # 为每种单元类型创建子网格并合并
        for elem_type in ["tri", "quad"]:
            if elem_type in global_dict:
                elem_data = global_dict[elem_type]
                elements = elem_data["Elements"]
                vtk_cell_type = elem_type_to_vtk[elem_type]

                # 构建 cells 列表
                cells = []
                for elem in elements:
                    cells.extend([len(elem)] + elem.tolist())  # 添加点数量和点索引

                # 转换 cells 为 NumPy 数组
                cells = np.array(cells, dtype=np.int64)
                cell_types = np.full(len(elements), vtk_cell_type, dtype=np.uint8)

                # 使用点和单元创建子网格
                sub_grid = pv.UnstructuredGrid(cells, cell_types, vertices)

                # 将子网格合并到组合网格中
                combined_grid = combined_grid.merge(sub_grid)

        for key, point_point_data in point_data_dict.items():
            combined_grid.point_data[key] = point_point_data

        # 保存合并的网格为 VTU 文件
        combined_grid.save(filename)
        
        print("Mesh saved as", filename)
  
    def extract_mesh(self, mesh_only=True):

        node_type, surf_mask, periodic_idx, periodic_domain = self.set_node_type()

        self.save_mesh_as_vtu(
            global_dict=self.mesh_file,
            point_data_dict={
                "node_type": node_type[:, None],
                "periodic_domain": periodic_domain[:, None]
            },
            filename=f"{self.file_dir}/node_type_with_mesh.vtu",
        )
        
        # fmt: off
        """ compose cells_node, cells_index and edge_index"""
        cells_node = []
        cells_index = []
        face_node = []
        cells_face = []
        count_cells = 0
        count_faces = 0
        for elem_type in ["tri", "quad"]:
            if elem_type in self.mesh_file:
                elem_data = self.mesh_file[elem_type]
                elements = elem_data["Elements"]
                cells_node.append(elements.reshape(-1))
                cell_index = (
                    np.arange(count_cells, count_cells+elements.shape[0])[:, None]
                    .repeat(axis=1, repeats=elements.shape[1])
                    .reshape(-1,1)
                )
                cells_index.append(cell_index)
                count_cells+=elements.shape[0]

                unique_edge, cell_face = self.element_to_faces(elements=elements)
                face_node.append(unique_edge)
                cells_face.append(cell_face+count_faces)
                count_faces+=unique_edge.shape[1]
                
        cells_node = np.concatenate(cells_node).squeeze()
        cells_index = np.concatenate(cells_index).squeeze()
        face_node = np.concatenate(face_node,axis=1).squeeze()
        cells_face = np.concatenate(cells_face,axis=0).squeeze()
        """ compose cells_node, cells_index and edge_index"""
        # fmt: on
        
        if surf_mask.any():
            surf_edge_index, _ = filter_adj(
                edge_index=face_node,
                perm=np.arange(self.mesh_file["vertices"].shape[0])[surf_mask],
                num_nodes=self.mesh_file["vertices"].shape[0],
            )
            
            write_vtp_file(
                mesh_pos = self.mesh_file["vertices"][surf_mask],
                edge_index=surf_edge_index,
                output_filename=f"{self.file_dir}/surf_edge.vtp",
            )
        
        if mesh_only:
            mesh = {
                "node|pos": torch.from_numpy(self.mesh_file["vertices"]),
                "node|surf_mask": torch.from_numpy(surf_mask).bool(),
                "node|node_type": torch.from_numpy(node_type).long(),
                "face|face_node": torch.from_numpy(face_node).long(),
                "cells_node": torch.from_numpy(cells_node).long(),
                "cells_index": torch.from_numpy(cells_index).long(),
                "cells_face": torch.from_numpy(cells_face).long(),
                "periodic_idx": torch.from_numpy(periodic_idx).long(),
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
    if file_path.endswith(".mphtxt"):
        data = Cosmol_manager(
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
    # debug_file_path = "datasets/Tayler-Green/mesh.mphtxt"

    case = 0  # 0 stands for 980/PM9A1
    if case == 0:
        path = {
            "simulator": "COMSOL",
            "comsol_dataset_path": "datasets/Taylor_Green",
            "mesh_only": True,
        }

    # stastic total number of data samples
    total_samples = 0
    file_paths = []
    for subdir, _, files in os.walk(path["comsol_dataset_path"]):
        for data_name in files:
            if data_name.endswith(".mphtxt") or data_name.endswith(".dat"):
                file_paths.append(os.path.join(subdir, data_name))

    # 统计选中的文件总数
    assert total_samples == 0, "Found no mesh files"
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
            results = process_file(
                        0,
                        debug_file_path,
                        path,
                        queue,
                    ),
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
