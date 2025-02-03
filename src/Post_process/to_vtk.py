import torch
import numpy as np
import os
from os import path as osp
import pyvista as pv
import trimesh
import trimesh.sample as sample
import vtk
import h5py
from torch_geometric.data import Data


def to_pv_cells_nodes_and_cell_types(cells_node:torch.Tensor,cells_face:torch.Tensor,cells_index:torch.Tensor):
    
    from Extract_mesh.parse_to_h5 import seperate_domain
    
    # 暂时先写vtk来可视化
    domain_list = seperate_domain(cells_node, cells_face, cells_index)
    
    pv_cells_node=[]
    pv_cells_type=[]
    for domain in domain_list:
        
        _ct, _cells_node, _, _ = domain
        _cells_node = _cells_node.reshape(-1,_ct)
        num_cells = _cells_node.shape[0]
        _cells_node = torch.cat(
            (torch.full((_cells_node.shape[0],1),_ct), _cells_node),
            dim=1,
        ).reshape(-1)
        pv_cells_node.append(_cells_node)
        
        # 根据顶点数设置单元类型（3=三角形, 4=四边形, >4=多边形）
        if _ct == 3:
            cells_types = torch.full((num_cells,),pv.CellType.TRIANGLE)
        elif _ct == 4:
            cells_types = torch.full((num_cells,),pv.CellType.QUAD)
        else:
            cells_types = torch.full((num_cells,),pv.CellType.POLYGON)
        pv_cells_type.append(cells_types)
        
    pv_cells_node = torch.cat(pv_cells_node,dim=0) 
    pv_cells_type = torch.cat(pv_cells_type,dim=0)
    
    return pv_cells_node,pv_cells_type
        
# trainning ply data(num_points:3586, num_cells:7186)
def load_mesh_ply_vtk(file_path):

    mesh = pv.read(file_path)
    points = mesh.points
    cells_vtk = list(mesh.cell)

    ## Print Info
    # prt_fmt = 'process file:[{}], num_points:{}, num_cells:{}'
    # print(prt_fmt.format(file_path, points.shape[0], len(cells_vtk)))
    cells = []

    for cell_vtk in cells_vtk:
        cell = []
        for id in range(cell_vtk.GetNumberOfPoints()):
            cell.append(cell_vtk.GetPointId(id))
        cells.append(cell)

    points = np.array(points)
    cells = np.array(cells)

    return points, cells


# 读取 .vtk 文件
def read_vtk(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


# 将四边形转换为三角形
def convert_quads_to_tris(unstructured_grid):
    # 使用 vtkGeometryFilter 将 UnstructuredGrid 转换为 PolyData
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_grid)
    geometry_filter.Update()

    poly_data = geometry_filter.GetOutput()

    # 构建三角形拓扑
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(poly_data)
    triangle_filter.Update()

    return triangle_filter.GetOutput()


# 计算法向量并添加为顶点数据
def compute_and_add_normals(poly_data):
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(poly_data)
    normal_generator.ComputePointNormalsOn()
    normal_generator.ComputeCellNormalsOff()
    normal_generator.Update()

    normals = normal_generator.GetOutput().GetPointData().GetNormals()

    return normals


def get_points(poly_data):
    points = poly_data.GetPoints()
    num_points = points.GetNumberOfPoints()
    points_array = np.zeros((num_points, 3))

    for i in range(num_points):
        points_array[i, :] = points.GetPoint(i)

    return points_array


def get_pressure_data(poly_data):
    pressure_array = poly_data.GetPointData().GetArray("point_scalars")
    num_points = poly_data.GetNumberOfPoints()

    if pressure_array is None:
        raise ValueError("Pressure data not found in the input VTK file.")

    pressure = np.zeros((num_points, 1))
    for i in range(num_points):
        pressure[i, 0] = pressure_array.GetValue(i)

    return pressure


def get_velocity_data(poly_data):
    point_data = poly_data.GetPointData()
    num_arrays = point_data.GetNumberOfArrays()

    for i in range(num_arrays):
        array = point_data.GetArray(i)
        if array.GetNumberOfComponents() == 3:  # 检查是否为3维向量数据
            # 将数据转换为Numpy数组
            velocity_field = np.array(
                [array.GetTuple(j) for j in range(array.GetNumberOfTuples())]
            )
            print(f"Found velocity field with array name: {array.GetName()}")
            return velocity_field

    raise ValueError("No 3D velocity field found in the VTK file.")


def extract_vertices_and_cells(polydata):
    # 获取顶点坐标
    points = polydata.GetPoints()
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    # 获取单元索引
    cells = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        cell_points = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        cells.append(cell_points)
    cells = np.array(cells)

    return vertices, cells, polydata


def extract_triangle_indices(poly_data):
    poly_data.BuildLinks()
    num_cells = poly_data.GetNumberOfCells()
    triangle_indices = []

    for cell_id in range(num_cells):
        cell = poly_data.GetCell(cell_id)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            point_ids = cell.GetPointIds()
            indices = [point_ids.GetId(i) for i in range(3)]
            triangle_indices.append(indices)

    return np.array(triangle_indices)


def write_to_vtk(data: dict, write_file_path):
    # dict :{'node|pos'         : position}
    #       {'node|sth'         : node_value}
    #       {'cell|cells_node'  : cells}
    #       {'cell|sth'         : cell_value}

    grid = vtk.vtkUnstructuredGrid()

    # process node
    points = data["node|pos"]
    points_vtk = vtk.vtkPoints()
    [points_vtk.InsertNextPoint(point) for point in points]
    grid.SetPoints(points_vtk)
    point_data = grid.GetPointData()
    for key in data.keys():
        if not key.startswith("node"):
            continue
        if key == "node|pos":
            continue
        array_data = data[key]
        vtk_data_array = vtk.vtkFloatArray()
        k = (
            1
            if type(array_data[0]) is np.float64 or type(array_data[0]) is np.float32
            else len(array_data[0])
        )
        vtk_data_array.SetNumberOfComponents(k)

        if k == 1:
            [vtk_data_array.InsertNextTuple([value]) for value in array_data]
        else:
            [vtk_data_array.InsertNextTuple(value) for value in array_data]

        vtk_data_array.SetName(key)
        point_data.AddArray(vtk_data_array)

    cells = data["cells_node"].reshape(-1, 3)

    # # process cell
    cell_array = vtk.vtkCellArray()
    for cell in cells:
        triangle = vtk.vtkTriangle()
        for i, id in enumerate(cell):
            triangle.GetPointIds().SetId(i, id)
        cell_array.InsertNextCell(triangle)
    grid.SetCells(vtk.vtkTriangle().GetCellType(), cell_array)

    cell_data = grid.GetCellData()

    for key in data.keys():
        if not key.startswith("cell|"):
            continue
        if key == "cell|cells_node":
            continue
        array_data = data[key]
        vtk_data_array = vtk.vtkFloatArray()
        k = (
            1
            if type(array_data[0]) is np.float64 or type(array_data[0]) is np.float32
            else len(array_data[0])
        )
        vtk_data_array.SetNumberOfComponents(k)

        if k == 1:
            [vtk_data_array.InsertNextTuple([value]) for value in array_data]
        else:
            [vtk_data_array.InsertNextTuple(value) for value in array_data]
        vtk_data_array.SetName(key)
        cell_data.AddArray(vtk_data_array)
    # 将网格保存为 VTU 文件
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(write_file_path)
    writer.SetInputData(grid)
    writer.Write()
    print(f"vtu file saved:{write_file_path}")


def write_point_cloud_to_vtk(data: dict, write_file_path):
    # dict :{'node|pos'         : position}
    #       {'node|sth'         : node_value}

    grid = vtk.vtkUnstructuredGrid()

    # process node
    points = data["node|pos"]
    points_vtk = vtk.vtkPoints()
    [points_vtk.InsertNextPoint(point) for point in points]
    grid.SetPoints(points_vtk)
    point_data = grid.GetPointData()
    for key in data.keys():
        if not key.startswith("node"):
            continue
        if key == "node|pos":
            continue
        array_data = data[key]
        vtk_data_array = vtk.vtkFloatArray()
        k = (
            1
            if type(array_data[0]) is np.float64 or type(array_data[0]) is np.float32
            else len(array_data[0])
        )
        vtk_data_array.SetNumberOfComponents(k)

        if k == 1:
            [vtk_data_array.InsertNextTuple([value]) for value in array_data]
        else:
            [vtk_data_array.InsertNextTuple(value) for value in array_data]

        vtk_data_array.SetName(key)
        point_data.AddArray(vtk_data_array)

    # 将网格保存为 VTU 文件
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(write_file_path)
    writer.SetInputData(grid)
    writer.Write()
    print(f"vtu file saved:[{write_file_path}]")


def write_vtu_file_3D(mesh_pos, cells, point_data_dict, file_path):
    # 创建UnstructuredGrid对象
    unstructured_grid = vtk.vtkUnstructuredGrid()

    # 设置点（顶点）数据
    points = vtk.vtkPoints()
    for pos in mesh_pos:
        points.InsertNextPoint(pos)
    unstructured_grid.SetPoints(points)

    # 设置单元（面片）数据
    for cell in cells:
        cell_id_list = vtk.vtkIdList()
        for point_id in cell:
            cell_id_list.InsertNextId(point_id)
        unstructured_grid.InsertNextCell(vtk.VTK_POLYGON, cell_id_list)

    # 设置点数据（例如法向量或其他数据）
    if point_data_dict is not None:
        for name, data_array in point_data_dict.items():
            vtk_data_array = vtk.vtkDoubleArray()
            vtk_data_array.SetNumberOfComponents(
                len(data_array[0])
            )  # 根据数据维度设置组件数
            vtk_data_array.SetName(name)  # 使用字典的键作为数据数组的名称
            for data in data_array:
                vtk_data_array.InsertNextTuple(data)
            unstructured_grid.GetPointData().AddArray(vtk_data_array)

    # 写入VTU文件
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(unstructured_grid)
    writer.Write()


def write_hybrid_mesh_to_vtu_2D(mesh_pos, data, cells_node, cells_type=None, filename="output.vtu"):
    """
    使用 PyVista 写入包含多个顶点和单元数据的 vtu 文件。

    参数:
    - mesh_pos: 二维网格坐标 (numpy 数组或 torch 张量, 形状为 [N, 2])
    - data: 字典，包含顶点或单元数据，键名以 'node|' 开头表示顶点数据，以 'cell|' 开头表示单元数据
            例如: {'node|temperature': [...], 'cell|pressure': [...]}
    - cells_node: 单元信息，格式为 [顶点数, 顶点1, 顶点2, ...]，例如 [4, 0,1,2,3, 3, 4,5,6]
    - cells_type: （可选）单元类型列表，如果不提供，将根据顶点数自动推断
    - filename: 保存的文件名，默认为 'output.vtu'
    """

    # 确保输入为 numpy 数组
    if not isinstance(mesh_pos, np.ndarray):
        mesh_pos = mesh_pos.numpy()
    
    # 转换 mesh_pos 为 3D 坐标 (添加 z 坐标)
    if mesh_pos.shape[1] == 2:
        points = np.c_[mesh_pos, np.zeros(mesh_pos.shape[0])]
    else:
        points = mesh_pos
    
    # 构建 cells 数据
    if cells_type is None:
        offset = 0
        cells_type = []
        cells = []
        n = len(cells_node)
        while offset < n:
            num_points = cells_node[offset]
            cell_points = cells_node[offset + 1 : offset + 1 + num_points]
            cells.append(np.hstack(([num_points], cell_points)))
            
            # 根据顶点数设置单元类型（3=三角形, 4=四边形, >4=多边形）
            if num_points == 3:
                cells_type.append(pv.CellType.TRIANGLE)
            elif num_points == 4:
                cells_type.append(pv.CellType.QUAD)
            else:
                cells_type.append(pv.CellType.POLYGON)
            
            offset += 1 + num_points
        cells = np.concatenate(cells)
    else:
        cells = cells_node

    # 创建 PyVista 的 UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cells_type, points)
    
    # 添加顶点和单元数据
    for key, values in data.items():
        if key.startswith("node|"):
            # 添加顶点数据，去掉前缀 'node|'
            grid.point_data[key] = np.array(values)
        elif key.startswith("cell|"):
            # 添加单元数据，去掉前缀 'cell|'
            grid.cell_data[key] = np.array(values)
        else:
            print(f"警告: 未知的数据前缀 {key}，将忽略该数据。")
    
    # 写入到 vtu 文件
    grid.save(filename)
    print(f"文件已保存到 {filename}")
    
    
# ############### Grid SDF ####################


def write_vtu_file_2D_quad(mesh_pos, cells, point_data_dict, file_path):
    # 创建UnstructuredGrid对象
    unstructured_grid = vtk.vtkUnstructuredGrid()

    # 设置点（顶点）数据
    points = vtk.vtkPoints()
    for pos in mesh_pos:
        points.InsertNextPoint(pos)
    unstructured_grid.SetPoints(points)

    # 设置单元（四边形面片）数据
    for cell in cells:
        if len(cell) == 4:  # 确保每个单元都是四边形
            cell_id_list = vtk.vtkIdList()
            for point_id in cell:
                cell_id_list.InsertNextId(point_id)
            unstructured_grid.InsertNextCell(vtk.VTK_QUAD, cell_id_list)
        else:
            print(f"Warning: Skipping cell {cell} because it does not have 4 vertices.")

    # 设置点数据（例如法向量或其他数据）
    if point_data_dict is not None:
        for name, data_array in point_data_dict.items():
            vtk_data_array = vtk.vtkDoubleArray()
            vtk_data_array.SetNumberOfComponents(
                len(data_array[0])
            )  # 根据数据维度设置组件数
            vtk_data_array.SetName(name)  # 使用字典的键作为数据数组的名称
            for data in data_array:
                vtk_data_array.InsertNextTuple(data)
            unstructured_grid.GetPointData().AddArray(vtk_data_array)

    # 写入VTU文件
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    print("VTU has been saved to:", file_path)


def write_vtu_file_2D_quad_subprocess(
    vis_pos, graph_node_valid, pred_np, target_np, logger, epoch, current_files_name
):
    """
    子进程中调用的函数，用于调用 write_vtu_file_2D_quad
    """
    write_vtu_file_2D_quad(
        mesh_pos=vis_pos,
        cells=graph_node_valid.face.mT.cpu().numpy(),
        point_data_dict={
            "node|u": pred_np[:, 0:1],
            "node|v": pred_np[:, 1:2],
            "node|p": pred_np[:, 2:3],
            "node|nut": pred_np[:, 3:4],
            "node|u_true": target_np[:, 0:1],
            "node|v_true": target_np[:, 1:2],
            "node|p_true": target_np[:, 2:3],
            "node|nut_true": target_np[:, 3:4],
        },
        file_path=f"{logger.valid_visualization}/valid{epoch}_{current_files_name}.vtu",
    )
    exit(0)


def add_point_data_and_save_vtu(unstructured_grid, data_dict, filepath):
    """
    将额外的顶点数据添加到 vtkUnstructuredGrid 对象中，并保存为 .vtu 文件。
    支持标量和向量数据。

    参数:
    unstructured_grid (vtkUnstructuredGrid): 原始的非结构化网格对象
    data_dict (dict): 包含要添加的顶点数据的字典。键是数据名称，值是与顶点数量相同的数组。
    filepath (str): 保存 .vtu 文件的完整路径

    返回:
    vtkUnstructuredGrid: 更新后的非结构化网格对象
    """
    num_points = unstructured_grid.GetNumberOfPoints()

    for data_name, data_array in data_dict.items():
        data_array = np.array(data_array)

        # 确保数据数组的第一维与顶点数量相匹配
        if data_array.shape[0] != num_points:
            raise ValueError(
                f"数据 '{data_name}' 的长度 ({data_array.shape[0]}) 与网格顶点数 ({num_points}) 不匹配"
            )

        # 确定数据的维度
        if data_array.ndim == 1:
            num_components = 1
        elif data_array.ndim == 2:
            num_components = data_array.shape[1]
        else:
            raise ValueError(f"数据 '{data_name}' 的维度不正确。应为1维或2维数组。")

        # 创建适当类型的 VTK 数组
        if data_array.dtype == np.float64 or data_array.dtype == np.float32:
            vtk_array = vtk.vtkFloatArray()
        elif np.issubdtype(data_array.dtype, np.integer):
            vtk_array = vtk.vtkIntArray()
        else:
            raise ValueError(f"不支持的数据类型: {data_array.dtype}")

        vtk_array.SetName(data_name)
        vtk_array.SetNumberOfComponents(num_components)

        # 将数据添加到 VTK 数组
        if num_components == 1:
            for value in data_array:
                vtk_array.InsertNextValue(value)
        else:
            for point_data in data_array:
                vtk_array.InsertNextTuple(point_data)

        # 将 VTK 数组添加到网格的点数据中
        unstructured_grid.GetPointData().AddArray(vtk_array)

    # 保存为 .vtu 文件
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    print(f"文件已保存至: {filepath}")

    return unstructured_grid


def normalize_points(points, bounds):
    """
    Normalize points to a cube defined by [-1, 1] in each dimension.

    Parameters:
    points (numpy.ndarray): An array of shape (N, 3) representing the points.
    bounds (numpy.ndarray): An array of shape (2, 3) representing the min and max bounds for x, y, z.

    Returns:
    numpy.ndarray: The normalized points.
    """
    # Extract min and max bounds
    min_bounds = bounds[0]
    max_bounds = bounds[1]

    # Calculate the center and half range for each dimension
    center = (min_bounds + max_bounds) / 2.0
    half_range = (max_bounds - min_bounds) / 2.0

    # Normalize points to [-1, 1]
    normalized_points = (points - center) / half_range

    return normalized_points


def compute_mean_std(data):
    mean = 0.0
    std = 0.0
    n_samples = 0

    for x in data:
        x = x.reshape(-1, 1)
        n_samples += x.shape[0]
        mean += x.sum(0)

    mean /= n_samples

    for x in data:
        x = x.reshape(-1, 1)
        std += ((x - mean) ** 2).sum(0)

    std = torch.sqrt(std / n_samples)

    return mean.to(torch.float32), std.to(torch.float32)


def compute_mean_std_3dvector(data):
    normals = np.concatenate(data, axis=0)

    # 计算均值向量
    mean_vector = np.mean(normals, axis=0)

    # 计算方差
    variance_vector = np.var(normals, axis=0)

    return torch.from_numpy(mean_vector).to(torch.float32), torch.from_numpy(
        variance_vector
    ).to(torch.float32)


def dict2Device(data: dict, device):
    for key, v in data.items():
        data[key] = v.to(torch.float32).to(device)
    return data


def compute_ao(ply_file, n_samples=64):
    # model = trimesh.Trimesh(points=points,  faces=cells, force="mesh")
    model = trimesh.load(ply_file, force="mesh")
    # model = trimesh.load("untitled.ply", force="mesh")
    assert isinstance(model, trimesh.Trimesh)
    # model.fix_normals()

    # how many rays to send out from each vertex
    NDIRS = n_samples
    # how far away do surfaces still block the light?
    # relative to the model diagonal
    RELSIZE = 0.05

    sphere_pts, _ = sample.sample_surface_even(trimesh.primitives.Sphere(), count=NDIRS)

    normal_dir_similarities = model.vertex_normals @ sphere_pts.T
    assert normal_dir_similarities.shape[0] == len(model.vertex_normals)
    assert normal_dir_similarities.shape[1] == len(sphere_pts)

    normal_dir_similarities[normal_dir_similarities <= 0] = 0
    normal_dir_similarities[normal_dir_similarities > 0] = 1

    vert_idxs, dir_idxs = np.where(normal_dir_similarities)
    del normal_dir_similarities

    normals = model.vertex_normals[vert_idxs]
    origins = model.vertices[vert_idxs] + normals * model.scale * 0.0005
    directions = sphere_pts[dir_idxs]
    assert len(origins) == len(directions)
    # print("origins", origins[:100])
    # print("directions", directions[:100])

    hit_pts, idxs_rays, _ = model.ray.intersects_location(
        ray_origins=origins, ray_directions=directions
    )

    # don't check infinitely long rays
    succ_origs = origins[idxs_rays]
    distances = np.linalg.norm(succ_origs - hit_pts, axis=1)
    idxs_rays = idxs_rays[distances < RELSIZE * model.scale]

    idxs_orig = vert_idxs[idxs_rays]
    uidxs, uidxscounts = np.unique(idxs_orig, return_counts=True)
    assert len(uidxs) == len(uidxscounts)

    counts_verts = np.zeros(len(model.vertices))
    counts_verts[uidxs] = uidxscounts
    counts_verts = counts_verts / np.max(counts_verts) * 255

    counts_verts = 255 - counts_verts.astype(int).reshape(-1, 1)

    AO = counts_verts / np.full_like(counts_verts, 255.0)
    return AO


def write_airfrans_vtu(pred, gt=None, case_name=None, save_path=None, params=None):

    vtu_path = os.path.dirname(params.validset)
    vtu_path = os.path.join(
        vtu_path, f"../original_datasets/Dataset/{case_name}/{case_name}_internal.vtu"
    )
    grid = pv.read(vtu_path)

    assert grid.n_points == pred.shape[0]

    # 将 pred 和 gt 添加为点数据
    grid.point_data["Pred"] = pred
    if gt is not None:
        grid.point_data["GT"] = gt

    # 保存带有预测和真值的新 vtu 文件
    output_file = save_path
    grid.save(output_file)
    print(f"vtu viz saved:{output_file}")
    pass


def write_vtp_file(mesh_pos, edge_index, output_filename):
    # 创建 vtkPoints 对象并将顶点坐标添加进去
    points = vtk.vtkPoints()
    for pos in mesh_pos:
        points.InsertNextPoint(pos[0], pos[1], 0)  # 假设 z = 0，创建二维点

    # 创建 vtkCellArray 对象并将线条（edges）添加进去
    lines = vtk.vtkCellArray()
    for edge in edge_index.T:  # edge_index 形状为 [2, E]，需要转置以遍历每条边
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])  # 边的第一个顶点
        line.GetPointIds().SetId(1, edge[1])  # 边的第二个顶点
        lines.InsertNextCell(line)

    # 创建 vtkPolyData 对象
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)

    # 使用 vtkXMLPolyDataWriter 写入到 .vtp 文件
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(poly_data)
    writer.Write()
    print(f"VTP 文件已保存到：{output_filename}")


if __name__ == "__main__":
    # 示例调用
    h5_file_path = (
        "/lvm_data/litianyu/mycode-new/CIKM_car_race/datasets/conveted_dataset/test.h5"
    )