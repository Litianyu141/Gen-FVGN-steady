import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

# matplotlib.use("agg")
import pandas as pd
import circle_fit as cf
from circle_fit import hyper_fit
from Utils.utilities import NodeType


def write_array_to_file(field, file_handle):
    """
    将NumPy数组每5个元素为一组写入文件，每组占一行。在文件末尾添加一个换行符。
    
    参数:
    - field: NumPy数组，维度为[10000]，类型为np.float32。
    - filename: 要写入数据的文件名。
    """
    # 确保数组是一维的
    assert field.ndim == 1, "数组必须是一维的"

    # 每5个元素为一组，格式化为字符串并写入一行
    for i in range(0, len(field), 5):
        # 提取当前组的元素，并将其转换为字符串列表
        line = ' '.join(map(str, field[i:i+5]))
        # 写入一行数据
        file_handle.write(line + '\n')
    
    # 在文件最后写入一个换行符
    file_handle.write('\n')
        
def formatnp_vectorized(data):
    """
    Generate appropriate format string for a numpy array, formatting integers and floats differently.
    Groups every three elements into a row, with proper handling for the last row if it has less than three elements.
    
    Arguments:
        - data: a numpy array
    """
    # 替换NaN值为0，并将inf替换为0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    formatted_lines = []  # To store the formatted lines
    for i in range(0, len(data), 3):
        # Select up to 3 elements
        segment = data[i:i+3]
        formatted_segment = []
        for value in segment:
            if np.issubdtype(type(value), np.integer):
                formatted_segment.append("{:d}".format(value))
            else:
                formatted_segment.append("{:e}".format(value))
        formatted_lines.append(" ".join(formatted_segment))
    
    return "\n".join(formatted_lines)


def formatnp(data, file_handle, amounts_per_line=3):
    """
    Write formatted numpy array data to a file, with each line containing a specified number of elements.

    Arguments:
        - data: a list or numpy array of data to write.
        - file_handle: an open file handle for writing.
        - amounts_per_line: the number of data elements per line (default is 3).
    """
    for i in range(len(data)):
        if np.issubdtype(data[i], np.integer):
            file_handle.write(" {:d}".format(data[i].item()))
        else:
            file_handle.write(" {:e}".format(data[i].item()))
        if (i + 1) % amounts_per_line == 0:
            file_handle.write("\n")
    
    # Ensure the file ends with a newline character
    if len(data) % amounts_per_line != 0:
        file_handle.write("\n")


def has_more_than_three_duplicates(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return np.any(counts > 3)


def count_cells_num_node(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return counts


def write_cell_index(Cells, Cells_index, writer):
    # print("start running has_more_than_three_duplicates")
    rtval = has_more_than_three_duplicates(Cells_index)

    FE_num_nodes_counter = 0
    cell_data_to_write = []  # Use a list to accumulate the data to be written for each line
    # print("start writing individual cell index to file")
    for index in range(Cells.shape[0]):
        cell_value_str = str(int(Cells[index]))

        # Always add the current cell's value to the list
        cell_data_to_write.append(cell_value_str)
        FE_num_nodes_counter += 1

        # Check conditions to decide whether to write the current cell's data
        is_last_cell = index == Cells.shape[0] - 1
        if not is_last_cell:
            next_cell_differs = not is_last_cell and Cells_index[index] != Cells_index[index + 1]
        else:
            next_cell_differs = False

        # If this is the last cell or the next cell's index is different, write the data
        if is_last_cell or next_cell_differs:
            # If there are more than three duplicates and not enough nodes, add the current cell again
            if rtval and FE_num_nodes_counter <= 3:
                cell_data_to_write.append(cell_value_str)

            # Write the accumulated cell data as a single space-separated string, then reset
            if cell_data_to_write:  # 如果cell_data_to_write非空
                writer.write(" " + " ".join(cell_data_to_write) + "\n")
            else:  # 如果cell_data_to_write为空
                writer.write("\n")
            cell_data_to_write = []  # Reset for the next group of cells
            FE_num_nodes_counter = 0
            
def write_face_index(faces, writer):
    for index in range(faces.shape[0]):
        formatnp(faces[index], writer, amounts_per_line=2)


def write_poly_face_index(faces, writer):
    amounts_per_line = 10
    write_line = []

    for index in range(faces.shape[0]):
        if len(write_line) >= amounts_per_line:
            concat_line = np.concatenate(write_line, axis=0)
            formatnp(concat_line, writer, amounts_per_line=amounts_per_line)
            write_line = []

        # 添加当前的face到write_line
        write_line.append(faces[index])

    # 检查循环结束后write_line中是否还有数据
    if write_line:
        concat_line = np.concatenate(write_line, axis=0)
        formatnp(concat_line, writer, amounts_per_line=amounts_per_line)
        writer.write("\n")


def detect_var_loacation(fluid_zone=None):
    _VARIABLES_NAME = ["VARIABLES ="]
    _NODAL_VAR = []
    _CELLCENTERED_VAR = []
    _VAR_sequence = 1
    x_dir_velocity_var_location = 0
    y_dir_velocity_var_location = 0
    pressure_var_location = 0
    rho = 1.0  # 默认值
    mu = 0.001  # 默认值

    for key, value in fluid_zone.items():
        count_var = key.split("|")
        if len(count_var) >= 2:
            if count_var[0] == "node":
                _VARIABLES_NAME.append('"node|' + count_var[1] + '"' + "\n")
                _NODAL_VAR.append(key)

            elif count_var[0] == "cell":
                _VARIABLES_NAME.append('"cell|' + count_var[1] + '"' + "\n")
                _CELLCENTERED_VAR.append(key)

            # 检测特殊变量位置（不区分大小写）
            var_name_upper = count_var[1].upper()
            if var_name_upper == "U":
                x_dir_velocity_var_location = _VAR_sequence
            elif var_name_upper == "V":
                y_dir_velocity_var_location = _VAR_sequence
            elif var_name_upper == "P":
                pressure_var_location = _VAR_sequence

            _VAR_sequence += 1

        else:
            # 处理特殊参数
            if key == "rho":
                rho = value
            elif key == "mu":
                mu = value

    return (
        _NODAL_VAR,
        _CELLCENTERED_VAR,
        _VAR_sequence,
        _VARIABLES_NAME,
        x_dir_velocity_var_location,
        y_dir_velocity_var_location,
        pressure_var_location,
        rho,
        mu,
    )


def write_title(file_handle=None, fluid_zone=None):
    file_handle.write('TITLE = "Visualization of the volumetric solution"\n')
    (
        _NODAL_VAR,
        _CELLCENTERED_VAR,
        _VAR_sequence,
        _VARIABLES_NAME,
        x_dir_velocity_var_location,
        y_dir_velocity_var_location,
        pressure_var_location,
        rho,
        mu,
    ) = detect_var_loacation(fluid_zone=fluid_zone)

    file_handle.write(" ".join(_VARIABLES_NAME))

    file_handle.write('DATASETAUXDATA Common.Incompressible="TRUE"\n')
    file_handle.write('DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"\n')
    file_handle.write(f'DATASETAUXDATA Common.Viscosity="{mu}"\n')
    file_handle.write(f'DATASETAUXDATA Common.Density="{rho}"\n')
    
    # 只有在存在U,V,P变量时才写入这些AUXDATA
    if x_dir_velocity_var_location > 0:
        file_handle.write(f'DATASETAUXDATA Common.UVar="{x_dir_velocity_var_location}"\n')
    if y_dir_velocity_var_location > 0:
        file_handle.write(f'DATASETAUXDATA Common.VVar="{y_dir_velocity_var_location}"\n')
    if pressure_var_location > 0:
        file_handle.write(f'DATASETAUXDATA Common.PressureVar="{pressure_var_location}"\n')

    return _NODAL_VAR, _CELLCENTERED_VAR, _VAR_sequence


def write_varlocation_and_datatype(file_handle=None, node_var=None, cell_var=None):
    var_num = len(node_var) + len(cell_var)
    _DT = []
    _VARLOCATION_NODAL = []
    _VARLOCATION_CELLCENTERED = []
    for i in range(1, var_num + 1):
        _DT.append("SINGLE")
        if i <= len(node_var):
            _VARLOCATION_NODAL.append(str(i))
        else:
            _VARLOCATION_CELLCENTERED.append(str(i))
    if len(_VARLOCATION_NODAL) > 0 and len(_VARLOCATION_CELLCENTERED) > 0:
        _VARLOCATION = (
            " VARLOCATION=(["
            + ",".join(_VARLOCATION_NODAL)
            + "]=NODAL,["
            + ",".join(_VARLOCATION_CELLCENTERED)
            + "]=CELLCENTERED)\n"
        )
    elif len(_VARLOCATION_NODAL) > 0:
        _VARLOCATION = " VARLOCATION=([" + ",".join(_VARLOCATION_NODAL) + "]=NODAL)\n"
    elif len(_VARLOCATION_CELLCENTERED) > 0:
        _VARLOCATION = (
            " VARLOCATION=([" + ",".join(_VARLOCATION_CELLCENTERED) + "]=NODAL)\n"
        )

    file_handle.write(f"{_VARLOCATION}")
    _DT = " ".join(_DT)
    file_handle.write(f" DT=({_DT} )\n")


def write_interior_zone(
    file_handle=None, zone=None, node_var=None, cell_var=None, t=None
):
    zonename = zone["name"]
    file_handle.write(f'ZONE T="{zonename}"\n')
    dt = zone["dt"]
    file_handle.write(f" STRANDID=1, SOLUTIONTIME={str(t*dt)}\n")
    X = zone["node|X"]
    cells_node = zone["cells_node"][0]
    cells_index = zone["cells_index"][0]
    face_node = zone["face_node"][0]
    neighbor_cell = zone["neighbor_cell"][0]
    counts = count_cells_num_node(cells_index)
    write_face = False
    
    if counts.max() <= 3:
        file_handle.write(
            f" Nodes={X.size}, Elements={cells_index.max().item()+1}, "
            "ZONETYPE=FETRIANGLE\n"
        )
        write_face = False
    elif 3 < counts.max() <= 4:
        file_handle.write(
            f" Nodes={X.size}, Elements={cells_index.max().item()+1}, "
            "ZONETYPE=FEQuadrilateral\n"
        )
        write_face = False
    elif counts.max() > 4:
        file_handle.write(
            f" Nodes={X.size}, Faces={face_node.shape[0]},Elements={cells_index.max().item()+1}, "
            "ZONETYPE=FEPolygon\n"
        )
        file_handle.write(
            f"NumConnectedBoundaryFaces=0, TotalNumBoundaryConnections=0\n"
        )
        write_face = True

    file_handle.write(" DATAPACKING=BLOCK\n")

    write_varlocation_and_datatype(
        file_handle=file_handle, node_var=node_var, cell_var=cell_var
    )

    field = []
    for var_index, var in enumerate(node_var + cell_var):
        if var == "node|X" or var == "node|Y":
            field.append(zone[var][0, :, 0])
        else:
            # 检查变量是否存在于zone中
            if var in zone:
                if isinstance(zone[var], np.ndarray):
                    if zone[var].ndim >= 3:
                        field.append(zone[var][t, :, 0])
                    elif zone[var].ndim == 2:
                        field.append(zone[var][t, :])
                    else:
                        field.append(zone[var][:])
                else:
                    # 如果是标量值，创建一个与节点数相同的数组
                    num_points = X.size
                    field.append(np.full(num_points, float(zone[var])))
            else:
                # 如果变量不存在，用零填充
                num_points = X.size
                field.append(np.zeros(num_points))
                
    field = np.concatenate(field, axis=0)
    file_handle.write(formatnp_vectorized(field))

    if not write_face:
        write_cell_index(cells_node + 1, zone["cells_index"][0], file_handle)
    else:
        if face_node.min() == 0:
            write_face_node = face_node+1
        if neighbor_cell.min() == 0:
            write_neighbor_cell = neighbor_cell+1

        mask_self_loop = (write_neighbor_cell[:, 0:1] == write_neighbor_cell[:, 1:2]).reshape(-1)
        write_neighbor_cell[mask_self_loop, 0:1] = 0
        file_handle.write("\n# face nodes\n")
        write_poly_face_index(write_face_node, file_handle)
        file_handle.write("\n# left elements\n")
        write_poly_face_index(write_neighbor_cell[:, 0:1], file_handle)
        file_handle.write("\n# right elements\n")
        write_poly_face_index(write_neighbor_cell[:, 1:2], file_handle)


def write_boundary_zone(file_handle=None, zone=None, t=None):
    zonename = zone["name"]
    file_handle.write(f'ZONE T="{zonename}"\n')
    dt = zone["dt"]
    file_handle.write(f" STRANDID=2, SOLUTIONTIME={str(t*dt)}\n")
    X = zone["node|X"]
    face = zone["face"]
    file_handle.write(
        f" Nodes={X.shape[1]}, Elements={face.shape[1]}, ZONETYPE=FELineSeg\n"
    )
    file_handle.write('AUXDATA Common.BoundaryCondition="Wall"\n')
    file_handle.write('AUXDATA Common.IsBoundaryZone="TRUE"\n')
    file_handle.write(" DATAPACKING=BLOCK\n")

    node_var, cell_var, _ = detect_var_loacation(fluid_zone=zone)[:3]

    write_varlocation_and_datatype(
        file_handle=file_handle, node_var=node_var, cell_var=cell_var
    )

    field = []
    for var_index, var in enumerate(node_var + cell_var):
        if var == "node|X" or var == "node|Y":
            field.append(zone[var][0, :, 0])
        else:
            # 检查变量是否存在于zone中
            if var in zone:
                if isinstance(zone[var], np.ndarray):
                    if zone[var].ndim >= 3:
                        field.append(zone[var][t, :, 0])
                    elif zone[var].ndim == 2:
                        field.append(zone[var][t, :])
                    else:
                        field.append(zone[var][:])
                else:
                    # 如果是标量值，创建一个与节点数相同的数组
                    num_points = X.shape[1]
                    field.append(np.full(num_points, float(zone[var])))
            else:
                # 如果变量不存在，用零填充
                num_points = X.shape[1]
                field.append(np.zeros(num_points))

    field = np.concatenate(field, axis=0)
    file_handle.write(formatnp_vectorized(field))
    write_face_index(zone["face"][0] + 1, file_handle)


def write_tecplotzone(
    filename="flowcfdgcn.dat", datasets=None, time_step_length=100
):
    """
    写入Tecplot格式文件，支持多种data_array和多个zone
    
    参数:
    - filename: 输出文件名
    - datasets: 包含zones的列表，每个zone可以是interior或boundary zone
    - time_step_length: 时间步数
    """
    print("writing to tecplot file")

    if not datasets or len(datasets) == 0:
        print("Warning: No datasets provided")
        return

    # 过滤掉None的zones
    valid_zones = [zone for zone in datasets if zone is not None]
    
    if len(valid_zones) == 0:
        print("Warning: No valid zones found")
        return

    # 使用第一个有效zone来写入标题和变量信息
    main_zone = valid_zones[0]

    with open(filename, "w") as file_handle:
        interior_NODAL_VAR, interior_CELLCENTERED_VAR, _VAR_sequence = write_title(
            file_handle=file_handle, fluid_zone=main_zone
        )

        for time_step in range(time_step_length):
            # 写入所有zones
            for zone_idx, zone in enumerate(datasets):
                if zone is None:
                    continue
                    
                # 判断zone类型
                if "face" in zone and isinstance(zone["face"], np.ndarray):
                    # 这是boundary zone
                    write_boundary_zone(
                        file_handle=file_handle, zone=zone, t=time_step
                    )
                else:
                    # 这是interior zone
                    # 为每个zone重新检测变量
                    zone_NODAL_VAR, zone_CELLCENTERED_VAR = detect_var_loacation(fluid_zone=zone)[:2]
                    
                    write_interior_zone(
                        file_handle=file_handle,
                        zone=zone,
                        node_var=zone_NODAL_VAR,
                        cell_var=zone_CELLCENTERED_VAR,
                        t=time_step,
                    )

    print("write done")
    
# Define the function to write data in a subprocess
def write_tecplot_in_subprocess(saving_path, write_zone, time_step_length):
    
    write_tecplotzone(
        saving_path,
        datasets=write_zone,
        time_step_length=time_step_length,
    )
    exit(0)
    
if __name__ == "__main__":
  pass