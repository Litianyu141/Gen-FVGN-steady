import os
import sys

sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
from ast import Pass
from turtle import circle
import torch
import numpy as np
import pickle
import enum
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import pandas as pd
import circle_fit as cf
from circle_fit import hyper_fit


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



def triangles_to_faces(faces, deform=False):
    """Computes mesh edges from triangles."""
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
        unique_edges = torch.stack((receivers, senders), dim=1)
        return {
            "two_way_connectivity": two_way_connectivity,
            "senders": senders,
            "receivers": receivers,
            "unique_edges": unique_edges,
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
        if np.issubsctype(data[i], np.integer):
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

    for key, value in fluid_zone.items():
        count_var = key.split("|")
        if len(count_var) >= 2:
            if count_var[0] == "node":
                _VARIABLES_NAME.append('"' + count_var[1] + '"' + "\n")
                _NODAL_VAR.append(key)

            elif count_var[0] == "cell":
                _VARIABLES_NAME.append('"' + count_var[1] + '"' + "\n")
                _CELLCENTERED_VAR.append(key)

            if count_var[1] == "U":
                x_dir_velocity_var_location = _VAR_sequence
            elif count_var[1] == "V":
                y_dir_velocity_var_location = _VAR_sequence
            elif count_var[1] == "P":
                pressure_var_location = _VAR_sequence

            _VAR_sequence += 1

        else:
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
    # _VARIABLES_NAME = ['VARIABLES =']
    # _NODAL_VAR = []
    # _CELLCENTERED_VAR = []
    # _VAR_sequence = 1
    # x_dir_velocity_var_location = 0
    # y_dir_velocity_var_location = 0
    # # z_dir_velocity_var_location = 0
    # pressure_var_location = 0

    # for key,value in fluid_zone.items():
    #     count_var = key.split('|')
    #     if len(count_var)>=2:
    #         if count_var[0]=="node":
    #             _VARIABLES_NAME.append("\""+count_var[1]+"\""+"\n")
    #             _NODAL_VAR.append(key)

    #         elif count_var[0]=="cell":
    #             _VARIABLES_NAME.append("\""+count_var[1]+"\""+"\n")
    #             _CELLCENTERED_VAR.append(key)

    #         if count_var[1] == "U":
    #             x_dir_velocity_var_location=_VAR_sequence
    #         elif count_var[1] == "V":
    #             y_dir_velocity_var_location=_VAR_sequence
    #         elif count_var[1] == "P":
    #             pressure_var_location = _VAR_sequence

    #         _VAR_sequence+=1

    #     else:
    #         if key=="rho":
    #             rho = value
    #         elif key=="mu":
    #             mu=value

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
    file_handle.write(f'DATASETAUXDATA Common.UVar="{x_dir_velocity_var_location}"\n')
    file_handle.write(f'DATASETAUXDATA Common.VVar="{y_dir_velocity_var_location}"\n')
    file_handle.write(f'DATASETAUXDATA Common.PressureVar="{pressure_var_location}"\n')

    # _VARLOCATION = " VARLOCATION=(["+','.join(_VARLOCATION_NODAL)+"],["+','.join(_VARLOCATION_CELLCENTERED)+"])\n"

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
    neighbour_cell = zone["neighbour_cell"][0]
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
            field.append(zone[var][t, :, 0])
    field = np.concatenate(field, axis=0)
    file_handle.write(formatnp_vectorized(field))

    if not write_face:
        write_cell_index(cells_node + 1, zone["cells_index"][0], file_handle)
    else:
        if face_node.min() == 0:
            face_node += 1
        if neighbour_cell.min() == 0:
            neighbour_cell += 1

        mask_self_loop = (neighbour_cell[:, 0:1] == neighbour_cell[:, 1:2]).reshape(-1)
        neighbour_cell[mask_self_loop, 0:1] = 0
        file_handle.write("# face nodes\n")
        write_poly_face_index(face_node, file_handle)
        file_handle.write("# left elements\n")
        write_poly_face_index(neighbour_cell[:, 0:1], file_handle)
        file_handle.write("# right elements\n")
        write_poly_face_index(neighbour_cell[:, 1:2], file_handle)


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

    node_var, cell_var, _, _, _, _, _, _, _ = detect_var_loacation(fluid_zone=zone)

    write_varlocation_and_datatype(
        file_handle=file_handle, node_var=node_var, cell_var=cell_var
    )

    field = []
    for var_index, var in enumerate(node_var + cell_var):
        if var == "node|X" or var == "node|Y":
            field.append(zone[var][0, :, 0])
        else:
            field.append(zone[var][t, :, 0])

    field = np.concatenate(field, axis=0)
    file_handle.write(formatnp_vectorized(field))
    write_face_index(zone["face"][0] + 1, file_handle)


def write_tecplotzone_test(
    filename="flowcfdgcn.dat", datasets=None, time_step_length=100
):
    # print("writing to tecplot file")

    interior_zone, boundary_zone = datasets[0], datasets[1]

    with open(filename, "w") as file_handle:
        interor_NODAL_VAR, interor_CELLCENTERED_VAR, _VAR_sequence = write_title(
            file_handle=file_handle, fluid_zone=interior_zone
        )

        # write_varlocation_and_datatype(file_handle=file_handle,node_var=_NODAL_VAR,cell_var=_CELLCENTERED_VAR)

        for time_step in range(time_step_length):
            write_interior_zone(
                file_handle=file_handle,
                zone=interior_zone,
                node_var=interor_NODAL_VAR,
                cell_var=interor_CELLCENTERED_VAR,
                t=time_step,
            )

            if boundary_zone:
                write_boundary_zone(
                    file_handle=file_handle, zone=boundary_zone, t=time_step
                )

    # print("write done")


def write_tecplotzone(
    filename="flowcfdgcn.dat",
    datasets=None,
    time_step_length=100,
    has_cell_centered=False,
    synatic=False,
):
    interior_zone = datasets[0]
    mu = interior_zone["mu"]
    rho = interior_zone["rho"]

    with open(filename, "w") as f:
        f.write('TITLE = "Gen-FVGN-train solution"\n')
        f.write('VARIABLES = "X"\n"Y"\n"U"\n"V"\n"P"\n')
        f.write('DATASETAUXDATA Common.Incompressible="TRUE"\n')
        f.write('DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"\n')
        f.write(f'DATASETAUXDATA Common.Viscosity="{mu}"\n')
        f.write(f'DATASETAUXDATA Common.Density="{rho}"\n')
        f.write(f'DATASETAUXDATA Common.UVar="3"\n')
        f.write(f'DATASETAUXDATA Common.VVar="4"\n')
        f.write(f'DATASETAUXDATA Common.PressureVar="5"\n')

        for i in range(time_step_length):
            for zone in datasets:
                zonename = zone["zonename"]
                if zonename == "Fluid":
                    f.write('ZONE T="{0}"\n'.format(zonename))
                    X = zone["mesh_pos"][i, :, 0]
                    Y = zone["mesh_pos"][i, :, 1]
                    U = zone["velocity"][i, :, 0]
                    V = zone["velocity"][i, :, 1]
                    P = zone["pressure"][i, :, 0]
                    field = np.concatenate((X, Y, U, V, P), axis=0)
                    Cells = zone["cells"][i, :, :] + 1
                    Cells_index = zone["cells_index"][i, :, :]
                    face_node = zone["face"][i]
                    neighbour_cell = zone["neighbour_cell"][i]

                    counts = count_cells_num_node(Cells_index)
                    write_face = False
                    if counts.max() <= 3:
                        f.write(
                            f" Nodes={X.size}, Elements={Cells_index.max().item()+1}, "
                            "ZONETYPE=FETRIANGLE\n"
                        )
                        write_face = False
                    elif 3 < counts.max() <= 4:
                        f.write(
                            f" Nodes={X.size}, Elements={Cells_index.max().item()+1}, "
                            "ZONETYPE=FEQuadrilateral\n"
                        )
                        write_face = False
                    elif counts.max() > 4:
                        f.write(
                            f" Nodes={X.size}, Faces={face_node.shape[1]},Elements={Cells_index.max().item()+1}, "
                            "ZONETYPE=FEPolygon\n"
                        )
                        f.write(
                            f"NumConnectedBoundaryFaces=0, TotalNumBoundaryConnections=0\n"
                        )
                        write_face = True

                    f.write(" DATAPACKING=BLOCK\n")
                    data_packing_type = zone["data_packing_type"]
                    if data_packing_type == "cell":
                        f.write(" VARLOCATION=([3,4,5]=CELLCENTERED)\n")
                    elif data_packing_type == "node":
                        f.write(" VARLOCATION=([3,4,5]=NODAL)\n")
                    f.write(" DT=(SINGLE SINGLE SINGLE SINGLE SINGLE)\n")
                    try:
                        # print(f"start writing interior field data, size in {field.size},shape in {field.shape}")
                        write_array_to_file(field, f)
                    except Exception as e:
                        raise ValueError(f"Error formatting data: {e}")
                        
                    # print("Start writing interior cell")
                    if not write_face:
                        write_cell_index(Cells, Cells_index, f)
                    else:
                        if face_node.min() == 0:
                            write_face_node = face_node + 1
                        if neighbour_cell.min() == 0:
                            write_neighbour_cell = neighbour_cell + 1
                        if write_face_node.shape[0] <= 2:
                            write_face_node = write_face_node.T
                        if write_neighbour_cell.shape[0] <= 2:
                            write_neighbour_cell = write_neighbour_cell.T
                        mask_self_loop = (
                            write_neighbour_cell[:, 0:1] == write_neighbour_cell[:, 1:2]
                        ).reshape(-1)
                        write_neighbour_cell[mask_self_loop, 0:1] = 0
                        f.write("# face nodes\n")
                        write_poly_face_index(write_face_node, f)
                        f.write("# left elements\n")
                        write_poly_face_index(write_neighbour_cell[:, 0:1], f)
                        f.write("# right elements\n")
                        write_poly_face_index(write_neighbour_cell[:, 1:2], f)
                    # print("End writing interior cell")
                    
                elif (
                    zonename == "OBSTICALE_BOUNDARY" or zonename.find("BOUNDARY") != -1
                ):
                    f.write('ZONE T="{0}"\n'.format(zonename))
                    X = zone["mesh_pos"][i, :, 0]
                    Y = zone["mesh_pos"][i, :, 1]
                    U = zone["velocity"][i, :, 0]
                    V = zone["velocity"][i, :, 1]
                    P = zone["pressure"][i, :, 0]
                    field = np.concatenate((X, Y, U, V, P), axis=0)
                    faces = zone["face"][i, :, :] + 1

                    f.write(" STRANDID=2, SOLUTIONTIME={0}\n".format(0.01 * i))
                    f.write(
                        f" Nodes={X.size}, Elements={faces.shape[0]}, "
                        "ZONETYPE=FELineSeg\n"
                    )
                    f.write(" DATAPACKING=BLOCK\n")
                    f.write('AUXDATA Common.BoundaryCondition="Wall"\n')
                    f.write('AUXDATA Common.IsBoundaryZone="TRUE"\n')
                    data_packing_type = zone["data_packing_type"]
                    if data_packing_type == "cell":
                        f.write(" VARLOCATION=([3,4,5]=CELLCENTERED)\n")
                    elif data_packing_type == "node":
                        f.write(" VARLOCATION=([3,4,5]=NODAL)\n")
                    f.write(" DT=(SINGLE SINGLE SINGLE SINGLE SINGLE )\n")
                    
                    # print("start writing boundary field data")
                    write_array_to_file(field, f)
                    
                    # print("start writing boundary face")
                    write_face_index(faces, f)
                    
    print("saved tecplot file at " + filename)

def write_tecplotzone_in_process(save_path, datasets, time_step_length, has_cell_centered, synatic):
    # 假设 write_tec 是已经导入的模块，包含write_tecplotzone方法
    write_tecplotzone(
        filename=save_path,
        datasets=datasets,
        time_step_length=time_step_length,
        has_cell_centered=has_cell_centered,
        synatic=synatic,
    )
    exit(0)

def write_tecplot_ascii_nodal(raw_data, is_tfrecord, pkl_path, saving_path):
    cylinder_pos = []
    cylinder_velocity = []
    cylinder_pressure = []
    cylinder_index = []
    cylinder = {}
    if is_tfrecord:
        dataset = raw_data
    else:
        with open(pkl_path, "rb") as fp:
            dataset = pickle.load(fp)
    for j in range(600):
        new_pos_dict = {}
        mesh_pos = dataset["mesh_pos"][j]
        coor_y = dataset["mesh_pos"][j, :, 1]
        mask_F = np.full(coor_y.shape, False)
        mask_T = np.full(coor_y.shape, True)
        node_type = dataset["node_type"][j, :, 0]
        mask_of_coor = np.where(
            (node_type == NodeType.WALL_BOUNDARY)
            & (coor_y > np.min(coor_y))
            & (coor_y < np.max(coor_y)),
            mask_T,
            mask_F,
        )
        mask_of_coor_index = np.argwhere(
            (node_type == NodeType.WALL_BOUNDARY)
            & (coor_y > np.min(coor_y))
            & (coor_y < np.max(coor_y))
        )
        cylinder_x = dataset["mesh_pos"][j, :, 0][mask_of_coor]
        cylinder_u = dataset["velocity"][j, :, 0][mask_of_coor]
        cylinder_y = coor_y[mask_of_coor]
        cylinder_v = dataset["velocity"][j, :, 1][mask_of_coor]
        cylinder_p = dataset["pressure"][j, :, 0][mask_of_coor]
        coor = np.stack((cylinder_x, cylinder_y), axis=-1)
        cylinder_speed = np.stack((cylinder_u, cylinder_v), axis=-1)

        cylinder_pos.append(coor)
        cylinder_velocity.append(cylinder_speed)
        cylinder_pressure.append(cylinder_p)

        for index in range(coor.shape[0]):
            new_pos_dict[str(coor[index])] = index
        cells_node = torch.from_numpy(dataset["cells"][j]).to(torch.int32)
        decomposed_cells = triangles_to_faces(cells_node)
        senders = decomposed_cells["senders"]
        receivers = decomposed_cells["receivers"]
        mask_F = np.full(senders.shape, False)
        mask_T = np.full(senders.shape, True)
        mask_index_s = np.isin(senders, mask_of_coor_index)
        mask_index_r = np.isin(receivers, mask_of_coor_index)

        mask_index_of_face = np.where((mask_index_s) & (mask_index_r), mask_T, mask_F)

        senders = senders[mask_index_of_face]
        receivers = receivers[mask_index_of_face]
        senders_f = []
        receivers_f = []
        for i in range(senders.shape[0]):
            senders_f.append(new_pos_dict[str(mesh_pos[senders[i]])])
            receivers_f.append(new_pos_dict[str(mesh_pos[receivers[i]])])
        cylinder_boundary_face = np.stack(
            (np.asarray(senders_f), np.asarray(receivers_f)), axis=-1
        )
        cylinder_index.append(cylinder_boundary_face)
    dataset["zonename"] = "Fluid"
    flow_zone = dataset
    cylinder["zonename"] = "Cylinder_Boundary"
    cylinder["mesh_pos"] = np.asarray(cylinder_pos)
    cylinder["velocity"] = np.asarray(cylinder_velocity)
    cylinder["pressure"] = np.expand_dims(np.asarray(cylinder_pressure), -1)
    cylinder["face"] = np.asarray(cylinder_index)
    cylinder_zone = cylinder
    tec_saving_path = saving_path
    write_tecplotzone(tec_saving_path, [flow_zone, cylinder_zone])


def rearrange_dict(zone):
    """transform dict to list, so pandas dataframe can handle it properly"""
    dict_list = []
    build = False
    for k, v in zone.items():
        if k == "zonename" or k == "mean_u" or k == "relonyds_num" or k == "cylinder_D":
            continue
        if v.shape[2] > 1:
            for j in range(v.shape[2]):
                for index in range(zone["mesh_pos"].shape[0]):
                    dict_new = {}
                    if (
                        k == "zonename"
                        or k == "mean_u"
                        or k == "relonyds_num"
                        or k == "cylinder_D"
                    ):
                        continue
                    elif k == "centroid" or k == "cell_area":
                        dict_new[k + str(j)] = v[0][:, j]
                    else:
                        dict_new[k + str(j)] = v[index][:, j]
                    if not build:
                        dict_list.append(dict_new)
                build = True
                for index in range(zone["mesh_pos"].shape[0]):
                    if (
                        k == "zonename"
                        or k == "mean_u"
                        or k == "relonyds_num"
                        or k == "cylinder_D"
                    ):
                        continue
                    elif k == "centroid" or k == "cell_area":
                        dict_list[index][k + str(j)] = v[0][:, j]
                    else:
                        dict_list[index][k + str(j)] = v[index][:, j]
        else:
            for index in range(zone["mesh_pos"].shape[0]):
                if (
                    k == "zonename"
                    or k == "mean_u"
                    or k == "relonyds_num"
                    or k == "cylinder_D"
                ):
                    continue
                elif k == "centroid" or k == "cell_area":
                    dict_list[index][k] = v[0][:, 0]
                else:
                    dict_list[index][k] = v[index][:, 0]
        build = True
    return dict_list



def write_tecplot_ascii_cell_centered(
    raw_data,
    saving_path,
    plot_boundary_p_time_step=None,
    save_tec=False,
    plot_boundary=False,
):
    # boundary zone
    if save_tec or plot_boundary_p_time_step is not None:
        cylinder_zone = extract_boundary_thread(
            raw_data=raw_data, save_path=saving_path, plot_boundary=plot_boundary
        )

    # plot boundary zone
    if plot_boundary_p_time_step is not None:
        plot_boundary_pressure(cylinder_zone, saving_path, plot_boundary_p_time_step)

    # write interior zone and boundary zone to tecplot file
    raw_data["zonename"] = "Fluid"
    flow_zone = raw_data
    tec_saving_path = saving_path
    if save_tec:
        write_tecplotzone(
            filename=tec_saving_path,
            datasets=[flow_zone, cylinder_zone],
            time_step_length=raw_data["velocity"].shape[0],
            has_cell_centered=True,
        )
