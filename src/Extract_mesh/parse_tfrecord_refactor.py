# 2 -*- encoding: utf-8 -*-
"""
@File    :   parse_tfrecord.py
@Author  :   litianyu 
@Version :   1.0
@Contact :   lty1040808318@163.com
"""
# 解析tfrecord数据
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import functools
import json
import numpy as np
import h5py
import pickle
import enum
import multiprocessing as mp
import time
import logging
import torch
from torch_scatter import scatter
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch_geometric.transforms as T
from write_tec import write_tecplot_ascii_nodal
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation

from circle_fit import hyper_fit

c_NORMAL_max = 0
c_OBSTACLE_max = 0
c_AIRFOIL_max = 0
c_HANDLE_max = 0
c_INFLOW_max = 0
c_OUTFLOW_max = 0
c_WALL_BOUNDARY_max = 0
c_SIZE_max = 0

c_NORMAL_min = 3000
c_OBSTACLE_min = 3000
c_AIRFOIL_min = 3000
c_HANDLE_min = 3000
c_INFLOW_min = 3000
c_OUTFLOW_min = 3000
c_WALL_BOUNDARY_min = 3000
c_SIZE_min = 3000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("done loading packges")
marker_cell_sp1 = []
marker_cell_sp2 = []

mask_of_mesh_pos_sp1 = np.arange(5233)
mask_of_mesh_pos_sp2 = np.arange(5233)

inverse_of_marker_cell_sp1 = np.arange(1)
inverse_of_marker_cell_sp2 = np.arange(1)

new_mesh_pos_iframe_sp1 = np.arange(1)
new_mesh_pos_iframe_sp2 = np.arange(1)

new_node_type_iframe_sp1 = np.arange(1)
new_node_type_iframe_sp2 = np.arange(1)

pos_dict_1 = {}
pos_dict_2 = {}
switch_of_redress = True


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


case = 4
# 980PRO
if case == 0:
    path = {
        "tf_datasetPath": "H:/repo-datasets/meshgn/airfoil",
        "pickl_save_path": "H:/repo-datasets/meshgn/airfoil/pickle/airfoilonlyvsp1.pkl",
        "h5_save_path": "H:/repo-datasets/meshgn/h5/airfoil",
        "tfrecord_sp": "H://repo-datasets/meshgn/airfoil/tfrecord_splited/",
        "mesh_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/",
        "tec_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/",
        "saving_tec": False,
        "stastic": True,
        "saving_mesh": True,
        "mask_features": True,
        "saving_sp_tf": True,
        "saving_sp_tf_single": False,
        "saving_sp_tf_mp": True,
        "saving_pickl": False,
        "saving_h5": False,
        "h5_sep": False,
        "print_tf": False,
    }
    model = {"name": "airfoil", "maxepoch": 10}
elif case == 1:
    # jtedu-lidequan1967
    path = {
        "tf_datasetPath": "/root/meshgraphnets/datasets/",
        "pickl_save_path": "/root/meshgraphnets/datasets/pickle/airfoilonlyvsp1.pkl",
        "h5_save_path": "/root/meshgraphnets/datasets/h5/",
        "tfrecord_sp": "/root/meshgraphnets/datasets/tfrecord_splited/",
        "mesh_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/",
        "tec_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/",
        "saving_tec": False,
        "stastic": True,
        "saving_mesh": True,
        "mask_features": True,
        "saving_sp_tf": True,
        "saving_sp_tf_single": False,
        "saving_sp_tf_mp": True,
        "saving_pickl": False,
        "saving_h5": False,
        "h5_sep": False,
        "print_tf": False,
    }
    model = {"name": "airfoil", "maxepoch": 10}
elif case == 2:
    # jtedu-DOOMDUKE2
    path = {
        "tf_datasetPath": "/root/meshgraphnets/datasets/cylinder_flow/cylinder_flow/",
        "pickl_save_path": "/root/meshgraphnets/datasets/cylinder_flow/pickle/airfoil/cylinder_flowonlyvsp1.pkl",
        "h5_save_path": "/root/meshgraphnets/datasets/cylinder_flow/h5",
        "tfrecord_sp": "/root/meshgraphnets/datasets/cylinder_flow/tfrecord_splited/",
        "mesh_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/",
        "tec_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/meshs/",
        "saving_tec": False,
        "stastic": True,
        "saving_mesh": True,
        "mask_features": False,
        "saving_sp_tf": True,
        "saving_sp_tf_single": False,
        "saving_sp_tf_mp": True,
        "saving_pickl": False,
        "saving_h5": False,
        "h5_sep": False,
        "print_tf": False,
        "plot_order": True,
    }
    model = {"name": "airfoil", "maxepoch": 10}
elif case == 3:
    # centre`s DL machine
    path = {
        "tf_datasetPath": "/data/litianyu/dataset/MeshGN/cylinder_flow/origin_dataset",
        "pickl_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/cylinder_flow.pkl",
        "h5_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/mesh_with_ghost_state/",
        "tfrecord_sp": "/data/litianyu/dataset/MeshGN/cylinder_flow/tfrecord_splited/",
        "mesh_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/mesh_with_ghost_state/",
        "tec_save_path": "/home/litianyu/mycode/repos-py/FVM/my_FVNN/rollouts/tecplot/",
        "mode": "cylinder_mesh",
        "renum_origin_dataset": False,
        "saving_tec": False,
        "stastic": False,
        "saving_mesh": False,
        "mask_features": False,
        "saving_sp_tf": False,
        "saving_sp_tf_single": False,
        "saving_sp_tf_mp": False,
        "saving_pickl": False,
        "saving_h5": True,
        "h5_sep": False,
        "print_tf": False,
        "plot_order": False,
    }
    model = {"name": "cylinder_flow", "maxepoch": 10}
elif case == 4:
    # 980-wsl-linux
    path = {
        "tf_datasetPath": "/mnt/f/repo-datasets/meshgn/airfoil",
        "pickl_save_path": "/mnt/f/repo-datasets/meshgn/airfoilonlyvsp1.pkl",
        "h5_save_path": "/mnt/f/repo-datasets/meshgn/",
        "tfrecord_sp": "/mnt/f/repo-datasets/meshgn/",
        "mesh_save_path": "/mnt/f/repo-datasets/meshgn/",
        "tec_save_path": "/mnt/f/repo-datasets/meshgn/",
        "mode": "cylinder_mesh",
        "renum_origin_dataset": False,
        "saving_tec": True,
        "stastic": False,
        "saving_mesh": False,
        "mask_features": False,
        "saving_sp_tf": False,
        "saving_sp_tf_single": False,
        "saving_sp_tf_mp": False,
        "saving_pickl": False,
        "saving_h5": False,
        "h5_sep": False,
        "print_tf": False,
        "plot_order": False,
    }

    model = {"name": "airfoil", "maxepoch": 10}


class parser:
    @staticmethod
    def mask_features(datasets, features: str, mask_factor):
        """mask_factor belongs between [0,1]"""
        datasets["target|" + features] = datasets[features]
        for frame_index in range(datasets[features].shape[0]):
            if features == "velocity":
                shape = datasets[features][frame_index].shape
                pre_mask = np.arange(shape[0])
                choosen_pos = np.random.choice(pre_mask, int(shape[0] * mask_factor))
                for i in range(choosen_pos.shape[0]):
                    datasets[features][frame_index][choosen_pos[i]] = torch.zeros(
                        (1, shape[1]), dtype=torch.float64
                    )
                """mask = np.random.randint(0, 2, size=shape[0])
        mask = np.expand_dims(mask,1).repeat(2,axis = 1)
        masked_velocity_frame = np.multiply(mask,datasets[features][frame_index])"""
        return datasets


def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    """for index,frame in enumerate(ds):
      data = _parse(frame, meta)"""
    ds = ds.prefetch(1)
    return ds


def dividing_line(index, x):
    if index == 0:
        return x
    else:
        return 0.1 * index * x


def stastic_nodeface_type(frame):
    if len(frame.shape) > 1:
        flatten = frame[:, 0]
    else:
        flatten = frame
    c_NORMAL = flatten[flatten == NodeType.NORMAL].shape[0]
    c_OBSTACLE = flatten[flatten == NodeType.OBSTACLE].shape[0]
    c_AIRFOIL = flatten[flatten == NodeType.AIRFOIL].shape[0]
    c_HANDLE = flatten[flatten == NodeType.HANDLE].shape[0]
    c_INFLOW = flatten[flatten == NodeType.INFLOW].shape[0]
    c_OUTFLOW = flatten[flatten == NodeType.OUTFLOW].shape[0]
    c_WALL_BOUNDARY = flatten[flatten == NodeType.WALL_BOUNDARY].shape[0]
    c_SIZE = flatten[flatten == NodeType.SIZE].shape[0]
    c_GHOST_INFLOW = flatten[flatten == NodeType.GHOST_INFLOW].shape[0]
    c_GHOST_OUTFLOW = flatten[flatten == NodeType.GHOST_OUTFLOW].shape[0]
    c_GHOST_WALL_BOUNDARY = flatten[flatten == NodeType.GHOST_WALL].shape[0]
    c_GHOST_AIRFOIL = flatten[flatten == NodeType.GHOST_AIRFOIL].shape[0]
    # for i in range(flatten.shape[0]):
    #       if(flatten[i]==NodeType.NORMAL):
    #             c_NORMAL+=1
    #       elif(flatten[i]==NodeType.OBSTACLE):
    #             c_OBSTACLE+=1
    #       elif(flatten[i]==NodeType.AIRFOIL):
    #             c_AIRFOIL+=1
    #       elif(flatten[i]==NodeType.HANDLE):
    #             c_HANDLE+=1
    #       elif(flatten[i]==NodeType.INFLOW):
    #             c_INFLOW+=1
    #       elif(flatten[i]==NodeType.OUTFLOW):
    #             c_OUTFLOW+=1
    #       elif(flatten[i]==NodeType.WALL_BOUNDARY):
    #             c_WALL_BOUNDARY+=1
    #       elif(flatten[i]==NodeType.SIZE):
    #             c_SIZE+=1
    #       elif(flatten[i]==NodeType.GHOST_INFLOW):
    #             c_GHOST_INFLOW+=1
    #       elif(flatten[i]==NodeType.GHOST_OUTFLOW):
    #             c_GHOST_OUTFLOW+=1
    #       elif(flatten[i]==NodeType.GHOST_WALL):
    #             c_GHOST_WALL+=1
    #       elif(flatten[i]==NodeType.GHOST_AIRFOIL):
    #             c_GHOST_AIRFOIL+=1
    print(
        "NORMAL: {0} OBSTACLE: {1} AIRFOIL: {2} HANDLE: {3} INFLOW: {4} OUTFLOW: {5} WALL_BOUNDARY: {6} SIZE: {7} GHOST_INFLOW: {8} GHOST_OUTFLOW: {9} GHOST_WALL_BOUNDARY: {10} GHOST_AIRFOIL: {11}".format(
            c_NORMAL,
            c_OBSTACLE,
            c_AIRFOIL,
            c_HANDLE,
            c_INFLOW,
            c_OUTFLOW,
            c_WALL_BOUNDARY,
            c_SIZE,
            c_GHOST_INFLOW,
            c_GHOST_OUTFLOW,
            c_GHOST_WALL_BOUNDARY,
            c_GHOST_AIRFOIL,
        )
    )
    rtval = np.zeros(int(max(NodeType)) + 1).astype(np.int64)
    for node_type in enumerate(NodeType):
        try:
            rtval[node_type[0]] = eval(
                "c_" + str(node_type[1]).replace("NodeType.", "")
            )
        except:
            rtval[node_type[0]] = 0
    return rtval


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

        elif flatten[i] == NodeType.OBSTACLE:
            c_OBSTACLE += 1

        elif flatten[i] == NodeType.AIRFOIL:
            c_AIRFOIL += 1

        elif flatten[i] == NodeType.HANDLE:
            c_HANDLE += 1

        elif flatten[i] == NodeType.INFLOW:
            c_INFLOW += 1

        elif flatten[i] == NodeType.OUTFLOW:
            c_OUTFLOW += 1

        elif flatten[i] == NodeType.WALL_BOUNDARY:
            c_WALL_BOUNDARY += 1

        elif flatten[i] == NodeType.SIZE:
            c_SIZE += 1

    c_NORMAL_max = max(c_NORMAL_max, c_NORMAL)
    c_NORMAL_min = min(c_NORMAL_min, c_NORMAL)
    c_OBSTACLE_max = max(c_OBSTACLE_max, c_OBSTACLE)
    c_OBSTACLE_min = min(c_OBSTACLE_min, c_OBSTACLE)
    c_AIRFOIL_max = max(c_AIRFOIL_max, c_AIRFOIL)
    c_OBSTACLE_min = min(c_AIRFOIL_min, c_AIRFOIL)
    c_HANDLE_max = max(c_HANDLE_max, c_HANDLE)
    c_HANDLE_min = min(c_HANDLE_min, c_HANDLE)
    c_INFLOW_max = max(c_INFLOW_max, c_INFLOW)
    c_INFLOW_min = min(c_INFLOW_min, c_INFLOW)
    c_OUTFLOW_max = max(c_OUTFLOW_max, c_OUTFLOW)
    c_OUTFLOW_min = min(c_OUTFLOW_min, c_OUTFLOW)
    c_WALL_BOUNDARY_max = max(c_WALL_BOUNDARY_max, c_WALL_BOUNDARY)
    c_WALL_BOUNDARY_min = min(c_WALL_BOUNDARY_min, c_WALL_BOUNDARY)
    c_SIZE_max = max(c_SIZE_max, c_SIZE)
    c_SIZE_min = min(c_SIZE_min, c_SIZE)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(record, mode="airfoil"):
    feature = {}
    for key, value in record.items():
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.tobytes()])
        )

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()

    # if mode=='airfoil':
    #   feature = {
    #       "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
    #       "cells": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells'].tobytes()])),
    #       "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
    #       "density": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['density'].tobytes()])),
    #       "pressure": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['pressure'].tobytes()])),
    #       "velocity": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['velocity'].tobytes()]))
    #   }
    #   example = tf.train.Example(features=tf.train.Features(feature=feature))
    #   return example.SerializeToString()
    # elif mode=='cylinder_flow':
    #   feature = {
    #       "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
    #       "cells": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells'].tobytes()])),
    #       "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
    #       "pressure": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['pressure'].tobytes()])),
    #       "velocity": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['velocity'].tobytes()]))
    #   }
    #   example = tf.train.Example(features=tf.train.Features(feature=feature))
    #   return example.SerializeToString()
    # elif mode == 'cylinder_mesh':
    #   feature = {
    #       "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
    #       "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
    #       "cells_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_node'].tobytes()])),
    #       "cell_factor": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cell_factor'].tobytes()])),
    #       "centroid": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['centroid'].tobytes()])),
    #       "face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face'].tobytes()])),
    #       "face_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_type'].tobytes()])),
    #       "face_length": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_length'].tobytes()])),
    #       "neighbour_cell": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['neighbour_cell'].tobytes()])),
    #       "cells_face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_face'].tobytes()])),
    #       "cells_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_type'].tobytes()])),
    #       # "BOUNDARY_CELL_cells_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['BOUNDARY_CELL_cells_type'].tobytes()])),
    #       "cells_area": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_area'].tobytes()])),
    #       "unit_norm_v": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['unit_norm_v'].tobytes()])),
    #       "target|velocity_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|velocity_on_node'].tobytes()])),
    #       "target|pressure_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|pressure_on_node'].tobytes()])),
    #       'mean_u': tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mean_u'].tobytes()])),
    #       'cylinder_diameter': tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cylinder_diameter'].tobytes()]))
    #   }
    #   example = tf.train.Example(features=tf.train.Features(feature=feature))
    #   return example.SerializeToString()
    # elif mode == 'airfoil_mesh':
    #   feature = {
    #       "node_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['node_type'].tobytes()])),
    #       "mesh_pos": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['mesh_pos'].tobytes()])),
    #       "cells_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_node'].tobytes()])),
    #       "cell_factor": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cell_factor'].tobytes()])),
    #       "centroid": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['centroid'].tobytes()])),
    #       "face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face'].tobytes()])),
    #       "face_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_type'].tobytes()])),
    #       "face_length": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['face_length'].tobytes()])),
    #       "neighbour_cell": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['neighbour_cell'].tobytes()])),
    #       "cells_face": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_face'].tobytes()])),
    #       "cells_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_type'].tobytes()])),
    #       # "BOUNDARY_CELL_cells_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['BOUNDARY_CELL_cells_type'].tobytes()])),
    #       "cells_area": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['cells_area'].tobytes()])),
    #       "unit_norm_v": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['unit_norm_v'].tobytes()])),
    #       "target|velocity_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|velocity_on_node'].tobytes()])),
    #       "target|pressure_on_node": tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['target|pressure_on_node'].tobytes()]))
    #   }
    #   example = tf.train.Example(features=tf.train.Features(feature=feature))
    #   return example.SerializeToString()


def write_tfrecord_one(tfrecord_path, records, mode):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        serialized = serialize_example(records, mode=mode)
        writer.write(serialized)


def write_tfrecord_one_with_writer(writer, records, mode):
    serialized = serialize_example(records, mode=mode)
    writer.write(serialized)


def write_tfrecord(tfrecord_path, records, np_index):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for index, frame in enumerate(records):
            serialized = serialize_example(frame)
            writer.write(serialized)
            print("process:{0} is writing traj:{1}".format(np_index, index))


def write_tfrecord_mp(tfrecord_path_1, tfrecord_path_2, records):
    procs = []
    npc = 0
    n_shard = 2
    for shard_id in range(n_shard):
        if shard_id == 0:
            args = (tfrecord_path_1, records[0], npc)
        elif shard_id == 1:
            args = (tfrecord_path_2, records[1], npc + 1)
        p = mp.Process(target=write_tfrecord, args=args)
        p.start()
        procs.append(p)

    for proc in procs:
        proc.join()


def find_pos(mesh_point, mesh_pos_sp1):
    for k in range(mesh_pos_sp1.shape[0]):
        if (mesh_pos_sp1[k] == mesh_point).all():
            print("found{}".format(k))
            return k
    return False


def convert_to_tensors(input_dict):
    # 遍历字典中的所有键
    for key in input_dict.keys():
        # 检查值的类型
        value = input_dict[key]
        if isinstance(value, np.ndarray):
            # 如果值是一个Numpy数组，使用torch.from_numpy进行转换
            input_dict[key] = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            # 如果值不是一个PyTorch张量，使用torch.tensor进行转换
            input_dict[key] = torch.tensor(value)
        # 如果值已经是一个PyTorch张量，不进行任何操作

    # 返回已更新的字典
    return input_dict


def plot_mesh(
    mesh_pos,
    edge_index,
    node_type,
    face_center_pos,
    face_type,
    centroid,
    unit_normal_vector=None,
    other_cell_centered_vector=None,
    plot_mesh=False,
    path=None,
):
    mesh_pos = np.array(mesh_pos)
    edge_index = np.array(edge_index)

    # prepare for plot file saving
    mesh_file_path = path["mesh_file_path"]
    subdir = os.path.dirname(mesh_file_path)
    mesh_name = os.path.basename(mesh_file_path)
    fig_dir = f"{subdir}/mesh{''.join(char for char in mesh_name if char.isdigit())}"
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.cla()
    ax.set_aspect("equal")

    # 通过索引获取每一条边的两个点的坐标
    point1 = mesh_pos[edge_index[:, 0]]
    point2 = mesh_pos[edge_index[:, 1]]

    # 将每一对点的坐标合并，方便绘图
    lines = np.hstack([point1, point2]).reshape(-1, 2, 2)

    # 使用plot绘制所有的边
    if plot_mesh:
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
            centroid[:, 0],
            centroid[:, 1],
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

    if other_cell_centered_vector is not None:
        ax.quiver(
            centroid[:, 0],
            centroid[:, 1],
            other_cell_centered_vector[:, 0],
            other_cell_centered_vector[:, 1],
            units="height",
            color="cyan",
            angles="xy",
            scale_units="xy",
            scale=75,
            width=0.01,
            headlength=3,
            headwidth=2,
            headaxislength=3.5,
        )

    # 显示图形
    plt.show()
    plt.savefig(f"{fig_dir}/node distribution.png")
    plt.close()


def polygon_area(vertices):
    """
    使用shoelace formula（鞋带公式）来计算多边形的面积。
    :param vertices: 多边形的顶点坐标，一个二维numpy数组。
    :return: 多边形的面积。
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def plot_all_boundary_sate(mesh, path):
    mesh_file_path = path["mesh_file_path"]
    subdir = os.path.dirname(mesh_file_path)
    mesh_name = os.path.basename(mesh_file_path)
    fig_dir = f"{subdir}/mesh{''.join(char for char in mesh_name if char.isdigit())}"
    os.makedirs(fig_dir, exist_ok=True)

    mesh_pos = mesh["mesh_pos"]
    node_type = mesh["node_type"].long().view(-1)
    cells_node = mesh["cells_node"].long()
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.cla()
    ax.set_aspect("equal")
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells_node.view(-1, 3))
    ax.triplot(triang, "k-", ms=0.5, lw=0.3)
    plt.scatter(
        mesh_pos[node_type == NodeType.NORMAL, 0],
        mesh_pos[node_type == NodeType.NORMAL, 1],
        c="red",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        mesh_pos[node_type == NodeType.WALL_BOUNDARY, 0],
        mesh_pos[node_type == NodeType.WALL_BOUNDARY, 1],
        c="green",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        mesh_pos[node_type == NodeType.OUTFLOW, 0],
        mesh_pos[node_type == NodeType.OUTFLOW, 1],
        c="orange",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        mesh_pos[node_type == NodeType.INFLOW, 0],
        mesh_pos[node_type == NodeType.INFLOW, 1],
        c="blue",
        linewidths=1,
        s=20,
    )
    plt.savefig(f"{fig_dir}/node distribution.png")
    plt.close()

    face_center_pos = mesh["face_center_pos"]
    face_type = mesh["face_type"].long().view(-1)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.cla()
    ax.set_aspect("equal")
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells_node.view(-1, 3))
    ax.triplot(triang, "k-", ms=0.5, lw=0.3)
    plt.scatter(
        face_center_pos[face_type == NodeType.NORMAL, 0],
        face_center_pos[face_type == NodeType.NORMAL, 1],
        c="red",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        face_center_pos[face_type == NodeType.WALL_BOUNDARY, 0],
        face_center_pos[face_type == NodeType.WALL_BOUNDARY, 1],
        c="green",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        face_center_pos[face_type == NodeType.OUTFLOW, 0],
        face_center_pos[face_type == NodeType.OUTFLOW, 1],
        c="orange",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        face_center_pos[face_type == NodeType.INFLOW, 0],
        face_center_pos[face_type == NodeType.INFLOW, 1],
        c="blue",
        linewidths=1,
        s=20,
    )
    plt.savefig(f"{fig_dir}/face center distribution.png")
    plt.close()

    centroid = mesh["centroid"]
    cells_type = mesh["cells_type"].long().view(-1)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.cla()
    ax.set_aspect("equal")
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells_node.view(-1, 3))
    ax.triplot(triang, "k-", ms=0.5, lw=0.3)
    plt.scatter(
        centroid[cells_type == NodeType.NORMAL, 0],
        centroid[cells_type == NodeType.NORMAL, 1],
        c="red",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        centroid[
            (cells_type == NodeType.WALL_BOUNDARY)
            | (cells_type == NodeType.IN_WALL)
            | (cells_type == NodeType.OUT_WALL),
            0,
        ],
        centroid[
            (cells_type == NodeType.WALL_BOUNDARY)
            | (cells_type == NodeType.IN_WALL)
            | (cells_type == NodeType.OUT_WALL),
            1,
        ],
        c="green",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        centroid[cells_type == NodeType.OUTFLOW, 0],
        centroid[cells_type == NodeType.OUTFLOW, 1],
        c="orange",
        linewidths=1,
        s=20,
    )
    plt.scatter(
        centroid[cells_type == NodeType.INFLOW, 0],
        centroid[cells_type == NodeType.INFLOW, 1],
        c="blue",
        linewidths=1,
        s=20,
    )
    plt.savefig(f"{fig_dir}/cell center distribution.png")
    plt.close()

    # plot unv
    unit_normal_vector = mesh["unit_norm_v"]
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.set_title("unit_norm_vector")
    ax.set_aspect("equal")
    ax.triplot(triang, "k-", ms=0.5, lw=0.3, zorder=1)

    ax.quiver(
        centroid[:, 0].numpy(),
        centroid[:, 1].numpy(),
        unit_normal_vector[:, 0, 0].numpy(),
        unit_normal_vector[:, 0, 1].numpy(),
        units="height",
        color="red",
        angles="xy",
        scale_units="xy",
        scale=75,
        width=0.01,
        headlength=3,
        headwidth=2,
        headaxislength=4.5,
    )

    ax.quiver(
        centroid[:, 0].numpy(),
        centroid[:, 1].numpy(),
        unit_normal_vector[:, 1, 0].numpy(),
        unit_normal_vector[:, 1, 1].numpy(),
        units="height",
        color="blue",
        angles="xy",
        scale_units="xy",
        scale=75,
        width=0.01,
        headlength=3,
        headwidth=2,
        headaxislength=4.5,
    )

    ax.quiver(
        centroid[:, 0].numpy(),
        centroid[:, 1].numpy(),
        unit_normal_vector[:, 2, 0].numpy(),
        unit_normal_vector[:, 2, 1].numpy(),
        units="height",
        color="green",
        angles="xy",
        scale_units="xy",
        scale=75,
        width=0.01,
        headlength=3,
        headwidth=2,
        headaxislength=4.5,
    )
    plt.savefig(f"{subdir}/unit norm vector distribution.png")
    plt.close()


def calc_cell_centered_with_node_attr(
    node_attr, cells_node, cells_index, reduce="mean", map=True
):
    if cells_node.shape != cells_index.shape:
        raise ValueError("wrong cells_node/cells_index dim")

    if len(cells_node.shape) > 1:
        cells_node = cells_node.view(-1)

    if len(cells_index.shape) > 1:
        cells_index = cells_index.view(-1)

    if map:
        maped_node_attr = node_attr[cells_node]
    else:
        maped_node_attr = node_attr

    cell_attr = scatter(src=maped_node_attr, index=cells_index, dim=0, reduce=reduce)

    return cell_attr


def calc_node_centered_with_cell_attr(
    cell_attr, cells_node, cells_index, reduce="mean", map=True
):
    if cells_node.shape != cells_index.shape:
        raise ValueError(f"wrong cells_node/cells_index dim {path['mesh_file_path']}")

    if len(cells_node.shape) > 1:
        cells_node = cells_node.view(-1)

    if len(cells_index.shape) > 1:
        cells_index = cells_index.view(-1)

    if map:
        maped_cell_attr = cell_attr[cells_index]
    else:
        maped_cell_attr = cell_attr

    cell_attr = scatter(src=maped_cell_attr, index=cells_node, dim=0, reduce=reduce)

    return cell_attr


def extract_mesh_state(
    dataset,
    tf_writer,
    index,
    origin_writer=None,
    mode="cylinder_mesh",
    h5_writer=None,
    path=None,
    mesh_only=False,
    plot=False,
):
    """
    all input dataset values should be pytorch tensor object
    """
    dataset = convert_to_tensors(dataset)

    #   return False
    mesh = {}
    """>>> prepare for converting >>>"""
    mesh["mesh_pos"] = dataset["mesh_pos"][0]
    mesh["cells_node"] = dataset["cells_node"][0]
    mesh["cells_index"] = dataset["cells_index"][0]
    cells_node = mesh["cells_node"]
    cells_index = mesh["cells_index"]
    """<<< prepare for converting <<<"""

    """>>>compute centroid crds>>>"""
    mesh_pos = dataset["mesh_pos"][0]
    centroid = calc_cell_centered_with_node_attr(
        node_attr=mesh["mesh_pos"],
        cells_node=cells_node,
        cells_index=cells_index,
        reduce="mean",
    )
    mesh["centroid"] = centroid
    """<<<compute centroid crds<<<"""

    """ >>>   compose face  and face_center_pos >>> """
    decomposed_cells = make_edges_unique(
        dataset["cells_face_node"][0], cells_node.view(-1, 1), cells_index.view(-1, 1)
    )
    face_node_x = decomposed_cells["face_node_x"]
    face = decomposed_cells["face_with_bias"]
    if face.shape[0] > face.shape[1]:
        face = face.T
    mesh["face"] = face
    if face_node_x.shape[0] > face_node_x.shape[1]:
        face_node_x = face_node_x.T
    mesh["face_node_x"] = face_node_x

    face_center_pos = (mesh_pos[face[0]] + mesh_pos[face[1]]) / 2.0
    mesh["face_center_pos"] = face_center_pos
    """ <<<   compose face   <<< """

    node_neigbors, max_neighbors = create_neighbor_matrix(mesh_pos, face_node_x.T)
    mesh["node_neigbors"] = node_neigbors
    mesh["node_neigbors_shape"] = torch.tensor(node_neigbors.shape)

    """ >>>   compute face length   >>>"""
    mesh["face_length"] = torch.norm(
        (mesh_pos[face[0]] - mesh_pos[face[1]]), dim=1, keepdim=True
    )
    """ <<<   compute face length   <<<"""

    """ >>> compute face`s neighbor cell index >>> """
    # prepare for cells_type
    node_type = dataset["node_type"][0]
    cells_type = torch.zeros_like(centroid)[:, 0:1].long()
    cells_node_type_sum = 0
    flag = 0

    # prepare for face`s neighbor cell inde
    cells_face_node_biased = decomposed_cells["cells_face_node_biased"]
    cells_index = mesh["cells_index"]
    cells_face_node_cell_index_dict = {}
    neighbour_cell = []

    # prepare for cells_face
    face_list = face.T
    face_index = {}
    for i in range(face_list.shape[0]):
        face_index[str(face_list[i].numpy())] = i
    cells_face = torch.zeros_like(cells_face_node_biased)[:, 0:1]

    for edges_node_index_i, edges_node_index in enumerate(cells_face_node_biased):
        if str(edges_node_index.numpy()) in cells_face_node_cell_index_dict.keys():
            cells_face_node_cell_index_dict[str(edges_node_index.numpy())].append(
                cells_index[edges_node_index_i]
            )
        else:
            cells_face_node_cell_index_dict[str(edges_node_index.numpy())] = [
                cells_index[edges_node_index_i]
            ]

        cells_face[edges_node_index_i] = face_index[str(edges_node_index.numpy())]

        if cells_index[edges_node_index_i] <= flag:
            cells_node_type_sum += node_type[cells_node[edges_node_index_i], 0]
        else:
            if cells_node_type_sum > 0:
                cells_type[cells_index[edges_node_index_i - 1], 0] = torch.tensor(
                    [NodeType.BOUNDARY_CELL], dtype=torch.long
                )
            else:
                cells_type[cells_index[edges_node_index_i - 1], 0] = torch.tensor(
                    [NodeType.NORMAL], dtype=torch.long
                )
            cells_node_type_sum = 0
            cells_node_type_sum += node_type[cells_node[edges_node_index_i], 0]
            flag += 1

    mesh["cells_type"] = cells_type.to(torch.int64)
    mesh["cells_face"] = cells_face.to(torch.int64)

    for edges_i, edges in enumerate(face.T):
        edge_neighbour_raw = cells_face_node_cell_index_dict[str(edges.numpy())]
        if len(edge_neighbour_raw) < 2:
            edge_neighbour_raw.append(edge_neighbour_raw[0])
        neighbour_cell.append(torch.cat(edge_neighbour_raw))

    neighbour_cell = torch.stack(neighbour_cell, dim=0)

    # validation
    senders_cell = calc_node_centered_with_cell_attr(
        cell_attr=cells_index,
        cells_node=cells_face,
        cells_index=cells_index,
        reduce="max",
        map=False,
    ).squeeze(1)

    recivers_cell = calc_node_centered_with_cell_attr(
        cell_attr=cells_index,
        cells_node=cells_face,
        cells_index=cells_index,
        reduce="min",
        map=False,
    ).squeeze(1)
    neighbour_cell_valid = torch.stack((recivers_cell, senders_cell), dim=0)
    mask_neighbour_cell = neighbour_cell.T != neighbour_cell_valid
    mask_neighbour_cell = torch.logical_or(
        mask_neighbour_cell[0], mask_neighbour_cell[1]
    )
    if mask_neighbour_cell.all():
        raise ValueError(
            f"wrong neighbour cell generate at file {path['mesh_file_path']}"
        )
    # plot_edge_direction(centroid,torch.from_numpy(neighbour_cell[0]),mesh_pos,cells_node)
    # neighbour_cell_with_bias = reorder_face(centroid,torch.from_numpy(neighbour_cell),plot=False)
    mesh["neighbour_cell"] = neighbour_cell.T.to(torch.int64)
    """ <<< compute face`s neighbor cell index <<< """

    """ >>>  check-out face_type  >>> """
    face_type = torch.zeros((face.shape[1], 1)).to(torch.int64)
    a = torch.index_select(node_type, 0, face[0])
    b = torch.index_select(node_type, 0, face[1])
    face_center_pos = (mesh_pos[face[0]] + mesh_pos[face[1]]) / 2.0
    mesh["node_type"] = node_type
    mesh["face_center_pos"] = face_center_pos
    # print("After renumed data has node type:")
    # stastic_nodeface_type(node_type)

    if mode.find("airfoil") != -1:
        face_type = torch.from_numpy(face_type)
        Airfoil = torch.full(face_type.shape, NodeType.AIRFOIL).to(torch.int64)
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int64)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int64)
        a = torch.from_numpy(a).view(-1)
        b = torch.from_numpy(b).view(-1)
        face_type[
            (a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :
        ] = Airfoil[(a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :]
        face_type[
            (a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :
        ] = Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]
        face_type[
            (a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :
        ] = Inlet[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :]

    else:
        topwall = torch.max(face_center_pos[:, 1])
        bottomwall = torch.min(face_center_pos[:, 1])
        outlet = torch.max(face_center_pos[:, 0])
        inlet = torch.min(face_center_pos[:, 0])

        """ for more robustness """
        topwall_Upper_limit = topwall + 1e-5
        topwall_Lower_limit = topwall - 1e-5

        bottomwall_Upper_limit = bottomwall + 1e-5
        bottomwall_Lower_limit = bottomwall - 1e-5

        outlet_Upper_limit = outlet + 1e-5
        outlet_Lower_limit = outlet - 1e-5

        inlet_Upper_limit = inlet + 1e-5
        inlet_Lower_limit = inlet - 1e-5

        original_limit = [
            (topwall_Lower_limit, topwall_Upper_limit),
            (bottomwall_Lower_limit, bottomwall_Upper_limit),
            (outlet_Lower_limit, outlet_Upper_limit),
            (inlet_Lower_limit, inlet_Upper_limit),
        ]

        face_type = face_type
        WALL_BOUNDARY_t = torch.full(face_type.shape, NodeType.WALL_BOUNDARY).to(
            torch.int64
        )
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int64)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int64)
        Outlet = torch.full(face_type.shape, NodeType.OUTFLOW).to(torch.int64)
        a = a.view(-1)
        b = b.view(-1)

        """ Without considering the corner points """
        boundary_face_mask = neighbour_cell[:, 0] == neighbour_cell[:, 1]
        face_type[
            (a == b)
            & (a == NodeType.WALL_BOUNDARY)
            & (b == NodeType.WALL_BOUNDARY)
            & boundary_face_mask,
            :,
        ] = WALL_BOUNDARY_t[
            (a == b)
            & (a == NodeType.WALL_BOUNDARY)
            & (b == NodeType.WALL_BOUNDARY)
            & boundary_face_mask,
            :,
        ]
        face_type[
            (a == b)
            & (a == NodeType.INFLOW)
            & (b == NodeType.INFLOW)
            & boundary_face_mask,
            :,
        ] = Inlet[
            (a == b)
            & (a == NodeType.INFLOW)
            & (b == NodeType.INFLOW)
            & boundary_face_mask,
            :,
        ]
        face_type[
            (a == b)
            & (a == NodeType.OUTFLOW)
            & (b == NodeType.OUTFLOW)
            & boundary_face_mask,
            :,
        ] = Outlet[
            (a == b)
            & (a == NodeType.OUTFLOW)
            & (b == NodeType.OUTFLOW)
            & boundary_face_mask,
            :,
        ]
        face_type[
            (a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :
        ] = Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]

        """ Use position relationship to regulate the corner points """
        if path["flow_type"] == "pipe_flow":
            face_type[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ] = Inlet[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ]

            face_type[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ] = Outlet[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ]

        elif "cavity" in path["flow_type"]:
            face_type[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ] = WALL_BOUNDARY_t[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ]

            face_type[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ] = WALL_BOUNDARY_t[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ]

        else:
            face_type[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ] = Inlet[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.INFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ]

            face_type[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ] = Outlet[
                (
                    (
                        (a == NodeType.WALL_BOUNDARY)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.WALL_BOUNDARY)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ]

            face_type[
                (
                    (
                        (a == NodeType.INFLOW)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.INFLOW)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ] = Outlet[
                (
                    (
                        (a == NodeType.INFLOW)
                        & (b == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                    | (
                        (b == NodeType.INFLOW)
                        & (a == NodeType.OUTFLOW)
                        & (neighbour_cell[:, 0] == neighbour_cell[:, 1])
                    )
                ),
                :,
            ]

    mesh["face_type"] = face_type
    # print("After renumed data has face type:")
    # stastic_nodeface_type(face_type)
    """ <<<  check-out face_type  <<< """

    """ >>>         plot boundary cell center pos           >>>"""
    centroid = centroid

    if (len(cells_type.shape) > 1) and (len(cells_type.shape) < 3):
        cells_type = cells_type.reshape(-1)
    else:
        raise ValueError(f"chk cells_type dim {path['mesh_file_path']}")
    """ <<<         plot boundary cell center pos           <<<"""

    """ >>>         compose ghost cells           >>>"""
    # # compse mesh_pos to node_type map#
    # pos_to_type = {}
    # for i in range(node_type.shape[0]):
    #   pos_to_type[str(mesh_pos[i])]=node_type[i]

    # cell_three_vertex_type = cells_node.clone()
    # for i in range(cells_node.shape[0]):
    #   for j in range(cells_node.shape[1]):
    #     cell_three_vertex_type[i][j] = pos_to_type[str(mesh_pos[cells_node[i][j]])]

    # pos_to_edge_index = {}
    # for i in range(face.shape[0]):
    #   try:
    #     pos_to_edge_index[str(mesh_pos[face[i][0]])].append(face_type[i])
    #   except:
    #     pos_to_edge_index[str(mesh_pos[face[i][0]])] = []
    #     pos_to_edge_index[str(mesh_pos[face[i][0]])].append(face_type[i])

    #   try:
    #     pos_to_edge_index[str(mesh_pos[face[i][1]])].append(face_type[i])
    #   except:
    #     pos_to_edge_index[str(mesh_pos[face[i][1]])] = []
    #     pos_to_edge_index[str(mesh_pos[face[i][1]])].append(face_type[i])

    #   # if pos_to_edge_index[str(mesh_pos[face[i][0]])] is None:
    #   #   pos_to_edge_index[str(mesh_pos[face[i][0]])] = []
    #   # else:
    #   #   pos_to_edge_index[str(mesh_pos[face[i][0]])].append(i)

    #   # if pos_to_edge_index[str(mesh_pos[face[i][1]])] is None:
    #   #   pos_to_edge_index[str(mesh_pos[face[i][1]])] = []
    #   # else:
    #   #   pos_to_edge_index[str(mesh_pos[face[i][1]])].append(i)

    # # select boundary cell threads
    # if (len(cells_type.shape)>1) and (len(cells_type.shape)<3):
    #   cells_type = cells_type.reshape(-1)
    # elif (len(cells_type.shape)>=3):
    #   raise ValueError("chk cells_type dim")
    # Inflow_cell_thread = [cells_node[(cells_type==NodeType.INFLOW)|(cells_type==NodeType.IN_WALL),:],cell_three_vertex_type[(cells_type==NodeType.INFLOW)|(cells_type==NodeType.IN_WALL),:],"inlet"]
    # Wall_cell_thread = [cells_node[(cells_type==NodeType.WALL_BOUNDARY)|(cells_type==NodeType.IN_WALL)|(cells_type==NodeType.OUT_WALL),:],cell_three_vertex_type[(cells_type==NodeType.WALL_BOUNDARY)|(cells_type==NodeType.IN_WALL)|(cells_type==NodeType.OUT_WALL),:],"wall"]
    # Outflow_cell_thread = [cells_node[(cells_type==NodeType.OUTFLOW)|(cells_type==NodeType.OUT_WALL),:],cell_three_vertex_type[(cells_type==NodeType.OUTFLOW)|(cells_type==NodeType.OUT_WALL),:],"outlet"]

    # # # select boundary face threads
    # # if (len(face_type.shape)>1) and (len(face_type.shape)<3):
    # #   face_type = face_type.reshape(-1)
    # # elif (len(face_type.shape)>=3):
    # #   raise ValueError("chk cells_type dim")

    # # Inflow_face_thread = face[(face_type==NodeType.INFLOW),:]
    # # Wall_face_thread = face[(face_type==NodeType.WALL_BOUNDARY),:]
    # # Outflow_face_thread = face[(face_type==NodeType.OUTFLOW),:]

    # boundary_domain = [[Inflow_cell_thread],[Wall_cell_thread],[Outflow_cell_thread]]
    # mesh,dataset,nodes_of_cell = make_ghost_cell(dataset.copy(),
    #                                              boundary_domain.copy(),
    #                                              torch.from_numpy(mesh_pos),
    #                                              cells_node,torch.from_numpy(node_type),
    #                                              mode,recover_unorder=unorder,
    #                                              limit = original_limit,
    #                                              pos_to_edge_index=pos_to_edge_index,
    #                                              index=index,
    #                                              fig=fig,
    #                                              ax=ax)
    # mesh_pos = mesh['mesh_pos']
    # cells_node = mesh['cells_node'][0]
    # face = mesh['face']
    # senders = torch.from_numpy(face[0,:])
    # receivers = torch.from_numpy(face[1,:])
    """ <<<         compose ghost cells           <<<"""

    """ >>> unit normal vector >>> """
    face_length = mesh["face_length"]
    senders_node, recivers_node = face[0], face[1]
    mesh_pos = mesh_pos
    pos_diff = mesh_pos[senders_node] - mesh_pos[recivers_node]
    unv = torch.cat((-pos_diff[:, 1:2], pos_diff[:, 0:1]), dim=1)
    unv = unv / (torch.norm(unv, dim=1, keepdim=True))

    if not torch.isfinite(unv).all():
        raise ValueError(f'unv Error mesh_file {path["mesh_file_path"]}')

    face_to_centroid = (
        face_center_pos[cells_face.view(-1)] - centroid[cells_index.view(-1)]
    )
    cells_face_unv = unv[cells_face.view(-1)]
    unv_dir_mask = (
        torch.sum(face_to_centroid * cells_face_unv, dim=1, keepdim=True) > 0.0
    ).repeat(1, 2)
    cells_face_unv_bias = torch.where(
        unv_dir_mask, cells_face_unv, (-1.0) * cells_face_unv
    )

    cells_face_length = face_length[cells_face.view(-1)]
    surface_vector = cells_face_unv_bias * cells_face_length

    if plot:
        plot_mesh(
            mesh_pos.numpy(),
            face_node_x.T.numpy(),
            node_type.view(-1).numpy(),
            face_center_pos.numpy(),
            face_type.view(-1).numpy(),
            centroid[cells_index.view(-1)].numpy(),
            plot_mesh=True,
            unit_normal_vector=None,
            other_cell_centered_vector=None,
            path=path,
        )

    valid = calc_cell_centered_with_node_attr(
        node_attr=surface_vector,
        cells_node=cells_face,
        cells_index=cells_index,
        reduce="sum",
        map=False,
    )
    if not (-1e-12 < valid.sum() < 1e-12):
        raise ValueError(f"wrong unv calculation {path['mesh_file_path']}")
    mesh["unit_norm_v"] = cells_face_unv_bias
    """ <<< unit normal vector <<< """

    """ >>> cells node face >>> """

    def find_indices(A, B):
        # 获取A的排序索引
        sorted_indices = torch.argsort(A)
        # 获取排序后的A
        sorted_A = A[sorted_indices]

        # 对B中的每个元素在排序后的A中进行搜索
        # 使用searchsorted找到B中元素在排序后A中的位置
        indices_in_sorted_A = torch.searchsorted(sorted_A, B)

        # 将这些位置转换回原始A中的索引
        # 因为sorted_indices是对应排序后A的，所以直接使用得到的位置索引即可
        indices_in_A = sorted_indices[indices_in_sorted_A]

        return indices_in_A

    cells_face_node = decomposed_cells["cells_face_node_unbiased"]

    cells_node_face = []

    last_two_way_current_cells_face_node_max = 0
    last_two_way_current_cells_face_node_face_local_max = 0

    cells_coincident_nodes = {}

    for cells_index_i in range(cells_index.max() + 1):
        current_cells_index_mask = (cells_index_i == cells_index).view(-1)

        current_cells_node = cells_node[current_cells_index_mask].view(-1)

        for nodes_index in current_cells_node:
            if not str(nodes_index) in cells_coincident_nodes.keys():
                cells_coincident_nodes[str(nodes_index)] = np.array([cells_index_i])
            elif not (cells_index_i in cells_coincident_nodes[str(nodes_index)]):
                cells_coincident_nodes[str(nodes_index)] = np.append(
                    cells_coincident_nodes[str(nodes_index)], cells_index_i
                )
            else:
                continue

        current_cells_face_node = cells_face_node[current_cells_index_mask]

        two_way_current_cells_face_node = torch.cat(
            (current_cells_face_node[:, 0], current_cells_face_node[:, 1]), dim=0
        )

        two_way_current_cells_face_node_local = find_indices(
            current_cells_node, two_way_current_cells_face_node
        )

        current_cells_face = (cells_face[current_cells_index_mask]).view(-1).repeat(2)

        senders_node_face = scatter(
            src=current_cells_face,
            index=two_way_current_cells_face_node_local,
            dim=0,
            reduce="max",
        )

        recivers_node_face = scatter(
            src=current_cells_face,
            index=two_way_current_cells_face_node_local,
            dim=0,
            reduce="min",
        )

        two_way_current_cells_face_node_face_local = (
            find_indices(
                current_cells_face,
                torch.cat((senders_node_face, recivers_node_face), dim=0),
            )
            + last_two_way_current_cells_face_node_face_local_max
        )
        last_two_way_current_cells_face_node_face_local_max = (
            two_way_current_cells_face_node_face_local.max() + 1
        )

        curent_cells_node_face = torch.stack(
            torch.chunk(two_way_current_cells_face_node_face_local, 2), dim=1
        )

        cells_node_face.append(curent_cells_node_face)

    cells_node_face = torch.cat(cells_node_face, dim=0)

    mesh["cells_node_face"] = cells_node_face

    cells_node_face_unv = 0.5 * (
        cells_face_unv_bias[cells_node_face[:, 0]]
        + cells_face_unv_bias[cells_node_face[:, 1]]
    )

    mesh["cells_node_face_unv"] = cells_node_face_unv

    # valid
    cells_face_surface_vector = surface_vector

    node_face_surface_vector = 0.5 * (
        cells_face_surface_vector[cells_node_face[:, 0]]
        + cells_face_surface_vector[cells_node_face[:, 1]]
    )

    mesh["node_face_surface_vector"] = node_face_surface_vector

    sum_node_face_surface_vector = calc_cell_centered_with_node_attr(
        node_attr=node_face_surface_vector,
        cells_node=cells_node,
        cells_index=cells_index,
        reduce="sum",
        map=False,
    )

    if not (-1e-12 < sum_node_face_surface_vector.sum() < 1e-12):
        raise ValueError(
            f"wrong sum_node_face_surface_vector calculation {path['mesh_file_path']}"
        )
    """ <<< cells node face <<< """

    """ >>> compute neighbor cell X >>> """
    neighbour_cell_bias, _ = torch.sort(neighbour_cell, dim=-1)
    neighbour_cell_x_list = []
    for cells_index_i in range(cells_index.max() + 1):
        current_cells_index_mask = (cells_index_i == cells_index).view(-1)
        current_cells_node = cells_node[current_cells_index_mask].view(-1)
        for current_cells_node_index in current_cells_node:
            current_cells_coincident_nodes = cells_coincident_nodes[
                str(current_cells_node_index)
            ]
            for coincident_nodes_cell_index in current_cells_coincident_nodes:
                if int(coincident_nodes_cell_index) != cells_index_i:
                    local_neighbor_cell, _ = torch.tensor([int(coincident_nodes_cell_index), cells_index_i], 
                                                          dtype=torch.long).sort()
                    # 检查 local_neighbor_cell 是否不在 neighbour_cell_bias 中
                    is_new_neighbor = not ((neighbour_cell_bias[:, 0] == local_neighbor_cell[0]) &
                                           (neighbour_cell_bias[:, 1] == local_neighbor_cell[1])).any()
                    if is_new_neighbor:
                        neighbour_cell_x_list.append(local_neighbor_cell.unsqueeze(0))
                else:
                    continue
                    
    # 处理完所有单元格后
    if neighbour_cell_x_list:  # 确保列表不为空
        neighbour_cell_x_list = torch.cat(neighbour_cell_x_list, dim=0)
        neighbour_cell_x_list, _ = torch.unique(neighbour_cell_x_list, dim=0, return_inverse=True)
        neighbour_cell_x = torch.cat((neighbour_cell, neighbour_cell_x_list), dim=0)
    else:
        neighbour_cell_x = neighbour_cell_bias

    mesh["neighbour_cell_x"] = neighbour_cell_x.T
    # # plot neighbour_cell_x
    # centroid_x = np.array(centroid)
    # neighbour_cell_x = np.array(neighbour_cell_x)

    # # plot neighbor_cell_x
    # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    # ax.cla()
    # ax.set_aspect('equal')

    # # 通过索引获取每一条边的两个点的坐标
    # point1_x = centroid_x[neighbour_cell_x[:, 0]]
    # point2_x = centroid_x[neighbour_cell_x[:, 1]]

    # # 将每一对点的坐标合并，方便绘图
    # lines_x = np.hstack([point1_x, point2_x]).reshape(-1, 2, 2)

    # # 使用plot绘制所有的边
    # plt.plot(lines_x[:, :, 0].T, lines_x[:, :, 1].T, 'k--', lw=1, alpha=0.2)

    # mesh_pos = np.array(mesh_pos)
    # edge_index = np.array(face.T)

    # # 通过索引获取每一条边的两个点的坐标
    # point1 = mesh_pos[edge_index[:, 0]]
    # point2 = mesh_pos[edge_index[:, 1]]

    # # 将每一对点的坐标合并，方便绘图
    # lines = np.hstack([point1, point2]).reshape(-1, 2, 2)

    # # 使用plot绘制所有的边
    # plt.plot(lines[:, :, 0].T, lines[:, :, 1].T, 'k-', lw=2, alpha=0.8)
    # plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1,s=5)

    # plt.show()
    # plt.close()
    """ <<< compute neighbor cell X <<< """

    """ >>> compute cell attribute V_BIC and P_BIC and cell_factor >>> """
    cell_node_centroid_dist = torch.norm(
        mesh_pos[cells_node.view(-1)] - centroid[cells_index.view(-1)],
        dim=1,
        keepdim=True,
    )
    cell_node_centroid_dist_total = calc_cell_centered_with_node_attr(
        node_attr=cell_node_centroid_dist,
        cells_node=cells_node,
        cells_index=cells_index,
        reduce="sum",
        map=False,
    )

    cell_factor = (
        cell_node_centroid_dist / cell_node_centroid_dist_total[cells_index.view(-1)]
    )

    mesh["cells_factor"] = cell_factor

    if not mesh_only:
        v_target = dataset["velocity"]
        p_target = dataset["pressure"]
        mesh[
            "target|velocity_on_node"
        ] = v_target  # obviously, velocity with BC, IC is v_target[0]
        mesh[
            "target|pressure_on_node"
        ] = p_target  # obviously, velocity with BC, IC is v_pressure[1]
        if "density" in dataset.keys():
            mesh["target|density"] = dataset["density"]
    """ <<< compute cell attribute V_BIC and P_BIC and  <<< """

    # compute cell_area
    cells_face = mesh["cells_face"]
    face_length = mesh["face_length"]

    surface_vector = surface_vector
    full_synataic_function = 0.5 * face_center_pos[cells_face.view(-1)]

    cells_area = calc_cell_centered_with_node_attr(
        node_attr=(full_synataic_function * surface_vector).sum(dim=1, keepdim=True),
        cells_node=cells_face,
        cells_index=cells_index,
        reduce="sum",
        map=False,
    )

    # use shoelace formula to validate
    cells_face_node_unbiased = decomposed_cells["cells_face_node_unbiased"]
    test_cells_area = []
    for i in range(cells_index.max().numpy() + 1):
        test_cells_area.append(
            polygon_area(mesh_pos[cells_node[(cells_index == i).view(-1)].view(-1)])
        )
    test_cells_area = torch.from_numpy(np.asarray(test_cells_area))

    valid_cells_area = (cells_area.view(-1) - test_cells_area).sum()

    if not (-1e-12 < valid_cells_area < 1e-12):
        mesh["cells_area"] = test_cells_area.unsqueeze(1)
        print(
            f"warning substitude cells area with shoelace formula(resdiual:{valid_cells_area.numpy()}) {path['mesh_file_path']}"
        )
    else:
        mesh["cells_area"] = cells_area

    mesh = cal_mean_u_and_cd(mesh, flow_type=path["flow_type"], mesh_only=mesh_only)

    # edge attr
    mesh["mesh_pos"] = mesh["mesh_pos"].unsqueeze(0).numpy()
    mesh["node_type"] = mesh["node_type"].to(torch.int32).unsqueeze(0).numpy()
    mesh["face"] = mesh["face"].to(torch.int32).unsqueeze(0).numpy()
    mesh["face_type"] = mesh["face_type"].to(torch.int32).unsqueeze(0).numpy()
    mesh["face_length"] = mesh["face_length"].to(torch.float32).unsqueeze(0).numpy()
    mesh["face_center_pos"] = mesh["face_center_pos"].unsqueeze(0).numpy()
    mesh["neighbour_cell"] = mesh["neighbour_cell"].to(torch.int32).unsqueeze(0).numpy()
    mesh["centroid"] = mesh["centroid"].unsqueeze(0).numpy()
    mesh["cells_node"] = mesh["cells_node"].to(torch.int32).unsqueeze(0).numpy()
    mesh["cells_index"] = mesh["cells_index"].to(torch.int32).unsqueeze(0).numpy()
    mesh["cells_face_node_biased"] = (
        cells_face_node_biased.to(torch.int32).unsqueeze(0).numpy()
    )
    mesh["cells_face_node_unbiased"] = (
        cells_face_node_unbiased.to(torch.int32).unsqueeze(0).numpy()
    )
    mesh["cells_face"] = mesh["cells_face"].to(torch.int32).unsqueeze(0).numpy()
    mesh["cells_type"] = mesh["cells_type"].to(torch.int32).unsqueeze(0).numpy()
    mesh["unit_norm_v"] = mesh["unit_norm_v"].to(torch.float32).unsqueeze(0).numpy()
    mesh["cells_factor"] = mesh["cells_factor"].to(torch.float32).unsqueeze(0).numpy()
    mesh["cells_area"] = mesh["cells_area"].to(torch.float32).unsqueeze(0).numpy()
    mesh["cylinder_diameter"] = (
        mesh["cylinder_diameter"].to(torch.float32).unsqueeze(0).numpy()
    )
    mesh["cells_node_face"] = (
        mesh["cells_node_face"].to(torch.int32).unsqueeze(0).numpy()
    )
    mesh["node_face_surface_vector"] = (
        mesh["node_face_surface_vector"].to(torch.float32).unsqueeze(0).numpy()
    )
    mesh["cells_node_face_unv"] = (
        mesh["cells_node_face_unv"].to(torch.float32).unsqueeze(0).numpy()
    )
    mesh["flow_type"] = (
        torch.tensor(path["flow_type_mapping"].get(path["flow_type"]))
        .unsqueeze(0)
        .unsqueeze(1)
        .unsqueeze(2)
        .numpy()
    )
    mesh["origin_mesh_path"] = np.expand_dims(
        np.asarray([float(ord(c)) for c in path["mesh_file_path"]]), axis=(0, 2)
    )

    # X info
    mesh["face_node_x"] = mesh["face_node_x"].to(torch.int32).unsqueeze(0).numpy()
    mesh["neighbour_cell_x"] = (
        mesh["neighbour_cell_x"].to(torch.int32).unsqueeze(0).numpy()
    )
    mesh["node_neigbors"] = mesh["node_neigbors"].to(torch.int32).numpy()
    mesh["node_neigbors_shape"] = mesh["node_neigbors_shape"].unsqueeze(0).numpy()

    dataset_info = {
        "simulator": path["simulator"],
        "dt": path["dt"],
        "rho": path["rho"],
        "mu": path["mu"],
        "max_neighbors": max_neighbors.item()
        if isinstance(max_neighbors, torch.Tensor)
        else max_neighbors,
        "features": None,
    }

    if path["saving_mesh"]:
        tf_dataset_info = dataset_info.copy()
        tf_dataset_info["features"] = mesh
        # write_dict_info_to_json(dataset_info,os.path.dirname(path["tf_saving_path"])+"/meta.json")
        # write_tfrecord_one_with_writer(tf_writer,mesh,mode="cylinder_mesh")

        tf_dataset = [tf_dataset_info, mesh]  # for multiprocessing

    if path["saving_origin"]:
        if not mesh_only:
            origin_mesh = {
                "mesh_pos": mesh["mesh_pos"],
                "cells": mesh["cells_node"],
                "node_type": mesh["node_type"],
                "velocity": mesh["velocity"],
                "pressure": mesh["pressure"],
            }

        if "density" in dataset.keys():
            origin_mesh["target|density"] = dataset["density"]

        else:
            origin_mesh = {
                "mesh_pos": mesh["mesh_pos"],
                "cells": mesh["cells_node"],
                "node_type": mesh["node_type"],
            }
        origin_dataset_info = dataset_info.copy()
        origin_dataset_info["features"] = origin_mesh
        # write_dict_info_to_json(dataset_info,os.path.dirname(path["origin_saving_path"])+"/meta.json")
        # write_tfrecord_one_with_writer(origin_writer,origin_mesh,mode=mode)

        origin_dataset = [origin_dataset_info, origin_mesh]  # for multiprocessing

    if path["saving_h5"]:
        # current_traj = h5_writer.create_group(str(index))
        # for key,value in mesh.items():
        #   current_traj.create_dataset(key,data=value)

        h5_dataset = [None, mesh]  # for multiprocessing

    print("{0}th mesh has been extracted".format(index))

    return tf_dataset, origin_dataset, h5_dataset


def analyze_value(value):
    if isinstance(value, np.ndarray):
        value_shape = value.shape
        adptive_value_shape = []
        for shape_idx, shape_num in enumerate(value_shape):
            if shape_idx == 0:
                # first dim should be time steps, skip it.
                adptive_value_shape.append(shape_num)
                continue
            if shape_num > 50:
                # other mesh-wise dim should be adptive shape
                adptive_value_shape.append(-1)
            else:
                adptive_value_shape.append(shape_num)

        return {
            "type": "dynamic" if value.shape[0] > 1 else "static",
            "shape": adptive_value_shape,
            "dtype": str(value.dtype),
        }

    elif isinstance(value, list):
        return value  # For this example, we assume lists are kept as-is.
    else:
        return value


def write_dict_info_to_json(input_dict, output_file):
    info_dict = {}
    trajectory_length = 1

    for key, value in input_dict.items():
        if isinstance(value, dict):
            info_dict[key] = {k: analyze_value(v) for k, v in value.items()}
            for subkey, subvalue in info_dict[key].items():
                if isinstance(subvalue, dict) and subvalue.get("type") == "dynamic":
                    trajectory_length = subvalue["shape"][0]
        else:
            info_dict[key] = analyze_value(value)

    # Add additional fields
    if "features" in info_dict:
        info_dict["field_names"] = list(info_dict["features"].keys())

    # Add the trajectory_length
    info_dict["trajectory_length"] = trajectory_length

    with open(output_file, "w") as file:
        json.dump(info_dict, file, indent=2)


def find_max_distance(points):
    # 获取点的数量
    n_points = points.size(0)

    # 初始化最大距离为0
    max_distance = 0

    # 遍历每一对点
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # 计算两点之间的欧几里得距离
            distance = torch.norm(points[i] - points[j])

            # 更新最大距离
            max_distance = max(max_distance, distance)

    # 返回最大距离
    return max_distance


def cal_mean_u_and_cd(
    trajectory, flow_type="pipe_flow", mesh_only=False, rho=1.0, mu=0.001
):
    edge_index = trajectory["face"]
    face_type = trajectory["face_type"].view(-1)
    node_type = trajectory["node_type"].view(-1)
    face_length = trajectory["face_length"][:, 0][face_type == NodeType.INFLOW]
    ghosted_mesh_pos = trajectory["mesh_pos"]
    mesh_pos = trajectory["mesh_pos"][
        (node_type != NodeType.GHOST_INFLOW)
        & (node_type != NodeType.GHOST_OUTFLOW)
        & (node_type != NodeType.GHOST_WALL)
    ]
    top = torch.max(mesh_pos[:, 1])
    bottom = torch.min(mesh_pos[:, 1])
    L0 = torch.tensor([1.0])

    if not mesh_only:
        target_on_node = torch.cat(
            (
                trajectory["target|velocity_on_node"][0],
                trajectory["target|pressure_on_node"][0],
            ),
            dim=1,
        )
        target_on_edge = (
            torch.index_select(target_on_node, 0, edge_index[0])
            + torch.index_select(target_on_node, 0, edge_index[1])
        ) / 2.0
        Inlet = target_on_edge[face_type == NodeType.INFLOW][:, 0]
        total_u = torch.sum(Inlet * face_length)
        mean_u = total_u / (top - bottom)
        trajectory["mean_u"] = mean_u.view(1, 1, 1).to(torch.float64)
        rho = rho
        mu = mu
        trajectory["relonyds_num"] = (
            ((mean_u * L0 * rho) / mu).view(1, 1, 1).to(torch.float64)
        )

    if flow_type == "pipe_flow":
        boundary_pos = ghosted_mesh_pos[node_type == NodeType.WALL_BOUNDARY]
        cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1)
        cylinder_not_mask = torch.logical_not(cylinder_mask)
        cylinder_mask = torch.where(
            ((boundary_pos[:, 1] > bottom) & (boundary_pos[:, 1] < top)),
            cylinder_mask,
            cylinder_not_mask,
        )
        cylinder_pos = boundary_pos[cylinder_mask]
        # _,left_index = torch.min(cylinder_pos[:,0:1],dim=0)
        # _,right_index = torch.max(cylinder_pos[:,0:1],dim=0)
        # L0 = torch.norm((cylinder_pos[right_index,:]-cylinder_pos[left_index,:]),dim=1)
        L0 = find_max_distance(cylinder_pos)

    elif flow_type == "cavity_flow":
        inflow_mesh_pos = ghosted_mesh_pos[node_type.view(-1) == NodeType.INFLOW]
        L0 = inflow_mesh_pos[:, 0].max() - inflow_mesh_pos[:, 0].min()

    elif flow_type == "farfield-circular":
        boundary_pos = ghosted_mesh_pos[node_type == NodeType.WALL_BOUNDARY]
        cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1)
        cylinder_not_mask = torch.logical_not(cylinder_mask)
        cylinder_mask = torch.where(
            ((boundary_pos[:, 1] > bottom) & (boundary_pos[:, 1] < top)),
            cylinder_mask,
            cylinder_not_mask,
        )
        cylinder_pos = boundary_pos[cylinder_mask]
        # _,left_index = torch.min(cylinder_pos[:,0:1],dim=0)
        # _,right_index = torch.max(cylinder_pos[:,0:1],dim=0)
        # L0 = torch.norm((cylinder_pos[right_index,:]-cylinder_pos[left_index,:]),dim=1)
        L0 = find_max_distance(cylinder_pos)

    elif flow_type == "circular-possion":
        boundary_pos = ghosted_mesh_pos[node_type == NodeType.WALL_BOUNDARY]
        cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1)
        cylinder_not_mask = torch.logical_not(cylinder_mask)
        cylinder_mask = torch.where(
            ((boundary_pos[:, 1] > bottom) & (boundary_pos[:, 1] < top)),
            cylinder_mask,
            cylinder_not_mask,
        )
        cylinder_pos = boundary_pos[cylinder_mask]
        # _,left_index = torch.min(cylinder_pos[:,0:1],dim=0)
        # _,right_index = torch.max(cylinder_pos[:,0:1],dim=0)
        # L0 = torch.norm((cylinder_pos[right_index,:]-cylinder_pos[left_index,:]),dim=1)
        L0 = find_max_distance(cylinder_pos)

    elif flow_type == "farfield-square":
        boundary_pos = ghosted_mesh_pos[node_type == NodeType.WALL_BOUNDARY]
        cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1)
        cylinder_not_mask = torch.logical_not(cylinder_mask)
        cylinder_mask = torch.where(
            ((boundary_pos[:, 1] > bottom) & (boundary_pos[:, 1] < top)),
            cylinder_mask,
            cylinder_not_mask,
        )
        cylinder_pos = boundary_pos[cylinder_mask]
        # _,left_index = torch.min(cylinder_pos[:,0:1],dim=0)
        # _,right_index = torch.max(cylinder_pos[:,0:1],dim=0)
        # L0 = torch.norm((cylinder_pos[right_index,:]-cylinder_pos[left_index,:]),dim=1)
        L0 = find_max_distance(cylinder_pos)

    elif flow_type == "farfield-half-circular-square":
        boundary_pos = ghosted_mesh_pos[node_type == NodeType.WALL_BOUNDARY]
        cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1)
        cylinder_not_mask = torch.logical_not(cylinder_mask)
        cylinder_mask = torch.where(
            ((boundary_pos[:, 1] > bottom) & (boundary_pos[:, 1] < top)),
            cylinder_mask,
            cylinder_not_mask,
        )
        cylinder_pos = boundary_pos[cylinder_mask]
        # _,left_index = torch.min(cylinder_pos[:,0:1],dim=0)
        # _,right_index = torch.max(cylinder_pos[:,0:1],dim=0)
        # L0 = torch.norm((cylinder_pos[right_index,:]-cylinder_pos[left_index,:]),dim=1)
        L0 = find_max_distance(cylinder_pos)

    if L0 <= 0.0:
        L0 = torch.tensor([1.0])

    trajectory["cylinder_diameter"] = L0.view(1, 1, 1).to(torch.float64)

    return trajectory


def make_dim_less(trajectory, params=None):
    target_on_node = torch.cat(
        (
            torch.from_numpy(trajectory["target|velocity_on_node"][0]),
            torch.from_numpy(trajectory["target|pressure_on_node"][0]),
        ),
        dim=1,
    )
    edge_index = torch.from_numpy(trajectory["face"][0])
    target_on_edge = (
        torch.index_select(target_on_node, 0, edge_index[0])
        + torch.index_select(target_on_node, 0, edge_index[1])
    ) / 2.0
    face_type = torch.from_numpy(trajectory["face_type"][0]).view(-1)
    node_type = torch.from_numpy(trajectory["node_type"][0]).view(-1)
    Inlet = target_on_edge[face_type == NodeType.INFLOW][:, 0]
    face_length = torch.from_numpy(trajectory["face_length"])[0][:, 0][
        face_type == NodeType.INFLOW
    ]
    total_u = torch.sum(Inlet * face_length)
    mesh_pos = torch.from_numpy(trajectory["mesh_pos"][0])
    top = torch.max(mesh_pos[:, 1]).numpy()
    bottom = torch.min(mesh_pos[:, 1]).numpy()
    mean_u = total_u / (top - bottom)

    boundary_pos = mesh_pos[node_type == NodeType.WALL_BOUNDARY].numpy()
    cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1).numpy()
    cylinder_not_mask = np.logical_not(cylinder_mask)
    cylinder_mask = np.where(
        ((boundary_pos[:, 1] > bottom) & (boundary_pos[:, 1] < top)),
        cylinder_mask,
        cylinder_not_mask,
    )

    cylinder_pos = torch.from_numpy(boundary_pos[cylinder_mask])

    xc, yc, R, _ = hyper_fit(np.asarray(cylinder_pos))

    # R = torch.norm(cylinder_pos[0]-torch.tensor([xc,yc]))
    L0 = R * 2.0

    rho = 1

    trajectory["target|velocity_on_node"] = (
        (trajectory["target|velocity_on_node"]) / mean_u
    ).numpy()

    trajectory["target|pressure_on_node"] = (
        trajectory["target|pressure_on_node"] / ((mean_u**2) * (L0**2) * rho)
    ).numpy()

    trajectory["mean_u"] = mean_u.view(1, 1, 1).numpy()
    trajectory["cylinder_diameter"] = L0.view(1, 1, 1).numpy()
    return trajectory


def seprate_cells(mesh_pos, cells, node_type, density, pressure, velocity, index):
    global marker_cell_sp1
    global marker_cell_sp2
    global mask_of_mesh_pos_sp1
    global mask_of_mesh_pos_sp2
    global inverse_of_marker_cell_sp1
    global inverse_of_marker_cell_sp2
    global pos_dict_1
    global pos_dict_2
    global switch_of_redress
    global new_node_type_iframe_sp1
    global new_node_type_iframe_sp2
    global new_mesh_pos_iframe_sp1
    global new_mesh_pos_iframe_sp2

    marker_cell_sp1 = []
    marker_cell_sp2 = []

    # separate cells into two species and saved as marker cells
    for i in range(cells.shape[1]):
        cell = cells[0][i]
        cell = cell.tolist()
        member = 0
        for j in cell:
            x_cord = mesh_pos[0][j][0]
            y_cord = mesh_pos[0][j][1]
            if dividing_line(index, x_cord) >= 0:
                marker_cell_sp1.append(cell)
                member += 1
                break
        if member == 0:
            marker_cell_sp2.append(cell)
    marker_cell_sp1 = np.asarray(marker_cell_sp1, dtype=np.int64)
    marker_cell_sp2 = np.asarray(marker_cell_sp2, dtype=np.int64)

    # use mask to filter the mesh_pos of sp1 and sp2
    marker_cell_sp1_flat = marker_cell_sp1.reshape(
        (marker_cell_sp1.shape[0]) * (marker_cell_sp1.shape[1])
    )
    marker_cell_sp1_flat_uq = np.unique(marker_cell_sp1_flat)

    marker_cell_sp2_flat = marker_cell_sp2.reshape(
        (marker_cell_sp2.shape[0]) * (marker_cell_sp2.shape[1])
    )
    marker_cell_sp2_flat_uq = np.unique(marker_cell_sp2_flat)

    # mask filter of mesh_pos tensor
    inverse_of_marker_cell_sp1 = np.delete(
        mask_of_mesh_pos_sp1, marker_cell_sp1_flat_uq
    )
    inverse_of_marker_cell_sp2 = np.delete(
        mask_of_mesh_pos_sp2, marker_cell_sp2_flat_uq
    )

    # apply mask for mesh_pos first
    new_mesh_pos_iframe_sp1 = np.delete(mesh_pos[0], inverse_of_marker_cell_sp1, axis=0)
    new_mesh_pos_iframe_sp2 = np.delete(mesh_pos[0], inverse_of_marker_cell_sp2, axis=0)

    # redress the mesh_pos`s indexs in the marker_cells,because,np.delete would only delete the element charged by the index and reduce the length of splited mesh_pos_frame tensor,but the original mark_cells stores the original index of the mesh_pos tensor,so we need to redress the indexs
    count = 0
    for i in range(new_mesh_pos_iframe_sp1.shape[0]):
        pos_dict_1[str(new_mesh_pos_iframe_sp1[i])] = i

    for index in range(marker_cell_sp1.shape[0]):
        cell = marker_cell_sp1[index]
        for j in range(3):
            mesh_point = str(mesh_pos[0][cell[j]])
            if pos_dict_1.get(mesh_point, 10000) < 6000:
                marker_cell_sp1[index][j] = pos_dict_1[mesh_point]
            if marker_cell_sp1[index][j] > new_mesh_pos_iframe_sp1.shape[0]:
                count += 1
                print("有{0}个cell 里的 meshPoint的索引值超过了meshnodelist的长度".format(count))

    for r in range(new_mesh_pos_iframe_sp2.shape[0]):
        pos_dict_2[str(new_mesh_pos_iframe_sp2[r])] = r

    count_1 = 0
    for index in range(marker_cell_sp2.shape[0]):
        cell = marker_cell_sp2[index]
        for j in range(3):
            mesh_point = str(mesh_pos[0][cell[j]])
            if pos_dict_2.get(mesh_point, 10000) < 6000:
                marker_cell_sp2[index][j] = pos_dict_2[mesh_point]
            if marker_cell_sp2[index][j] > new_mesh_pos_iframe_sp2.shape[0]:
                count_1 += 1
                print("有{0}个cell 里的 meshPoint的索引值超过了meshnodelist的长度".format(count))

    new_node_type_iframe_sp1 = np.delete(
        node_type[0], inverse_of_marker_cell_sp1, axis=0
    )
    new_node_type_iframe_sp2 = np.delete(
        node_type[0], inverse_of_marker_cell_sp2, axis=0
    )

    new_velocity_sp1 = np.delete(velocity, inverse_of_marker_cell_sp1, axis=1)
    new_density_sp1 = np.delete(density, inverse_of_marker_cell_sp1, axis=1)
    new_pressure_sp1 = np.delete(pressure, inverse_of_marker_cell_sp1, axis=1)

    new_velocity_sp2 = np.delete(velocity, inverse_of_marker_cell_sp2, axis=1)
    new_density_sp2 = np.delete(density, inverse_of_marker_cell_sp2, axis=1)
    new_pressure_sp2 = np.delete(pressure, inverse_of_marker_cell_sp2, axis=1)

    # rearrange_frame with v_sp1 and reshape to 1 dim
    start_time = time.time()
    re_mesh_pos = np.asarray(
        [new_mesh_pos_iframe_sp1]
    )  # mesh is static, so store only once will be enough not 601
    re_cells = np.tile(marker_cell_sp1, (1, 1, 1))  # same as above
    re_node_type = np.asarray([new_node_type_iframe_sp1])  # same as above
    rearrange_frame_1["node_type"] = re_node_type
    rearrange_frame_1["cells"] = re_cells
    rearrange_frame_1["mesh_pos"] = re_mesh_pos
    rearrange_frame_1["density"] = new_density_sp1
    rearrange_frame_1["pressure"] = new_pressure_sp1
    rearrange_frame_1["velocity"] = new_velocity_sp1

    re_mesh_pos = np.asarray(
        [new_mesh_pos_iframe_sp2]
    )  # mesh is static, so store only once will be enough not 601
    re_cells = np.tile(marker_cell_sp2, (1, 1, 1))  # same as above
    re_node_type = np.asarray([new_node_type_iframe_sp2])  # same as above
    rearrange_frame_2["node_type"] = re_node_type
    rearrange_frame_2["cells"] = re_cells
    rearrange_frame_2["mesh_pos"] = re_mesh_pos
    rearrange_frame_2["density"] = new_density_sp2
    rearrange_frame_2["pressure"] = new_pressure_sp2
    rearrange_frame_2["velocity"] = new_velocity_sp2

    end_time = time.time()

    return [rearrange_frame_1, rearrange_frame_2]


def transform_2(mesh_pos, cells, node_type, density, pressure, velocity, index):
    seprate_cells_result = seprate_cells(cells, mesh_pos)
    new_mesh_pos_sp1 = []
    new_mesh_pos_sp2 = []
    marker_cell_sp1 = np.asarray(seprate_cells_result[0], dtype=np.int64)
    marker_cell_sp2 = np.asarray(seprate_cells_result[0], dtype=np.int64)

    for i in range(marker_cell_sp1.shape[0]):
        cell = marker_cell_sp1[0][i]
        mesh_pos[0]


def parse_reshape(ds):
    rearrange_frame = {}
    re_mesh_pos = np.arange(1)
    re_node_type = np.arange(1)
    re_velocity = np.arange(1)
    re_cells = np.arange(1)
    re_density = np.arange(1)
    re_pressure = np.arange(1)
    count = 0
    for index, d in enumerate(ds):
        if count == 0:
            re_mesh_pos = np.expand_dims(d["mesh_pos"].numpy(), axis=0)
            re_node_type = np.expand_dims(d["node_type"].numpy(), axis=0)
            re_velocity = np.expand_dims(d["velocity"].numpy(), axis=0)
            re_cells = np.expand_dims(d["cells"].numpy(), axis=0)
            re_density = np.expand_dims(d["density"].numpy(), axis=0)
            re_pressure = np.expand_dims(d["pressure"].numpy(), axis=0)
            count += 1
            print("No.{0} has been added to the dict".format(index))
        else:
            re_mesh_pos = np.insert(re_mesh_pos, index, d["mesh_pos"].numpy(), axis=0)
            re_node_type = np.insert(
                re_node_type, index, d["node_type"].numpy(), axis=0
            )
            re_velocity = np.insert(re_velocity, index, d["velocity"].numpy(), axis=0)
            re_cells = np.insert(re_cells, index, d["cells"].numpy(), axis=0)
            re_density = np.insert(re_density, index, d["density"].numpy(), axis=0)
            re_pressure = np.insert(re_pressure, index, d["pressure"].numpy(), axis=0)
            print("No.{0} has been added to the dict".format(index))
    rearrange_frame["node_type"] = re_node_type
    rearrange_frame["cells"] = re_cells
    rearrange_frame["mesh_pos"] = re_mesh_pos
    rearrange_frame["density"] = re_density
    rearrange_frame["pressure"] = re_pressure
    rearrange_frame["velocity"] = re_velocity
    print("done")
    return rearrange_frame


def reorder_face(mesh_pos, edges, plot=False):
    senders = edges[:, 0]
    receivers = edges[:, 1]

    edge_vec = torch.index_select(mesh_pos, 0, senders) - torch.index_select(
        mesh_pos, 0, receivers
    )
    e_x = torch.cat(
        (torch.ones(edge_vec.shape[0], 1), (torch.zeros(edge_vec.shape[0], 1))), dim=1
    )

    edge_vec_dot_ex = edge_vec[:, 0] * e_x[:, 0] + edge_vec[:, 1] * e_x[:, 1]

    edge_op = torch.logical_or(
        edge_vec_dot_ex > 0, torch.full(edge_vec_dot_ex.shape, False)
    )
    edge_op = torch.stack((edge_op, edge_op), dim=-1)

    edge_op_1 = torch.logical_and(edge_vec[:, 0] == 0, edge_vec[:, 1] > 0)
    edge_op_1 = torch.stack((edge_op_1, edge_op_1), dim=-1)

    unique_edges = torch.stack((senders, receivers), dim=1)
    inverse_unique_edges = torch.stack((receivers, senders), dim=1)

    edge_with_bias = torch.where(
        ((edge_op) | (edge_op_1)), unique_edges, inverse_unique_edges
    )

    if plot:
        plot_edge_direction(mesh_pos, edge_with_bias)

    return edge_with_bias


def plot_edge_direction(mesh_pos, edges, node_pos=None, cells_node=None):
    senders = edges[:, 0]
    receivers = edges[:, 1]

    edge_vec = torch.index_select(mesh_pos, 0, senders) - torch.index_select(
        mesh_pos, 0, receivers
    )
    e_x = torch.cat(
        (torch.ones(edge_vec.shape[0], 1), (torch.zeros(edge_vec.shape[0], 1))), dim=1
    )
    e_y = torch.cat(
        (torch.zeros(edge_vec.shape[0], 1), (torch.ones(edge_vec.shape[0], 1))), dim=1
    )

    edge_vec_dot_ex = edge_vec[:, 0] * e_x[:, 0] + edge_vec[:, 1] * e_x[:, 1]
    edge_vec_dot_ey = edge_vec[:, 0] * e_y[:, 0] + edge_vec[:, 1] * e_y[:, 1]

    cosu = edge_vec_dot_ex / ((torch.norm(edge_vec, dim=1) * torch.norm(e_x, dim=1)))
    cosv = edge_vec_dot_ey / ((torch.norm(edge_vec, dim=1) * torch.norm(e_y, dim=1)))

    if cells_node is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.cla()
        ax.set_aspect("equal")
        triang = mtri.Triangulation(node_pos[:, 0], node_pos[:, 1], cells_node)
        ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

    plt.quiver(
        torch.index_select(mesh_pos[:, 0:1], 0, senders),
        torch.index_select(mesh_pos[:, 1:2], 0, senders),
        edge_vec[:, 0],
        edge_vec[:, 1],
        units="height",
        color="red",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.0025,
        headlength=3,
        headwidth=2,
        headaxislength=4.5,
    )
    plt.savefig("edge_dir.png")
    plt.show()


def triangles_to_faces(faces, mesh_pos):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = torch.cat(
        (faces[:, 0:2], faces[:, 1:3], torch.stack((faces[:, 2], faces[:, 0]), dim=1)),
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
        "cell_three_faces": packed_edges,
    }


def create_neighbor_matrix(vertex_coords, edges):
    """
    Create a matrix representing the neighbors for each vertex in a graph.

    Parameters:
    vertex_coords (Tensor): A tensor of shape [n, 2] representing n vertex coordinates.
    edges (Tensor): A tensor of shape [m, 2] representing m edges, where each edge is a pair of vertex indices.

    Returns:
    Tensor: A matrix where each row corresponds to a vertex and contains the indices of its neighbors.
    """
    # Adjust edges to ensure all indices are within the range of vertex_coords' first dimension
    edges_mod = edges % vertex_coords.shape[0]

    # Create a tensor to hold the counts of neighbors for each vertex
    counts = torch.zeros(vertex_coords.shape[0], dtype=torch.int64)

    # Count the occurrence of each index in edges_mod to determine the number of neighbors
    counts.scatter_add_(0, edges_mod.view(-1), torch.ones_like(edges_mod.view(-1)))

    # Find the maximum number of neighbors to define the second dimension of the neighbor matrix
    max_neighbors = counts.max()

    # Create a tensor to hold the neighbors, initialized with -1 (indicating no neighbor)
    neighbor_matrix = torch.full(
        (vertex_coords.shape[0], max_neighbors), -1, dtype=torch.int64
    )

    # Create an array to keep track of the current count of neighbors for each vertex
    current_count = torch.zeros(vertex_coords.shape[0], dtype=torch.int64)

    # Iterate through each edge and populate the neighbor matrix
    for edge in edges_mod:
        # Unpack the edge
        start, end = edge
        # Place the end vertex in the next available spot for the start vertex in neighbor_matrix
        neighbor_matrix[start, current_count[start]] = end
        # Increment the count for the start vertex
        current_count[start] += 1
        # Do the same for the end vertex, assuming undirected edges
        neighbor_matrix[end, current_count[end]] = start
        current_count[end] += 1

    return neighbor_matrix, max_neighbors


def generate_directed_edges(cells_node):
    # 生成给定单元的所有可能边组合，但只保留一个方向的边
    edges = []
    for i in range(len(cells_node)):
        for j in range(i + 1, len(cells_node)):
            edge = [cells_node[i], cells_node[j]]
            reversed_edge = [cells_node[j], cells_node[i]]

            # 只添加一个方向的边
            if reversed_edge not in edges:
                edges.append(edge)
    return edges


def compose_edge_index_x(face_node, cells_face_node_biased, cells_node, cells_index):
    face_node_x = face_node.clone()

    for i in range(cells_index.max() + 1):
        mask_cell = (cells_index == i).view(-1)
        current_cells_face_node_biased = cells_face_node_biased[mask_cell]
        current_cells_node = cells_node[mask_cell]
        all_possible_edges, _ = torch.tensor(
            generate_directed_edges(current_cells_node)
        ).sort(dim=-1)

        for edge in all_possible_edges:
            edge = edge.unsqueeze(0)
            if (edge.unsqueeze(0) == current_cells_face_node_biased).all(
                dim=-1
            ).sum() < 1:
                face_node_x = torch.cat((face_node_x, edge), dim=0)

    return face_node_x


def make_edges_unique(cells_face_node, cells_node, cells_index):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    cells_face_node_biased = torch.sort(cells_face_node, dim=1)[0]
    senders, receivers = cells_face_node_biased[:, 0], cells_face_node_biased[:, 1]
    packed_edges = torch.stack((senders, receivers), dim=1)
    unique_edges = torch.unique(
        packed_edges, return_inverse=False, return_counts=False, dim=0
    )
    senders_unique = unique_edges[:, 0].to(torch.int64)
    receivers_unique = unique_edges[:, 1].to(torch.int64)

    two_way_connectivity = torch.stack(
        (
            torch.cat((senders_unique, receivers_unique), dim=0),
            torch.cat((receivers_unique, senders_unique), dim=0),
        ),
        dim=-1,
    )

    face_node_x = compose_edge_index_x(
        face_node=unique_edges,
        cells_face_node_biased=cells_face_node_biased,
        cells_node=cells_node,
        cells_index=cells_index,
    )

    return {
        "two_way_connectivity": two_way_connectivity,
        "senders": senders,
        "receivers": receivers,
        "unique_edges": unique_edges,
        "face_with_bias": unique_edges,
        "cells_face_node_unbiased": cells_face_node,
        "cells_face_node_biased": packed_edges,
        "face_node_x": face_node_x,
    }


# This function is compromised to Tobias Paffs`s datasets
def mask_face_bonudary(
    face_types, faces, velocity_on_node, pressure_on_node, is_train=False
):
    if is_train:
        velocity_on_face = (
            (
                torch.index_select(velocity_on_node, 0, faces[0])
                + torch.index_select(velocity_on_node, 0, faces[1])
            )
            / 2.0
        ).numpy()
        pressure_on_face = (
            (
                torch.index_select(pressure_on_node, 0, faces[0])
                + torch.index_select(pressure_on_node, 0, faces[1])
            )
            / 2.0
        ).numpy()

    else:
        velocity_on_face = (
            torch.index_select(
                torch.from_numpy(velocity_on_node), 1, torch.from_numpy(faces[0])
            )
            + torch.index_select(
                torch.from_numpy(velocity_on_node), 1, torch.from_numpy(faces[1])
            )
        ) / 2.0
        pressure_on_face = (
            torch.index_select(
                torch.from_numpy(pressure_on_node), 1, torch.from_numpy(faces[0])
            )
            + torch.index_select(
                torch.from_numpy(pressure_on_node), 1, torch.from_numpy(faces[1])
            )
        ) / 2.0
        """
    face_types = torch.from_numpy(face_types)
    mask_of_p = torch.zeros_like(pressure_on_face)
    mask_of_v = torch.zeros_like(velocity_on_face)
    pressure_on_face_t = torch.where((face_types==NodeType.OUTFLOW)|(face_types==NodeType.INFLOW),pressure_on_face,mask_of_p)
    face_types = face_types.repeat(1,1)
    velocity_on_face_t = torch.where((face_types==NodeType.OUTFLOW)|(face_types==NodeType.INFLOW),velocity_on_face,mask_of_v).repeat(1,3)
    """
    return torch.cat((velocity_on_face, pressure_on_face), dim=2).numpy()


def direction_bias(dataset):
    mesh_pos = dataset["mesh_pos"][0]
    edge_vec = dataset["face"]


def renum_data(dataset, unorder=True, index=0, plot=None):
    fig = plot[1]
    ax = plot[2]
    plot = plot[0]
    re_index = np.linspace(
        0, int(dataset["mesh_pos"].shape[1]) - 1, int(dataset["mesh_pos"].shape[1])
    ).astype(np.int64)
    re_cell_index = np.linspace(
        0, int(dataset["cells"].shape[1]) - 1, int(dataset["cells"].shape[1])
    ).astype(np.int64)
    key_list = []
    new_dataset = {}
    for key, value in dataset.items():
        dataset[key] = torch.from_numpy(value)
        key_list.append(key)

    new_dataset = {}
    cells_node = dataset["cells"][0]
    dataset["centroid"] = np.zeros((cells_node.shape[0], 2), dtype=np.float64)
    for index_c in range(cells_node.shape[0]):
        cell = cells_node[index_c]
        centroid_x = 0.0
        centroid_y = 0.0
        for j in range(3):
            centroid_x += dataset["mesh_pos"].numpy()[0][cell[j]][0]
            centroid_y += dataset["mesh_pos"].numpy()[0][cell[j]][1]
        dataset["centroid"][index_c] = np.array(
            [centroid_x / 3, centroid_y / 3], dtype=np.float64
        )
    dataset["centroid"] = torch.from_numpy(np.expand_dims(dataset["centroid"], axis=0))

    for key, value in dataset.items():
        dataset[key] = value.numpy()

    # if unorder:
    #   np.random.shuffle(re_index)
    #   np.random.shuffle(re_cell_index)
    #   for key in key_list:
    #     value = dataset[key]
    #     if key=='cells':
    #       # TODO: cells_node is not correct, need implementation
    #       new_dataset[key]=torch.index_select(value,1,torch.from_numpy(re_cell_index).to(torch.long))
    #     elif  key=='boundary':
    #       new_dataset[key]=value
    #     else:
    #       new_dataset[key] = torch.index_select(value,1,torch.from_numpy(re_index).to(torch.long))
    #   cell_renum_dict = {}
    #   new_cells = np.empty_like(dataset['cells'][0])
    #   for i in range(new_dataset['mesh_pos'].shape[1]):
    #     cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i

    #   for j in range(dataset['cells'].shape[1]):
    #     cell = new_dataset['cells'][0][j]
    #     for node_index in range(cell.shape[0]):
    #       new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
    #   new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 )
    #   new_dataset['cells'] = torch.from_numpy(new_cells)

    #   cells_node = new_dataset['cells'][0]
    #   mesh_pos = new_dataset['mesh_pos']
    #   new_dataset['centroid'] = ((torch.index_select(mesh_pos,1,cells_node[:,0])+torch.index_select(mesh_pos,1,cells_node[:,1])+torch.index_select(mesh_pos,1,cells_node[:,2]))/3.).view(1,-1,2)
    #   for key,value in new_dataset.items():
    #     dataset[key] = value.numpy()
    #     new_dataset[key] = value.numpy()

    # else:

    #   data_cell_centroid = dataset['centroid'].to(torch.float64)[0]
    #   data_cell_Z = -4*data_cell_centroid[:,0]**(2)+data_cell_centroid[:,0]+data_cell_centroid[:,1]+3*data_cell_centroid[:,0]*data_cell_centroid[:,1]-2*data_cell_centroid[:,1]**(2)+20000.
    #   data_node_pos = dataset['mesh_pos'].to(torch.float64)[0]
    #   data_Z = -4*data_node_pos[:,0]**(2)+data_node_pos[:,0]+data_node_pos[:,1]+3*data_node_pos[:,0]*data_node_pos[:,1]-2*data_node_pos[:,1]**(2)+20000.
    #   a = np.unique(data_Z.cpu().numpy(), return_counts=True)
    #   b = np.unique(data_cell_Z.cpu().numpy(), return_counts=True)
    #   if a[0].shape[0] !=data_Z.shape[0] or b[0].shape[0] !=data_cell_Z.shape[0]:
    #     data_cell_Z = data_cell_centroid
    #     data_Z = data_node_pos
    #     print("data{0} renum faild, please consider change the projection function".format(index))

    #   sorted_data_Z,new_data_index = torch.sort(data_Z,descending=False)
    #   sorted_data_cell_Z,new_data_cell_index = torch.sort(data_cell_Z,descending=False)
    #   for key in key_list:
    #     value = dataset[key]
    #     if key=='cells':
    #       new_dataset[key]=torch.index_select(value,1,new_data_cell_index)
    #     elif key=='boundary':
    #       new_dataset[key]=value
    #     else:
    #       new_dataset[key] = torch.index_select(value,1,new_data_index)
    #   cell_renum_dict = {}
    #   new_cells = np.empty_like(dataset['cells'][0])
    #   for i in range(new_dataset['mesh_pos'].shape[1]):
    #     cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i
    #   for j in range(dataset['cells'].shape[1]):
    #     cell = dataset['cells'][0][j]
    #     for node_index in range(cell.shape[0]):
    #       new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
    #   new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 )
    #   new_dataset['cells'] = torch.index_select(torch.from_numpy(new_cells),1,new_data_cell_index)

    #   cells_node = new_dataset['cells'][0]
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   new_dataset['centroid'] = ((torch.index_select(mesh_pos,0,cells_node[:,0])+torch.index_select(mesh_pos,0,cells_node[:,1])+torch.index_select(mesh_pos,0,cells_node[:,2]))/3.).view(1,-1,2)
    #   for key,value in new_dataset.items():
    #     dataset[key] = value.numpy()
    #     new_dataset[key] = value.numpy()
    #   #new_dataset = reorder_boundaryu_to_front(dataset)
    # if plot is not None and plot=='cell':
    #   # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #   ax.cla()
    #   ax.set_aspect('equal')
    #   #bb_min = mesh['velocity'].min(axis=(0, 1))
    #   #bb_max = mesh['velocity'].max(axis=(0, 1))
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   faces = new_dataset['cells'][0]
    #   triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #   #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #   ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #   #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    #   plt.show()
    # elif plot is not None and plot=='node':
    #   # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #   ax.cla()
    #   ax.set_aspect('equal')
    #   #bb_min = mesh['velocity'].min(axis=(0, 1))
    #   #bb_max = mesh['velocity'].max(axis=(0, 1))
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   #faces = dataset['cells'][0]
    #   #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #   #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #   #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #   plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    #   plt.show()

    # elif plot is not None and plot=='centroid':
    #   # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #   ax.cla()
    #   ax.set_aspect('equal')
    #   #bb_min = mesh['velocity'].min(axis=(0, 1))
    #   #bb_max = mesh['velocity'].max(axis=(0, 1))
    #   mesh_pos = new_dataset['centroid'][0]
    #   #faces = dataset['cells'][0]
    #   #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #   #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #   #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #   plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    #   plt.show()

    # elif plot is not None and plot=='plot_order':
    #   fig = plt.figure()  # 创建画布
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   centroid = new_dataset['centroid'][0]
    #   display_centroid_list=[centroid[0],centroid[1],centroid[2]]
    #   display_pos_list=[mesh_pos[0],mesh_pos[1],mesh_pos[2]]
    #   ax1 = fig.add_subplot(211)
    #   ax2 = fig.add_subplot(212)

    #   def animate(num):

    #     if num < mesh_pos.shape[0]:
    #       display_pos_list.append(mesh_pos[num])
    #     display_centroid_list.append(centroid[num])
    #     if num%3 ==0 and num >0:
    #         display_pos = np.array(display_pos_list)
    #         display_centroid = np.array(display_centroid_list)
    #         p1 = ax1.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    #         ax1.legend(['node_pos'], loc=2, fontsize=10)
    #         p2 = ax2.scatter(display_centroid[:,0],display_centroid[:,1],c='green',linewidths=1)
    #         ax2.legend(['centroid'], loc=2, fontsize=10)
    #   ani = animation.FuncAnimation(fig, animate, frames=new_dataset['centroid'][0].shape[0], interval=100)
    #   if unorder:
    #     ani.save("unorder"+"test.gif", writer='pillow')
    #   else:
    #     ani.save("order"+"test.gif", writer='pillow')

    return dataset, True


def reorder_boundaryu_to_front(dataset, plot=None):
    boundary_attributes = {}

    node_type = torch.from_numpy(dataset["node_type"][0])[:, 0]
    face_type = torch.from_numpy(dataset["face_type"][0])[:, 0]
    cells_type = torch.from_numpy(dataset["cells_type"][0])[:, 0]

    node_mask_t = torch.full(node_type.shape, True)
    node_mask_i = torch.logical_not(node_mask_t)
    face_mask_t = torch.full(face_type.shape, True)
    face_mask_i = torch.logical_not(face_mask_t)
    cells_mask_t = torch.full(cells_type.shape, True)
    cells_mask_i = torch.logical_not(cells_mask_t)

    node_mask = torch.where(node_type == NodeType.NORMAL, node_mask_t, node_mask_i)
    face_mask = torch.where(face_type == NodeType.NORMAL, face_mask_t, face_mask_i)
    cells_mask = torch.where(cells_type == NodeType.NORMAL, cells_mask_t, cells_mask_i)

    boundary_node_mask = torch.logical_not(node_mask)
    boundary_face_mask = torch.logical_not(face_mask)
    boundary_cells_mask = torch.logical_not(cells_mask)

    """boundary attributes"""
    for key, value in dataset.items():
        if key == "mesh_pos":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "target|velocity_on_node":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "target|pressure_on_node":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "node_type":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_node":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "centroid":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "face":
            boundary_attributes = value[:, :, boundary_face_mask]
            Interior_attributes = value[:, :, face_mask]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=2
            )
        elif key == "face_length":
            boundary_attributes = value[:, boundary_face_mask, :]
            Interior_attributes = value[:, face_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "face_type":
            boundary_attributes = value[:, boundary_face_mask, :]
            Interior_attributes = value[:, face_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_face":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_type":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "unit_norm_v":
            boundary_attributes = value[:, boundary_cells_mask, :, :]
            Interior_attributes = value[:, cells_mask, :, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "neighbour_cell":
            boundary_attributes = value[:, :, boundary_face_mask]
            Interior_attributes = value[:, :, face_mask]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=2
            )
        elif key == "cell_factor":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_area":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )

    if plot is not None and plot == "cell":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.cla()
        ax.set_aspect("equal")
        # bb_min = mesh['velocity'].min(axis=(0, 1))
        # bb_max = mesh['velocity'].max(axis=(0, 1))
        mesh_pos = dataset["mesh_pos"][0]
        faces = dataset["cells"][0]
        triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)
        # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
        ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
        # plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
        plt.show()
    elif plot is not None and plot == "node":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.cla()
        ax.set_aspect("equal")
        # bb_min = mesh['velocity'].min(axis=(0, 1))
        # bb_max = mesh['velocity'].max(axis=(0, 1))
        mesh_pos = dataset["mesh_pos"][0]
        # faces = dataset['cells'][0]
        # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
        # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
        # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1], c="red", linewidths=1)
        plt.show()

    elif plot is not None and plot == "centroid":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.cla()
        ax.set_aspect("equal")
        # bb_min = mesh['velocity'].min(axis=(0, 1))
        # bb_max = mesh['velocity'].max(axis=(0, 1))
        mesh_pos = dataset["centroid"][0]
        # faces = dataset['cells'][0]
        # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
        # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
        # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1], c="red", linewidths=1)
        plt.show()

    elif plot is not None and plot == "plot_order":
        fig = plt.figure()  # 创建画布
        mesh_pos = dataset["mesh_pos"][0]
        centroid = dataset["centroid"][0]
        display_centroid_list = [centroid[0], centroid[1], centroid[2]]
        display_pos_list = [mesh_pos[0], mesh_pos[1], mesh_pos[2]]
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        def animate(num):
            if num < mesh_pos.shape[0]:
                display_pos_list.append(mesh_pos[num])
            display_centroid_list.append(centroid[num])
            if num % 3 == 0 and num > 0:
                display_pos = np.array(display_pos_list)
                display_centroid = np.array(display_centroid_list)
                p1 = ax1.scatter(
                    display_pos[:, 0], display_pos[:, 1], c="red", linewidths=1
                )
                ax1.legend(["node_pos"], loc=2, fontsize=10)
                p2 = ax2.scatter(
                    display_centroid[:, 0],
                    display_centroid[:, 1],
                    c="green",
                    linewidths=1,
                )
                ax2.legend(["centroid"], loc=2, fontsize=10)

        ani = animation.FuncAnimation(
            fig, animate, frames=dataset["centroid"][0].shape[0], interval=100
        )
        plt.show(block=True)
        ani.save("order" + "test.gif", writer="pillow")
    return dataset


def calc_symmetry_pos(two_pos, vertex):
    """(x1,y1),(x2,y2)是两点式确定的直线,(x3,y3)是需要被计算的顶点"""
    x1 = two_pos[0][0]
    y1 = two_pos[0][1]
    x2 = two_pos[1][0]
    y2 = two_pos[1][1]
    x3 = vertex[0]
    y3 = vertex[1]
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - y1 * x2
    x4 = x3 - 2 * A * ((A * x3 + B * y3 + C) / (A * A + B * B))
    y4 = y3 - 2 * B * ((A * x3 + B * y3 + C) / (A * A + B * B))

    return (x4, y4)


def is_between(span: tuple, x):
    """
    Determine if a value x is between two other values a and b.

    Parameters:
    - a (float or int): The lower bound.
    - b (float or int): The upper bound.
    - x (float or int): The value to check.

    Returns:
    - (bool): True if x is between a and b (inclusive), False otherwise.
    """
    a, b = span
    # Check if x is between a and b, inclusive
    if a <= x <= b or b <= x <= a:
        return True
    else:
        return False


def whether_corner(node, pos_to_edge_index):
    """which can determine whether a vertex is a corner vertex in rectangular domain"""

    neighbour_edge_type = torch.cat(pos_to_edge_index[str(node.numpy())])
    stastic_face_type = stastic_nodeface_type(neighbour_edge_type)
    if stastic_face_type[NodeType.WALL_BOUNDARY] > 0 and (
        stastic_face_type[NodeType.INFLOW] > 0
        or stastic_face_type[NodeType.OUTFLOW] > 0
    ):
        return True
    else:
        return False

    # if rtval['inlet'] == 1 and rtval['outlet'] == 1 and rtval['topwall'] == 1 and rtval['bottomwall'] == 1:

    # if (is_between(inlet,x) and is_between(topwall,y)) or (is_between(outlet,x) and is_between(topwall,y)):
    #   return True
    # elif (is_between(inlet,x) and is_between(bottomwall,y)) or (is_between(outlet,x) and is_between(bottomwall,y)):
    #   return True
    # else:
    #   return False


def make_ghost_cell(
    fore_dataset,
    domain: list,
    mesh_pos: torch.Tensor,
    cells,
    node_type,
    mode,
    recover_unorder=False,
    limit=None,
    pos_to_edge_index=None,
    index=None,
    fig=None,
    ax=None,
):
    """compose ghost cell, but TODO: corner cell can not be duplicated twice with current cell type"""

    ghost_start_node_index = torch.max(cells)
    new_mesh_pos = mesh_pos.clone()
    new_cells_node = cells.clone()
    new_node_type = node_type.clone()
    for thread in domain:
        cell_thread = thread[0]
        boundary_cells_node = thread[0][0]
        boundary_cell_three_vertex_type = thread[0][1]
        cell_three_vertex_0 = [
            torch.index_select(mesh_pos, 0, boundary_cells_node[:, 0]),
            boundary_cell_three_vertex_type[:, 0],
        ]
        cell_three_vertex_1 = [
            torch.index_select(mesh_pos, 0, boundary_cells_node[:, 1]),
            boundary_cell_three_vertex_type[:, 1],
        ]
        cell_three_vertex_2 = [
            torch.index_select(mesh_pos, 0, boundary_cells_node[:, 2]),
            boundary_cell_three_vertex_type[:, 2],
        ]

        # for every boundary cell thread
        ghost_pos = []
        ghost_node_type = []
        ghost_cells_node = []
        for i in range(boundary_cells_node.shape[0]):
            if cell_thread[2] == "wall":
                if cell_three_vertex_0[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_0[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

                elif cell_three_vertex_1[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_0[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_1[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][0],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

                elif cell_three_vertex_2[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_0[0][i],
                                    ],
                                    cell_three_vertex_2[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][0],
                            ]
                        )
                    )
                    ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

                else:
                    if (
                        cell_three_vertex_0[1][i] == NodeType.INFLOW
                        or cell_three_vertex_0[1][i] == NodeType.OUTFLOW
                    ) and (
                        not (
                            whether_corner(cell_three_vertex_0[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_0[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))
                    elif (
                        cell_three_vertex_1[1][i] == NodeType.INFLOW
                        or cell_three_vertex_1[1][i] == NodeType.OUTFLOW
                    ) and (
                        not (
                            whether_corner(cell_three_vertex_1[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_0[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_1[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][0],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))
                    elif (
                        cell_three_vertex_2[1][i] == NodeType.INFLOW
                        or cell_three_vertex_2[1][i] == NodeType.OUTFLOW
                    ) and (
                        not (
                            whether_corner(cell_three_vertex_2[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_0[0][i],
                                        ],
                                        cell_three_vertex_2[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][0],
                                ]
                            )
                        )
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

            else:
                """inlet thread and outlet thread"""
                if cell_three_vertex_0[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_0[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    if cell_thread[2] == "inlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                    elif cell_thread[2] == "outlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                elif cell_three_vertex_1[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_0[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_1[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][0],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    if cell_thread[2] == "inlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                    elif cell_thread[2] == "outlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                elif cell_three_vertex_2[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_0[0][i],
                                    ],
                                    cell_three_vertex_2[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][0],
                            ]
                        )
                    )
                    if cell_thread[2] == "inlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                    elif cell_thread[2] == "outlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                else:
                    if cell_three_vertex_0[1][i] == NodeType.WALL_BOUNDARY and (
                        not (
                            whether_corner(cell_three_vertex_0[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_0[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        if cell_thread[2] == "inlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                        elif cell_thread[2] == "outlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                    elif cell_three_vertex_1[1][i] == NodeType.WALL_BOUNDARY and (
                        not (
                            whether_corner(cell_three_vertex_1[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_0[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_1[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][0],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        if cell_thread[2] == "inlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                        elif cell_thread[2] == "outlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                    elif cell_three_vertex_2[1][i] == NodeType.WALL_BOUNDARY and (
                        not (
                            whether_corner(cell_three_vertex_2[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_0[0][i],
                                        ],
                                        cell_three_vertex_2[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][0],
                                ]
                            )
                        )
                        if cell_thread[2] == "inlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                        elif cell_thread[2] == "outlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
        try:
            new_mesh_pos = torch.cat((new_mesh_pos, torch.stack(ghost_pos)), dim=0)
            new_cells_node = torch.cat(
                (new_cells_node, torch.stack(ghost_cells_node)), dim=0
            )
            new_node_type = torch.cat(
                (new_node_type, torch.stack(ghost_node_type)), dim=0
            )
        except:
            pass
        new_velocity = torch.cat(
            (
                torch.from_numpy(fore_dataset["velocity"]),
                torch.zeros(
                    (new_mesh_pos.shape[0] - mesh_pos.shape[0], 2), dtype=torch.float64
                )
                .view(1, -1, 2)
                .repeat(fore_dataset["velocity"].shape[0], 1, 1),
            ),
            dim=1,
        )
        new_pressure = torch.cat(
            (
                torch.from_numpy(fore_dataset["pressure"]),
                torch.zeros(
                    (new_mesh_pos.shape[0] - mesh_pos.shape[0], 1), dtype=torch.float64
                )
                .view(1, -1, 1)
                .repeat(fore_dataset["pressure"].shape[0], 1, 1),
            ),
            dim=1,
        )

    """ >>>         plot ghosted boundary node pos           >>>"""
    fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    ax.cla()
    ax.set_aspect("equal")
    # bb_min = mesh['velocity'].min(axis=(0, 1))
    # bb_max = mesh['velocity'].max(axis=(0, 1))
    plt.scatter(
        new_mesh_pos[new_node_type == NodeType.NORMAL, 0],
        new_mesh_pos[new_node_type == NodeType.NORMAL, 1],
        c="red",
        linewidths=1,
        s=20.5,
        zorder=5,
    )
    plt.scatter(
        new_mesh_pos[new_node_type == NodeType.WALL_BOUNDARY, 0],
        new_mesh_pos[new_node_type == NodeType.WALL_BOUNDARY, 1],
        c="green",
        linewidths=1,
        s=20.5,
        zorder=5,
    )
    plt.scatter(
        new_mesh_pos[new_node_type == NodeType.INFLOW, 0],
        new_mesh_pos[new_node_type == NodeType.INFLOW, 1],
        c="blue",
        linewidths=1,
        s=20.5,
        zorder=5,
    )
    plt.scatter(
        new_mesh_pos[new_node_type == NodeType.OUTFLOW, 0],
        new_mesh_pos[new_node_type == NodeType.OUTFLOW, 1],
        c="orange",
        linewidths=1,
        s=20.5,
        zorder=5,
    )

    plt.scatter(
        new_mesh_pos[new_node_type == NodeType.GHOST_WALL, 0],
        new_mesh_pos[new_node_type == NodeType.GHOST_WALL, 1],
        c="cyan",
        linewidths=1,
        s=20.5,
        zorder=5,
    )
    plt.scatter(
        new_mesh_pos[new_node_type == NodeType.GHOST_INFLOW, 0],
        new_mesh_pos[new_node_type == NodeType.GHOST_INFLOW, 1],
        c="yellow",
        linewidths=1,
        s=20.5,
        zorder=5,
    )
    plt.scatter(
        new_mesh_pos[new_node_type == NodeType.GHOST_OUTFLOW, 0],
        new_mesh_pos[new_node_type == NodeType.GHOST_OUTFLOW, 1],
        c="magenta",
        linewidths=1,
        s=20.5,
        zorder=5,
    )

    triang = mtri.Triangulation(new_mesh_pos[:, 0], new_mesh_pos[:, 1], new_cells_node)
    # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    ax.triplot(triang, "ko-", ms=0.5, lw=0.3, zorder=1)
    # plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    plt.savefig("ghosted boundary node pos.png")
    plt.close()
    """ <<<         plot ghosted boundary node pos           <<<"""

    new_domain = {
        "mesh_pos": new_mesh_pos.view(1, -1, 2)
        .repeat(fore_dataset["mesh_pos"].shape[0], 1, 1)
        .numpy(),
        "cells": new_cells_node.view(1, -1, 3)
        .repeat(fore_dataset["cells"].shape[0], 1, 1)
        .numpy(),
        "node_type": new_node_type.view(1, -1, 1)
        .repeat(fore_dataset["node_type"].shape[0], 1, 1)
        .numpy(),
        "velocity": new_velocity.numpy(),
        "pressure": new_pressure.numpy(),
    }
    new_mesh, nodes_of_cell = recover_ghosted_2_fore_mesh(
        new_domain, mode, recover_unorder, limit, index, fig, ax
    )
    return new_mesh, new_domain, nodes_of_cell


def recover_ghosted_2_fore_mesh(
    ghosted_domain,
    mode="cylinder_mesh",
    unorder=False,
    limit=None,
    index=None,
    fig=None,
    ax=None,
):
    dataset = ghosted_domain
    """for ploting"""
    fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    mesh = {}
    mesh["mesh_pos"] = dataset["mesh_pos"][0]
    mesh["cells_node"] = np.sort(dataset["cells"][0], axis=1)
    cells_node = torch.from_numpy(mesh["cells_node"]).long()
    mesh["cells_node"] = np.expand_dims(cells_node.numpy(), axis=0).astype(np.int64)

    """>>>compute centroid crds>>>"""
    mesh_pos = torch.from_numpy(dataset["mesh_pos"][0])
    centroid = (
        mesh_pos[cells_node.T[0]]
        + mesh_pos[cells_node.T[1]]
        + mesh_pos[cells_node.T[2]]
    ) / 3.0
    mesh["centroid"] = centroid.unsqueeze(0).numpy()
    """<<<compute centroid crds<<<"""

    # compose face
    decomposed_cells = triangles_to_faces(cells_node, mesh["mesh_pos"])
    face = decomposed_cells["face_with_bias"]
    # senders = face[:,0]
    # receivers = face[:,1]
    edge_with_bias = decomposed_cells["edge_with_bias"]
    mesh["face"] = face.T.numpy().astype(np.int64)

    # compute face length
    g_tmp = Data(
        pos=torch.from_numpy(mesh["mesh_pos"]),
        edge_index=torch.from_numpy(mesh["face"]).to(torch.long),
    )
    g_with_ED_dist = transformer(g_tmp)
    mesh["face_length"] = g_with_ED_dist.edge_attr.to(torch.float64).numpy()

    # check-out face_type
    face_type = np.zeros((mesh["face"].shape[1], 1), dtype=np.int64)
    a = torch.index_select(
        torch.from_numpy(dataset["node_type"][0]), 0, torch.from_numpy(mesh["face"][0])
    ).numpy()
    b = torch.index_select(
        torch.from_numpy(dataset["node_type"][0]), 0, torch.from_numpy(mesh["face"][1])
    ).numpy()
    face_center_pos = (
        torch.index_select(
            torch.from_numpy(mesh["mesh_pos"]), 0, torch.from_numpy(mesh["face"][0])
        ).numpy()
        + torch.index_select(
            torch.from_numpy(mesh["mesh_pos"]), 0, torch.from_numpy(mesh["face"][1])
        ).numpy()
    ) / 2.0

    mesh_pos = dataset["mesh_pos"][0]
    node_type = dataset["node_type"][0].reshape(-1)

    """ >>>         stastic_ghosted_nodeface_type           >>>"""
    # print("After recoverd ghosted data has node type:")
    # stastic_nodeface_type(node_type)
    """ <<<         stastic_ghosted_nodeface_type           <<<"""

    """ >>>         plot boundary node pos           >>>"""
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # #bb_min = mesh['velocity'].min(axis=(0, 1))
    # #bb_max = mesh['velocity'].max(axis=(0, 1))
    # plt.scatter(mesh_pos[node_type==NodeType.NORMAL,0],mesh_pos[node_type==NodeType.NORMAL,1],c='red',linewidths=1,s=20.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.WALL_BOUNDARY,0],mesh_pos[node_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=20.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.INFLOW,0],mesh_pos[node_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=20.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.OUTFLOW,0],mesh_pos[node_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=20.5,zorder=5)
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
    # #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    # plt.savefig("node distribution.png")
    # plt.close()
    """ <<<         plot boundary node pos           <<<"""

    if mode.find("airfoil") != -1:
        face_type = torch.from_numpy(face_type)
        Airfoil = torch.full(face_type.shape, NodeType.AIRFOIL).to(torch.int64)
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int64)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int64)
        ghost_Airfoil = torch.full(face_type.shape, NodeType.GHOST_AIRFOIL).to(
            torch.int64
        )
        ghost_Inlet = torch.full(face_type.shape, NodeType.GHOST_INFLOW).to(torch.int64)
        a = torch.from_numpy(a).view(-1)
        b = torch.from_numpy(b).view(-1)
        face_type[
            (a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :
        ] = Airfoil[(a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :]
        face_type[
            (a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :
        ] = Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]
        face_type[
            (a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :
        ] = Inlet[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :]
        face_type[
            (a == b) & (a == NodeType.GHOST_INFLOW) & (b == NodeType.GHOST_INFLOW), :
        ] = ghost_Inlet[
            (a == b) & (a == NodeType.GHOST_INFLOW) & (b == NodeType.GHOST_INFLOW), :
        ]
        face_type[
            (a == b) & (a == NodeType.GHOST_AIRFOIL) & (b == NodeType.GHOST_AIRFOIL), :
        ] = ghost_Airfoil[
            (a == b) & (a == NodeType.GHOST_AIRFOIL) & (b == NodeType.GHOST_AIRFOIL), :
        ]

    else:
        # topwall = np.max(face_center_pos[:,1])
        # bottomwall = np.min(face_center_pos[:,1])
        # outlet = np.max(face_center_pos[:,0])
        # inlet = np.min(face_center_pos[:,0])

        """for more robustness"""
        topwall_Lower_limit, topwall_Upper_limit = limit[0]

        bottomwall_Lower_limit, bottomwall_Upper_limit = limit[1]

        outlet_Lower_limit, outlet_Upper_limit = limit[2]

        inlet_Lower_limit, inlet_Upper_limit = limit[3]

        face_type = torch.from_numpy(face_type)
        WALL_BOUNDARY_t = torch.full(face_type.shape, NodeType.WALL_BOUNDARY).to(
            torch.int64
        )
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int64)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int64)
        Outlet = torch.full(face_type.shape, NodeType.OUTFLOW).to(torch.int64)
        ghost_WALL_BOUNDARY_t = torch.full(face_type.shape, NodeType.GHOST_WALL).to(
            torch.int64
        )
        ghost_Inlet = torch.full(face_type.shape, NodeType.GHOST_INFLOW).to(torch.int64)
        ghost_Outlet = torch.full(face_type.shape, NodeType.GHOST_OUTFLOW).to(
            torch.int64
        )
        a = torch.from_numpy(a).view(-1)
        b = torch.from_numpy(b).view(-1)

        """ Without considering the corner points """
        face_type[
            (a == b) & (a == NodeType.WALL_BOUNDARY) & (b == NodeType.WALL_BOUNDARY), :
        ] = WALL_BOUNDARY_t[
            (a == b) & (a == NodeType.WALL_BOUNDARY) & (b == NodeType.WALL_BOUNDARY), :
        ]
        face_type[
            (a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :
        ] = Inlet[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :]
        face_type[
            (a == b) & (a == NodeType.OUTFLOW) & (b == NodeType.OUTFLOW), :
        ] = Outlet[(a == b) & (a == NodeType.OUTFLOW) & (b == NodeType.OUTFLOW), :]
        face_type[
            (a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :
        ] = Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]

        face_type[
            (a == NodeType.GHOST_WALL) | (b == NodeType.GHOST_WALL), :
        ] = ghost_WALL_BOUNDARY_t[
            (a == NodeType.GHOST_WALL) | (b == NodeType.GHOST_WALL), :
        ]
        face_type[
            (a == NodeType.GHOST_INFLOW) | (b == NodeType.GHOST_INFLOW), :
        ] = ghost_Inlet[(a == NodeType.GHOST_INFLOW) | (b == NodeType.GHOST_INFLOW), :]
        face_type[
            (a == NodeType.GHOST_OUTFLOW) | (b == NodeType.GHOST_OUTFLOW), :
        ] = ghost_Outlet[
            (a == NodeType.GHOST_OUTFLOW) | (b == NodeType.GHOST_OUTFLOW), :
        ]

        """ Use position relationship to regulate the corner points """
        face_type[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.INFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.INFLOW))
            ),
            :,
        ] = WALL_BOUNDARY_t[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.INFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.INFLOW))
            ),
            :,
        ]

        face_type[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.OUTFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.OUTFLOW))
            ),
            :,
        ] = WALL_BOUNDARY_t[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.OUTFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.OUTFLOW))
            ),
            :,
        ]

    mesh["face_type"] = face_type
    """ >>>         stastic_ghosted_nodeface_type           >>>"""
    # print("After recoverd ghosted data has face type:")
    # stastic_nodeface_type(face_type)
    """ <<<         stastic_ghosted_nodeface_type           <<<"""

    """ >>>         plot boundary face center pos           >>>"""
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node.view(-1,3))
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.NORMAL,0],face_center_pos[face_type[:,0]==NodeType.NORMAL,1],c='red',linewidths=1,s=20,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.WALL_BOUNDARY,0],face_center_pos[face_type[:,0]==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=20,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.INFLOW,0],face_center_pos[face_type[:,0]==NodeType.INFLOW,1],c='blue',linewidths=1,s=20,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.OUTFLOW,0],face_center_pos[face_type[:,0]==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=20,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.GHOST_WALL,0],face_center_pos[face_type[:,0]==NodeType.GHOST_WALL,1],c='cyan',linewidths=1,s=20,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.GHOST_OUTFLOW,0],face_center_pos[face_type[:,0]==NodeType.GHOST_OUTFLOW,1],c='magenta',linewidths=1,s=20,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.GHOST_INFLOW,0],face_center_pos[face_type[:,0]==NodeType.GHOST_INFLOW,1],c='teal',linewidths=1,s=20,zorder=5)
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
    # plt.savefig("ghosted face distribution.png")
    # plt.close()
    """ <<<         plot boundary face center pos           <<<"""

    # compute cell_face index and cells_type
    face_list = torch.from_numpy(mesh["face"]).transpose(0, 1).numpy()
    face_index = {}
    for i in range(face_list.shape[0]):
        face_index[str(face_list[i])] = i
    nodes_of_cell = torch.stack(torch.chunk(edge_with_bias, 3, 0), dim=1)

    nodes_of_cell = nodes_of_cell.numpy()
    edges_of_cell = np.ones(
        (nodes_of_cell.shape[0], nodes_of_cell.shape[1]), dtype=np.int64
    )
    cells_type = np.zeros((nodes_of_cell.shape[0], 1), dtype=np.int64)

    for i in range(nodes_of_cell.shape[0]):
        three_face_index = [
            face_index[str(nodes_of_cell[i][0])],
            face_index[str(nodes_of_cell[i][1])],
            face_index[str(nodes_of_cell[i][2])],
        ]
        three_face_type = [
            face_type[three_face_index[0]],
            face_type[three_face_index[1]],
            face_type[three_face_index[2]],
        ]
        INFLOW_t = 0
        WALL_BOUNDARY_t = 0
        OUTFLOW_t = 0
        AIRFOIL_t = 0
        NORMAL_t = 0
        ghost_INFLOW_t = 0
        ghost_WALL_BOUNDARY_t = 0
        ghost_OUTFLOW_t = 0
        ghost_AIRFOIL_t = 0
        for type in three_face_type:
            if type == NodeType.INFLOW:
                INFLOW_t += 1
            elif type == NodeType.WALL_BOUNDARY:
                WALL_BOUNDARY_t += 1
            elif type == NodeType.OUTFLOW:
                OUTFLOW_t += 1
            elif type == NodeType.AIRFOIL:
                AIRFOIL_t += 1
            elif type == NodeType.GHOST_INFLOW:
                ghost_INFLOW_t += 1
            elif type == NodeType.GHOST_WALL:
                ghost_WALL_BOUNDARY_t += 1
            elif type == NodeType.GHOST_OUTFLOW:
                ghost_OUTFLOW_t += 1
            elif type == NodeType.GHOST_AIRFOIL:
                ghost_AIRFOIL_t += 1
            else:
                NORMAL_t += 1
        if ghost_INFLOW_t > 0:
            cells_type[i] = NodeType.GHOST_INFLOW
        elif ghost_WALL_BOUNDARY_t > 0:
            cells_type[i] = NodeType.GHOST_WALL
        elif ghost_OUTFLOW_t > 0:
            cells_type[i] = NodeType.GHOST_OUTFLOW
        elif ghost_AIRFOIL_t > 0:
            cells_type[i] = NodeType.GHOST_AIRFOIL

        # elif INFLOW_t>0 and WALL_BOUNDARY_t>0 and NORMAL_t>0: # left top vertx corner boundary(both wall and inflow)
        #   cells_type[i] = NodeType.IN_WALL

        # elif OUTFLOW_t>0 and WALL_BOUNDARY_t>0 and NORMAL_t>0: # right bottom vertx corner boundary(both wall and outflow)
        #   cells_type[i] = NodeType.OUT_WALL

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t == 0 and INFLOW_t > 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t == 0 and INFLOW_t == 0 and OUTFLOW_t > 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t > 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t > 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif AIRFOIL_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.AIRFOIL

        elif INFLOW_t > 0 and NORMAL_t > 0 and WALL_BOUNDARY_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.INFLOW

        elif OUTFLOW_t > 0 and NORMAL_t > 0 and WALL_BOUNDARY_t == 0 and INFLOW_t == 0:
            cells_type[i] = NodeType.OUTFLOW
        else:
            cells_type[i] = NodeType.NORMAL
        for j in range(3):
            single_face_index = face_index[str(nodes_of_cell[i][j])]
            edges_of_cell[i][j] = single_face_index
    mesh["cells_face"] = edges_of_cell
    mesh["cells_type"] = cells_type

    """ >>>         stastic_ghosted_nodeface_type           >>>"""
    print("After recoverd ghosted data has cell type:")
    stastic_type = stastic_nodeface_type(cells_type)
    # if stastic_type[NodeType.INFLOW]+1!=stastic_type[NodeType.GHOST_INFLOW] or stastic_type[NodeType.OUTFLOW]!=stastic_type[NodeType.GHOST_OUTFLOW] or stastic_type[NodeType.WALL_BOUNDARY]!=stastic_type[NodeType.GHOST_WALL]:
    #   raise ValueError("check ghosted result, try to plot it")
    """ <<<         stastic_ghosted_nodeface_type           <<<"""

    """ >>>         plot ghosted boundary cell center pos           >>>"""
    # centroid = mesh['centroid'][0]
    # if (len(cells_type.shape)>1)and (len(cells_type.shape)<3):
    #   cells_type = cells_type.reshape(-1)
    # else:
    #   raise ValueError("chk cells_type dim")
    # plt.close()
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # plt.scatter(centroid[cells_type==NodeType.NORMAL,0],centroid[cells_type==NodeType.NORMAL,1],c='red',linewidths=1,s=20,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.WALL_BOUNDARY,0],centroid[cells_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=20,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.OUTFLOW,0],centroid[cells_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=20,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.INFLOW,0],centroid[cells_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=20,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.GHOST_WALL,0],centroid[cells_type==NodeType.GHOST_WALL,1],c='cyan',linewidths=1,s=20,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.GHOST_OUTFLOW,0],centroid[cells_type==NodeType.GHOST_OUTFLOW,1],c='magenta',linewidths=1,s=20,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.GHOST_INFLOW,0],centroid[cells_type==NodeType.GHOST_INFLOW,1],c='teal',linewidths=1,s=20,zorder=5)
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # bb_min = dataset['velocity'].min(axis=(0, 1))
    # bb_max = dataset['velocity'].max(axis=(0, 1))
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
    # plt.savefig("ghosted cell center distribution"+str(index)+".png")
    # plt.close()
    """ <<<         plot ghosted boundary cell center pos           <<<"""

    return mesh, nodes_of_cell


def parse_origin_dataset(dataset, unorder=False, index_num=0, plot=None, writer=None):
    re_index = np.linspace(
        0, int(dataset["mesh_pos"].shape[1]) - 1, int(dataset["mesh_pos"].shape[1])
    ).astype(np.int64)
    re_cell_index = np.linspace(
        0, int(dataset["cells"].shape[1]) - 1, int(dataset["cells"].shape[1])
    ).astype(np.int64)
    key_list = []
    new_dataset = {}
    for key, value in dataset.items():
        dataset[key] = torch.from_numpy(value)
        key_list.append(key)

    new_dataset = {}
    cells_node = dataset["cells"][0]
    dataset["centroid"] = np.zeros((cells_node.shape[0], 2), dtype=np.float64)
    for index_c in range(cells_node.shape[0]):
        cell = cells_node[index_c]
        centroid_x = 0.0
        centroid_y = 0.0
        for j in range(3):
            centroid_x += dataset["mesh_pos"].numpy()[0][cell[j]][0]
            centroid_y += dataset["mesh_pos"].numpy()[0][cell[j]][1]
        dataset["centroid"][index_c] = np.array(
            [centroid_x / 3, centroid_y / 3], dtype=np.float64
        )
    dataset["centroid"] = torch.from_numpy(np.expand_dims(dataset["centroid"], axis=0))

    if unorder:
        np.random.shuffle(re_index)
        np.random.shuffle(re_cell_index)
        for key in key_list:
            value = dataset[key]
            if key == "cells":
                # TODO: cells_node is not correct, need implementation
                new_dataset[key] = torch.index_select(
                    value, 1, torch.from_numpy(re_cell_index).to(torch.long)
                )
            elif key == "boundary":
                new_dataset[key] = value
            else:
                new_dataset[key] = torch.index_select(
                    value, 1, torch.from_numpy(re_index).to(torch.long)
                )
        cell_renum_dict = {}
        new_cells = np.empty_like(dataset["cells"][0])
        for i in range(new_dataset["mesh_pos"].shape[1]):
            cell_renum_dict[str(new_dataset["mesh_pos"][0][i].numpy())] = i

        for j in range(dataset["cells"].shape[1]):
            cell = new_dataset["cells"][0][j]
            for node_index in range(cell.shape[0]):
                new_cells[j][node_index] = cell_renum_dict[
                    str(dataset["mesh_pos"][0].numpy()[cell[node_index]])
                ]
        new_cells = np.repeat(
            np.expand_dims(new_cells, axis=0), dataset["cells"].shape[0], axis=0
        )
        new_dataset["cells"] = torch.from_numpy(new_cells)

        cells_node = new_dataset["cells"][0]
        mesh_pos = new_dataset["mesh_pos"]
        new_dataset["centroid"] = (
            (
                torch.index_select(mesh_pos, 1, cells_node[:, 0])
                + torch.index_select(mesh_pos, 1, cells_node[:, 1])
                + torch.index_select(mesh_pos, 1, cells_node[:, 2])
            )
            / 3.0
        ).view(1, -1, 2)
        for key, value in new_dataset.items():
            dataset[key] = value.numpy()
            new_dataset[key] = value.numpy()

    else:
        data_cell_centroid = dataset["centroid"].to(torch.float64)[0]
        data_cell_Z = (
            -8 * data_cell_centroid[:, 0] ** (2)
            + 3 * data_cell_centroid[:, 0] * data_cell_centroid[:, 1]
            - 2 * data_cell_centroid[:, 1] ** (2)
            + 20.0
        )
        data_node_pos = dataset["mesh_pos"].to(torch.float64)[0]
        data_Z = (
            -8 * data_node_pos[:, 0] ** (2)
            + 3 * data_node_pos[:, 0] * data_node_pos[:, 1]
            - 2 * data_node_pos[:, 1] ** (2)
            + 20.0
        )
        a = np.unique(data_Z.cpu().numpy(), return_counts=True)
        b = np.unique(data_cell_Z.cpu().numpy(), return_counts=True)
        if a[0].shape[0] != data_Z.shape[0] or b[0].shape[0] != data_cell_Z.shape[0]:
            print("data renum faild{0}".format(index))
            return False
        else:
            sorted_data_Z, new_data_index = torch.sort(data_Z, descending=False)
            sorted_data_cell_Z, new_data_cell_index = torch.sort(
                data_cell_Z, descending=False
            )
            for key in key_list:
                value = dataset[key]
                if key == "cells":
                    new_dataset[key] = torch.index_select(value, 1, new_data_cell_index)
                elif key == "boundary":
                    new_dataset[key] = value
                else:
                    new_dataset[key] = torch.index_select(value, 1, new_data_index)
            cell_renum_dict = {}
            new_cells = np.empty_like(dataset["cells"][0])
            for i in range(new_dataset["mesh_pos"].shape[1]):
                cell_renum_dict[str(new_dataset["mesh_pos"][0][i].numpy())] = i
            for j in range(dataset["cells"].shape[1]):
                cell = dataset["cells"][0][j]
                for node_index in range(cell.shape[0]):
                    new_cells[j][node_index] = cell_renum_dict[
                        str(dataset["mesh_pos"][0].numpy()[cell[node_index]])
                    ]
            new_cells = np.repeat(
                np.expand_dims(new_cells, axis=0), dataset["cells"].shape[0], axis=0
            )
            new_dataset["cells"] = torch.index_select(
                torch.from_numpy(new_cells), 1, new_data_cell_index
            )

            cells_node = new_dataset["cells"][0]
            mesh_pos = new_dataset["mesh_pos"][0]
            new_dataset["centroid"] = (
                (
                    torch.index_select(mesh_pos, 0, cells_node[:, 0])
                    + torch.index_select(mesh_pos, 0, cells_node[:, 1])
                    + torch.index_select(mesh_pos, 0, cells_node[:, 2])
                )
                / 3.0
            ).view(1, -1, 2)
            for key, value in new_dataset.items():
                dataset[key] = value.numpy()
                new_dataset[key] = value.numpy()
            # new_dataset = reorder_boundaryu_to_front(dataset)
            new_dataset["cells"] = new_dataset["cells"][0:1, :, :]
            new_dataset["mesh_pos"] = new_dataset["mesh_pos"][0:1, :, :]
            new_dataset["node_type"] = new_dataset["node_type"][0:1, :, :]
            write_tfrecord_one_with_writer(writer, new_dataset, mode="cylinder_flow")
            print("origin datasets No.{} has been parsed mesh\n".format(index_num))


transformer = T.Compose([T.Distance(norm=False)])

if __name__ == "__main__":
    # choose wether to transform whole datasets into h5 file

    tf.enable_resource_variables()
    tf.enable_eager_execution()
    pickl_path = path["pickl_save_path"]
    tf_datasetPath = path["tf_datasetPath"]
    # tf_datasetPath='/home/litianyu/mycode/repos-py/MeshGraphnets/pytorch/meshgraphnets-main/datasets/airfoil'
    # tf_datasetPath='/root/share/meshgraphnets/datasets/airfoil'
    numofsd = 2
    os.makedirs(path["tf_datasetPath"], exist_ok=True)

    """set current work directory"""
    imgoutputdir = os.path.split(__file__)[0] + "/imgoutput"
    os.makedirs(imgoutputdir, exist_ok=True)
    current_file_dir = os.chdir(imgoutputdir)

    for split in ["valid", "train", "test"]:
        ds = load_dataset(tf_datasetPath, split)
        # d = tf.data.make_one_shot_iterator(ds).zget_next()
        rearrange_frame_sp_1 = []
        rearrange_frame_sp_2 = []
        # parse_reshape(ds)
        raw_data = {}
        tf_saving_mesh_path = (
            path["mesh_save_path"] + "_" + model["name"] + "_" + split + ".tfrecord"
        )
        save_path = (
            path["h5_save_path"] + "_" + model["name"] + "_" + split + "_" + ".h5"
        )
        with h5py.File(save_path, "w") as h5_writer:
            with tf.io.TFRecordWriter(tf_saving_mesh_path) as tf_writer:
                for index, d in enumerate(ds):
                    if "mesh_pos" in d:
                        mesh_pos = d["mesh_pos"].numpy()
                        raw_data["mesh_pos"] = mesh_pos
                    if "node_type" in d:
                        node_type = d["node_type"].numpy()
                        raw_data["node_type"] = node_type
                    if "velocity" in d:
                        velocity = d["velocity"].numpy()
                        raw_data["velocity"] = velocity
                    if "cells" in d:
                        cells = d["cells"].numpy()
                        raw_data["cells"] = cells
                    if "density" in d:
                        density = d["density"].numpy()
                        raw_data["density"] = density
                    if "pressure" in d:
                        pressure = d["pressure"].numpy()
                        raw_data["pressure"] = pressure
                    if path["renum_origin_dataset"]:
                        parse_origin_dataset(
                            raw_data,
                            unorder=False,
                            index_num=index,
                            plot=None,
                            writer=tf_writer,
                        )
                    # if index%889==0 and index>0:
                    dataset = raw_data
                    if True and index > 0:
                        dataset = raw_data
                        # dataset,rtvalue_renum = renum_data(dataset,True,index,"cell")
                        # dataset,rtvalue_renum = renum_data(dataset,False,index,None)
                        # if not rtvalue_renum:
                        #   raise ValueError('InvalidArgumentError')
                        rearrange_frame_1 = {}
                        rearrange_frame_2 = {}
                        if path["plot_order"]:
                            fig = plt.figure(figsize=(4, 3))  # 创建画布
                            mesh_pos = dataset["mesh_pos"][0]
                            display_pos_list = [mesh_pos[0], mesh_pos[1], mesh_pos[2]]
                            ax1 = fig.add_subplot(211)
                            ax2 = fig.add_subplot(212)

                            ax2.cla()
                            ax2.set_aspect("equal")
                            # bb_min = mesh['velocity'].min(axis=(0, 1))
                            # bb_max = mesh['velocity'].max(axis=(0, 1))
                            mesh_pos = dataset["mesh_pos"][0]
                            faces = dataset["cells"][0]
                            triang = mtri.Triangulation(
                                mesh_pos[:, 0], mesh_pos[:, 1], faces
                            )
                            # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
                            ax2.triplot(triang, "ko-", ms=0.5, lw=0.3)
                            # plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)

                            def animate(num):
                                if num < mesh_pos.shape[0]:
                                    display_pos_list.append(mesh_pos[num])
                                if num % 3 == 0 and num > 0:
                                    display_pos = np.array(display_pos_list)
                                    p1 = ax1.scatter(
                                        display_pos[:, 0],
                                        display_pos[:, 1],
                                        c="red",
                                        linewidths=1,
                                    )
                                    ax1.legend(["node_pos"], loc=2, fontsize=10)

                            ani = animation.FuncAnimation(
                                fig,
                                animate,
                                frames=dataset["mesh_pos"][0].shape[0],
                                interval=100,
                            )
                            plt.show(block=True)
                            ani.save("order" + "train.gif", writer="pillow")

                    if path["saving_tec"]:
                        tec_saving_path = (
                            path["tec_save_path"]
                            + model["name"]
                            + "_"
                            + split
                            + "_"
                            + str(index)
                            + ".dat"
                        )
                        write_tecplot_ascii_nodal(
                            dataset,
                            True,
                            "/home/litianyu/mycode/repos-py/FVM/my_FVNN/rollouts/0.pkl",
                            tec_saving_path,
                            is_boundary=False,
                        )

                    if path["stastic"]:
                        stastic_nodeface_type(dataset["node_type"][0])

                    if path["saving_h5"]:
                        rtval = extract_mesh_state(
                            raw_data,
                            h5_writer,
                            index,
                            mode=path["mode"],
                            h5_writer=h5_writer,
                            path=path,
                        )
                        if not rtval:
                            print("parse error")
                            exit()

                    if path["saving_mesh"]:
                        rtval = extract_mesh_state(
                            raw_data,
                            tf_writer,
                            index,
                            mode=path["mode"],
                            h5_writer=None,
                            path=path,
                        )
                        if not rtval:
                            print("parse error")
                            exit()
                    # if(path['mask_features']):
                    #   import plot_tfrecord as pltf
                    #   rt_dataset = parser.mask_features(dataset,'velocity',0.1)
                    # pltf.plot_tfrecord_tmp(rt_dataset)

                    if path["print_tf"]:
                        with tf.Session() as sess:
                            velocity_t = tf.convert_to_tensor(velocity[0])
                            node_type_t = node_type[0][:, 0].tolist()
                            res = tf.one_hot(indices=node_type_t, depth=NodeType.SIZE)
                            node_features = tf.concat([velocity_t, res], axis=-1)
                            print(sess.run(ds))
                            print(sess.run(res))
                            print(sess.run(node_features))
                        for i in range(node_type.shape[0]):
                            print(i)
                            stastic(node_type[i])

                    if path["saving_sp_tf"]:
                        start_time = time.time()
                        rt_traj = seprate_cells(
                            mesh_pos,
                            cells,
                            node_type,
                            density,
                            pressure,
                            velocity,
                            index,
                        )

                        end_time = time.time()

                        time_span = end_time - start_time

                        print(
                            "dataset`s frame index is{0}, done. Cost time:{1}:".format(
                                index, time_span
                            )
                        )

                        if path["saving_pickl"]:
                            pickle_save(pickl_path, rearrange_frame_1)

                        if path["saving_sp_tf_single"]:
                            tf_save_path1 = (
                                path["tfrecord_sp"]
                                + model["name"]
                                + "_"
                                + split
                                + "_sp1"
                                + ".tfrecord"
                            )
                            tf_save_path2 = (
                                path["tfrecord_sp"]
                                + model["name"]
                                + "_"
                                + split
                                + "_sp2"
                                + ".tfrecord"
                            )
                            write_tfrecord_one(tf_save_path1, rt_traj[0])
                            write_tfrecord_one(tf_save_path2, rt_traj[1])
                            if index == 2:
                                break
                        rearrange_frame_sp_1.append(rt_traj[0])
                        rearrange_frame_sp_2.append(rt_traj[1])
                    if path["saving_sp_tf_mp"]:
                        rearrange_frame_sp = [
                            rearrange_frame_sp_1,
                            rearrange_frame_sp_2,
                        ]
                        tf_save_path1 = (
                            path["tfrecord_sp"]
                            + model["name"]
                            + "_"
                            + split
                            + "_sp1"
                            + ".tfrecord"
                        )
                        tf_save_path2 = (
                            path["tfrecord_sp"]
                            + model["name"]
                            + "_"
                            + split
                            + "_sp2"
                            + ".tfrecord"
                        )

                        write_tfrecord_mp(
                            tf_save_path1, tf_save_path2, rearrange_frame_sp
                        )
                        print("splited mesh_pos1 is: ", rearrange_frame_sp_1.shape)
                        print("splited mesh_pos2 is: ", rearrange_frame_sp_2.shape)

        print("datasets {} has been extracted mesh\n".format(split))
