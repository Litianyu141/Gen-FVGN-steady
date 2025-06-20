"""Utility functions for reading the datasets."""

import sys
import os
file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(file_dir)

from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
import json
import random
import torch
import numpy as np
import h5py
import math
from Utils import utilities, get_param
from Utils.utilities import (
    generate_boundary_zone,
    calc_cell_centered_with_node_attr,
    calc_node_centered_with_cell_attr,
)
from Utils.utilities import NodeType
from Extract_mesh.parse_to_h5 import seperate_domain,build_k_hop_edge_index
from torch_geometric.nn import knn_graph,knn,radius,radius_graph,knn_interpolate
from Post_process.to_vtk import write_hybrid_mesh_to_vtu_2D,write_to_vtk,to_pv_cells_nodes_and_cell_types
from torch_geometric import utils as pyg_utils
from Load_mesh.Set_BC import velocity_profile
from FVMmodel.FVdiscretization.FVgrad import compute_normal_matrix

class CFDdatasetBase:
    # Base class for CFDdataset with process_trajectory method
    @staticmethod
    def select_PDE_coef(theta_PDE_list=None):
        (
            mean_U,
            rho,
            mu,
            source,
            aoa,
            dt,
            L,
        ) = random.choice(theta_PDE_list)

        return (
            mean_U, 
            rho, 
            mu, 
            source,
            aoa, 
            dt, 
            L,
        )

    @staticmethod
    def calc_charactisc_length(mesh):
        """prepare data for cal_relonyds_number"""
        # 输入为二维点云坐标，形状为 (N, 2)，其中 N 是点的数量
        # 扩展维度以便计算所有点对之间的欧几里得距离
        mesh_pos = mesh["node|pos"]
        surf = mesh["node|surf_mask"]
        
        if not surf.any():
            return torch.zeros(1) # There`s no surface in the mesh
        
        surf_pos = mesh_pos[surf]
        
        points_expanded1 = surf_pos.unsqueeze(0)  # 形状 (1, N, 2)
        points_expanded2 = surf_pos.unsqueeze(1)  # 形状 (N, 1, 2)
        
        # 计算所有点对之间的距离
        distances = torch.norm(points_expanded1 - points_expanded2, dim=2)  # 形状 (N, N)
        
        # 返回距离的最大值
        max_distance = torch.max(distances)

        return max_distance

    @staticmethod
    def init_env(
        mesh,
        mean_u=None,
    ):
        # init node uvp
        node_pos = mesh["node|pos"]
        uv_node, p_node = velocity_profile(
            inlet_node_pos=node_pos,
            mean_u=mean_u,
            aoa=mesh["aoa"],
            inlet_type=mesh["init_field_type"],
        )
        
        # set uniform initial field value
        uvp_node = torch.cat(
            (
                uv_node, 
                p_node
            ),
            dim=1
        ).to(torch.float32)
        
        # generate BC mask
        node_type = mesh["node|node_type"].long().squeeze()
        Wall_mask = (node_type == NodeType.WALL_BOUNDARY)
        Inlet_mask = (node_type == NodeType.INFLOW)|(node_type==NodeType.IN_WALL)|(node_type==NodeType.PRESS_POINT)
        In_wall_mask = (node_type == NodeType.IN_WALL)
        
        # generate velocity profile
        inlet_uvp_node, _ = velocity_profile(
            inlet_node_pos=node_pos[Inlet_mask],
            mean_u=mean_u,
            aoa=mesh["aoa"],
            inlet_type=mesh["inlet_type"],
        )
        inlet_uvp_node = inlet_uvp_node.to(torch.float32)
        
        # apply velocity profile and boundary condition
        uvp_node[Inlet_mask,0:2] = inlet_uvp_node[:,0:2]
        uvp_node[Wall_mask,0:2] = 0
        uvp_node[In_wall_mask] = uvp_node[In_wall_mask]/2.

        # store target node for dirchlet BC and make dimless if possible
        mesh["target|uvp"] = uvp_node[:,0:2].clone() / mean_u
 
        # # interpolate init field to cell
        # centroid = mesh["cell|centroid"]
        # uvp_cell = knn_interpolate(
        #     x = uvp_node, pos_x = node_pos, pos_y = centroid, k=8
        # ).to(torch.float32)
        
        return mesh, uvp_node

    @staticmethod
    def set_theta_PDE(mesh, params, mean_velocity, rho, mu, source, aoa, dt, dL):
        """
        设置用于 PDE 求解的参数 theta_PDE, 并将其添加到 mesh 字典中。

        参数：
        - mesh: 包含网格信息的字典。
        - params: 参数配置对象。
        - mean_velocity: 入口平均速度的标量值。
        - rho: 流体密度。
        - mu: 流体黏度。
        - source: 源项大小。
        - aoa: 攻角(angle of attack)以度为单位。
        - dt: 时间步长。
        - dL: 特征长度。

        返回：
        - mesh: 更新后的网格信息字典，包含了计算所得的 PDE 参数。
        - U_in: 乘上攻角之后的入口速度的二维张量。
        """
        U_in = mean_velocity*torch.tensor(
            [math.cos(math.radians(aoa)), math.sin(math.radians(aoa))]
        )
        
        mesh_pos = mesh["node|pos"][0]
        
        theta_PDE = mesh["theta_PDE_bak"]
        
        unsteady_coefficent = theta_PDE["unsteady"]

        continuity_eq_coefficent = theta_PDE["continuity"]

        convection_coefficent = theta_PDE["convection"]

        grad_p_coefficent = theta_PDE["grad_p"] / rho

        diffusion_coefficent = (
            (mu / mean_velocity) if 0 == convection_coefficent else # convection_coefficent=0 means poisson equation
            (mu / (rho * mean_velocity)) # Navier-Stokes equation
        )

        source_term = source / mean_velocity # if params.dimless else source

        dt_cell = dt * mean_velocity # if params.dimless else dt
        
        theta_PDE = torch.tensor(
            [
                unsteady_coefficent,
                continuity_eq_coefficent,
                convection_coefficent,
                grad_p_coefficent,
                diffusion_coefficent,
                source_term,
                U_in[0].item(),
                U_in[1].item(),
                mesh["Re"],
            ],
            device=mesh_pos.device,
            dtype=torch.float32,
        ).view(1,-1)
        mesh["theta_PDE"] = theta_PDE
        
        mesh["dt_graph"] = torch.tensor(
            [
                dt_cell
            ],
            device=mesh_pos.device,
            dtype=torch.float32,
        ).view(1,-1)
        
        mesh["sigma"] = torch.from_numpy(np.array(mesh["sigma"])).view(1,-1)
        
        mesh["uvp_dim"] = torch.tensor(
            [[[mean_velocity, mean_velocity, (mean_velocity**2)]]],
            device=mesh_pos.device,
            dtype=torch.float32,
        ).view(1,-1)

        return mesh, U_in

    @staticmethod
    def makedimless(
        mesh, params, case_name=None, theta_PDE_list=None
    ):
        (
            mean_u,
            rho,
            mu,
            source,
            aoa,
            dt,
            L,
        ) = CFDdatasetBase.select_PDE_coef(theta_PDE_list)
        
        mesh["mean_u"] = torch.tensor(mean_u, dtype=torch.float32)
        mesh["rho"] = torch.tensor(rho, dtype=torch.float32)
        mesh["mu"] = torch.tensor(mu, dtype=torch.float32)
        mesh["source"] = torch.tensor(source, dtype=torch.float32)
        mesh["aoa"] = torch.tensor(aoa, dtype=torch.float32)
        mesh["dt"] = torch.tensor(dt, dtype=torch.float32)
        mesh["L"] = torch.tensor(L, dtype=torch.float32)
        mesh["Re"] = torch.tensor(rho * mean_u * L, dtype=torch.float32) / \
            mu if mu!=0 else torch.tensor(0, dtype=torch.float32)
        
        (
            mesh,
            U_inlet,
        ) = CFDdatasetBase.set_theta_PDE(
            mesh, params, mean_u, rho, mu, source, aoa, dt, L
        )

        return mesh, mean_u, U_inlet

    @staticmethod
    def calc_WLSQ_A_B_normal_matrix(mesh, order):

        if not "A_node_to_node" in mesh.keys():
            
            """>>> compute WLSQ node to node left A matrix >>>"""
            mesh_pos = mesh["node|pos"]
            face_node_x = mesh["face_node_x"].long()
            support_edge = mesh["support_edge"].long()

            """ >>> compute WLSQ node to node left A matrix >>> """
            (A_node_to_node, two_way_B_node_to_node, single_way_B_node_to_node) = compute_normal_matrix(
                order=order,
                mesh_pos=mesh_pos,
                edge_index=face_node_x, # 默认应该是仅包含1阶邻居点+构成共点的单元的所有点
                extra_edge_index=support_edge, # 额外的模板。例如内部点指向边界点
                periodic_idx=None,
            )
            
            mesh["A_node_to_node"] = A_node_to_node.to(torch.float32)
            mesh["single_B_node_to_node"] = (
                torch.chunk(two_way_B_node_to_node, 2, dim=0)[0]
            ).to(torch.float32)
            mesh["extra_B_node_to_node"] = single_way_B_node_to_node.to(torch.float32)
            """ <<< compute WLSQ node to node right B matrix<<< """

        return mesh

    @staticmethod
    def calc_WLSQ_A_B_matrix(mesh):

        if not "R_inv_Q_t" in mesh.keys():
            """>>> compute WLSQ node to node R_inv_Q_t matrix >>>"""
            mesh_pos = mesh["mesh_pos"][0]
            node_neigbors = mesh["node_neigbors"][0].to(torch.long)
            max_neighbors = mesh["max_neighbors"]

            mask_fil = (node_neigbors != -1).unsqueeze(2)
            neigbor_pos = mesh_pos[node_neigbors] * mask_fil
            moments_left = neigbor_pos - mesh_pos.unsqueeze(1)
            weight_unfiltered = 1.0 / torch.norm(moments_left, dim=2, keepdim=True)
            weight = (
                torch.where(torch.isfinite(weight_unfiltered), weight_unfiltered, 0.0)
                * mask_fil
            )
            A = weight * moments_left

            Q, R = torch.linalg.qr(A)

            # # 创建同维度的单位矩阵
            # I = torch.eye(R.shape[1]).unsqueeze(0).repeat(R.shape[0],1,1)

            # # 使用torch.linalg.solve_triangular一次性求解整个逆矩阵
            # R_inv = torch.linalg.solve_triangular(R, I, upper=True)
            R_inv = torch.linalg.inv(R)

            R_inv_Q_t = torch.matmul(R_inv, Q.transpose(1, 2)) * (
                weight.transpose(1, 2)
            )

            if A.shape[1] < max_neighbors:
                R_inv_Q_t = torch.cat(
                    (
                        R_inv_Q_t,
                        torch.zeros(
                            (
                                R_inv_Q_t.shape[0],
                                R_inv_Q_t.shape[1],
                                max_neighbors - R_inv_Q_t.shape[2],
                            ),
                            device=R_inv_Q_t.device,
                        ),
                    ),
                    dim=2,
                )
                node_neigbors = torch.cat(
                    (
                        node_neigbors,
                        torch.full(
                            (node_neigbors.shape[0], max_neighbors - A.shape[1]),
                            -1,
                            device=node_neigbors.device,
                        ),
                    ),
                    dim=1,
                )
                mask_fil = torch.cat(
                    (
                        mask_fil,
                        torch.full(
                            (
                                mask_fil.shape[0],
                                max_neighbors - mask_fil.shape[1],
                                mask_fil.shape[2],
                            ),
                            False,
                            device=mask_fil.device,
                        ),
                    ),
                    dim=1,
                )
                mask_fil = torch.where(mask_fil, 1.0, 0.0)

            mesh["R_inv_Q_t"] = R_inv_Q_t.unsqueeze(0).to(torch.float32)
            mesh["node_neigbors"] = node_neigbors.unsqueeze(0)
            mesh["mask_node_neigbors_fil"] = mask_fil.unsqueeze(0)

        return mesh

    @staticmethod
    def cal_node_centered_element_area(mesh):
        cells_area = mesh["cells_area"][0]
        cells_node = mesh["cells_node"][0].to(torch.long)
        cells_index = mesh["cells_index"][0].to(torch.long)
        
        node_area = calc_node_centered_with_cell_attr(cell_attr=cells_area, 
                                          cells_node=cells_node, 
                                          cells_index=cells_index, 
                                          reduce="mean", 
                                          map=True)
        
        mesh["node_area"] = node_area.unsqueeze(0)
        
        return mesh
    
    @staticmethod
    def normalize_coords(coords):
        """
        将二维坐标张量的 x 和 y 分量归一化到 [-1, 1] 范围内。

        参数:
        coords (torch.Tensor): 维度为 (N, 2) 的张量，表示 N 个节点的二维坐标。

        返回:
        torch.Tensor: 归一化后的坐标张量，形状为 (N, 2)。
        """
        
        de_mean = coords - coords.mean(dim=0,keepdim=True)
        
        # 获取每个维度的最小值和最大值
        min_vals, _ = de_mean.min(dim=0)
        max_vals, _ = de_mean.max(dim=0)
        range = torch.maximum(max_vals.abs(),min_vals.abs())
        
        # 归一化到 [0, 1] 范围内
        normalized = de_mean / range.unsqueeze(0)

        return normalized

    @staticmethod
    def To_Cartesian(mesh, resultion:tuple):

        if "grids" not in mesh.keys():
            
            L,K = resultion
            mesh_pos = mesh["mesh_pos"][0].to(torch.float32)
            
            xmax = torch.max(mesh_pos[:,0])
            xmin = torch.min(mesh_pos[:,0])
            ymin = torch.min(mesh_pos[:,1])
            ymax = torch.max(mesh_pos[:,1])

            grid_y, grid_x= torch.meshgrid(torch.linspace(ymin, ymax, L), 
                                           torch.linspace(xmin, xmax, K),
                                           indexing='ij')
            
            grid_points = torch.stack((grid_x, grid_y), dim=-1)

            mesh["grids"] = grid_points.unsqueeze(0).to(torch.float32)
            
            mesh["query"] = CFDdatasetBase.normalize_coords(mesh_pos.clone()).unsqueeze(0)
            
        return mesh
    
    @staticmethod
    def construct_stencil(
        mesh, 
        k_hop=2,
        BC_interal_neigbors=4,
        order=None,
    ):
        if not "support_edge" in mesh.keys():
            mesh_pos = mesh["node|pos"]
            face_node = mesh["face|face_node"].long().squeeze()
            node_type = mesh["node|node_type"].long().squeeze()
            face_node_x = mesh["face_node_x"].long().to(torch.long)

            BC_mask = ((node_type==NodeType.WALL_BOUNDARY)|
                       (node_type==NodeType.INFLOW)|
                       (node_type==NodeType.OUTFLOW)|
                       (node_type==NodeType.PRESS_POINT)|
                       (node_type==NodeType.IN_WALL)).squeeze()

            
            ''' including other boundary points '''
            # node_index = torch.arange(mesh_pos.shape[0])
            # BC_edge_index = knn(x=mesh_pos, y=mesh_pos[BC_mask], k=BC_interal_neigbors)
            # filter_self_loop = (BC_edge_index[1]==node_index[BC_mask][BC_edge_index[0]]).squeeze()
            # BC_edge_index = torch.stack((BC_edge_index[1], node_index[BC_mask][BC_edge_index[0]]), dim=0)
            # BC_edge_index = BC_edge_index[:,~filter_self_loop]
            # BC_edge_index = BC_edge_index[:,~(BC_edge_index[0]==BC_edge_index[1])]
            ''' including other boundary points '''
            
            ''' exclude other boundary points '''
            # node_index = torch.arange(mesh_pos.shape[0])
            # BC_edge_index = knn(x=mesh_pos[~BC_mask], y=mesh_pos[BC_mask], k=BC_interal_neigbors)
            # BC_edge_index = torch.stack((node_index[~BC_mask][BC_edge_index[1]], node_index[BC_mask][BC_edge_index[0]]), dim=0)
            # BC_edge_index = BC_edge_index[:,~(BC_edge_index[0]==BC_edge_index[1])]
            ''' exclude other boundary points '''

            ''' only internal node to boundary node edge '''
            # extra_edge_index = []
            # for k in range(1, k_hop+1):
            #     extra_edge_index.append(build_k_hop_edge_index(torch.cat((face_node,face_node.flip(0)),dim=1),k=k))
            # extra_edge_index = torch.cat((extra_edge_index), dim=1) # 此时刚出k-hop是包含所有内部边的且dual edge的
            # extra_edge_index = extra_edge_index[:,
            #                                     (BC_mask[extra_edge_index[0]]&(~BC_mask[extra_edge_index[1]]))|\
            #                                     (BC_mask[extra_edge_index[1]]&(~BC_mask[extra_edge_index[0]]))|\
            #                                     (BC_mask[extra_edge_index[0]]&(BC_mask[extra_edge_index[1]]))
            # ] # 先选出仅为内部指向边界或者边界指向内部的边
            # extra_edge_index = torch.unique(extra_edge_index[:,~(extra_edge_index[0]==extra_edge_index[1])].sort(0)[0],dim=1) # 排除自环然后收缩为单向边
            
            # extra_edge_index_mirror = extra_edge_index.flip(0) 
            # internal_to_boundary = torch.where(
            #     BC_mask[extra_edge_index[0]][None,].repeat(2,1),extra_edge_index_mirror,extra_edge_index
            # ) # 然后调整指向，保证只有内部点（dim=0的0）指向边界点（dim=0的1）
            ''' only internal node to boundary node edge '''
            
            ''' 全局调用k-hop edge '''
            extra_edge_index = []
            for k in range(1, k_hop+1):
                extra_edge_index.append(build_k_hop_edge_index(torch.cat((face_node,face_node.flip(0)),dim=1),k=k))
            extra_edge_index = torch.cat((extra_edge_index), dim=1) # 此时刚出k-hop是包含所有内部边的且dual edge的
            # extra_edge_index = extra_edge_index[:,~(BC_mask[extra_edge_index[0]]&(BC_mask[extra_edge_index[1]]))] # 先排除边界指向边界的边ss
            extra_edge_index_unique = torch.unique(
                extra_edge_index[:,~(extra_edge_index[0]==extra_edge_index[1])].sort(0)[0],
                dim=1
            ) # 排除自环然后收缩为单向边

            mesh["face_node_x"] = torch.cat((face_node_x, extra_edge_index_unique), dim=1)
            mesh["support_edge"] = torch.tensor([[0,1],[1,0]]).to(mesh_pos.device) # 2025.5.30:临时解决方案，现在放弃使用extra_edge模板了，直接全局k-hop放入face_node_x中
            ''' 全局调用k-hop edge '''
            
            ''' 检查模板并绘制'度'的分布 '''
            # # start plotting
            # support_edge = torch.cat((
            #     face_node_x, 
            #     internal_to_boundary
            # ), dim=1)
            # in_degree = pyg_utils.degree(support_edge[1], num_nodes=mesh_pos.shape[0])
            # out_degree = pyg_utils.degree(support_edge[0], num_nodes=mesh_pos.shape[0])
            # node_degree = in_degree + out_degree
            # print("Degree max, mean ,min:", node_degree.max(), node_degree.mean(), node_degree.min())
            
            # # write to file
            # mesh["node_degree"] = node_degree
            # pv_cells_node,pv_cells_type = to_pv_cells_nodes_and_cell_types(
            #     cells_node=mesh["cells_node"], cells_face=mesh["cells_face"], cells_index=mesh["cells_index"]
            # )
            
            # write_hybrid_mesh_to_vtu_2D(
            #     mesh_pos=mesh_pos.cpu().numpy(), 
            #     data={
            #         f"node|in_degree":in_degree.cpu().numpy(),
            #         f"node|out_degree":out_degree.cpu().numpy(),
            #         f"node|node_degree":node_degree.cpu().numpy(),
            #     }, 
            #     cells_node=pv_cells_node.cpu().numpy(), 
            #     cells_type=pv_cells_type.cpu().numpy(),
            #     filename=f"Logger/Grad_test/degree_vis.vtu",
            # )
            ''' 检查模板并绘制度分布 '''
            
            # mesh["support_edge"] = internal_to_boundary
            
        return mesh
    
    @staticmethod
    def transform_mesh(
        mesh, 
        params=None
    ):
        
        theta_PDE_list = mesh["theta_PDE_list"]
        case_name = mesh["case_name"]
        
        mesh, mean_u, U_inlet = CFDdatasetBase.makedimless(
            mesh,
            theta_PDE_list=theta_PDE_list,
            case_name=case_name,
            params=params,
        )
        
        mesh = CFDdatasetBase.construct_stencil(
            mesh, 
            k_hop=mesh["stencil|khops"], 
            BC_interal_neigbors=mesh["stencil|BC_extra_points"],
            order=params.order,
        )

        mesh = CFDdatasetBase.calc_WLSQ_A_B_normal_matrix(mesh,params.order)
        
        mesh, init_uvp_node = CFDdatasetBase.init_env(
            mesh,        
            mean_u=mean_u,
        )

        # start to generate boundary zone
        surf = mesh["node|surf_mask"].squeeze()
        if surf.any():
            boundary_zone = generate_boundary_zone(
                dataset=mesh,
                surf_mask=surf,
                rho=mesh["rho"].item(),
                mu=mesh["mu"].item(),
                dt=mesh["dt"].item(),
            )
            mesh["boundary_zone"] = boundary_zone

        return mesh, init_uvp_node

class H5CFDdataset(Dataset):
    def __init__(self, params, file_list):
        super().__init__()

        self.file_list = file_list
        self.params = params
        
    def __getitem__(self, index):
        path = self.file_list[index]
        file_dir = os.path.dirname(path)
        case_name = os.path.basename(file_dir)
        h5_file = h5py.File(path, "r")

        try:
            BC_file = json.load(open(f"{file_dir}/BC.json", "r"))
        except:
            raise ValueError(f"BC.json is not found in the {path}")
        
        key_list = list(h5_file.keys())
        mesh_handle = h5_file[key_list[0]]
        mesh = {"case_name":case_name} # set mesh name
        
        # convert to tensors
        for key in mesh_handle.keys():
            mesh[key] = torch.from_numpy(mesh_handle[key][()])

        # import all BC.json item into mesh dict
        for key, value in BC_file.items():
            mesh[key] = value
        mesh["theta_PDE_bak"] = mesh["theta_PDE"] # 后续生成单独case参数时候theta_PDE会被覆盖，所以备份一下
        
        # generate all valid theta_PDE combinations
        theta_PDE = mesh["theta_PDE_bak"]
        theta_PDE_list = (
                    get_param.generate_combinations(
                        U_range=theta_PDE["inlet"],
                        rho_range=theta_PDE["rho"],
                        mu_range=theta_PDE["mu"],
                        source_range=theta_PDE["source"],
                        aoa_range=theta_PDE["aoa"],
                        dt=theta_PDE["dt"],
                        L=theta_PDE["L"],
                        Re_max=theta_PDE["Re_max"],
                        Re_min=theta_PDE["Re_min"],
                    )
                )
        mesh["theta_PDE_list"] = theta_PDE_list
        
        # start to calculate other attributes like stencil, WLSQ matrix, etc.
        mesh_transformed, init_uvp_node = CFDdatasetBase.transform_mesh(
            mesh, 
            self.params
        )

        # return to CPU!
        return mesh_transformed, init_uvp_node 

    def __len__(self):
        return len(self.file_list)

