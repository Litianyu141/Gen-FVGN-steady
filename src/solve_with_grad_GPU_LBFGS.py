import sys
import os

cur_path = os.path.split(__file__)[0]
sys.path.append(cur_path)

import torch
from torch.optim import Adam
import numpy as np
from FVMmodel.importer import NNmodel as Simulator

# import os
from Load_mesh import Graph_loader
from Utils import get_param, scheduler
import time
from Utils.get_param import get_hyperparam
from Utils.Logger import Logger
from torch_geometric.nn import global_add_pool,global_mean_pool
from torch_geometric.data.batch import Batch
from math import ceil
import random
import datetime

        
class Trainer():
    def __init__(
        self, 
        params, 
        logger,
        model, 
        loader, 
        datasets, 
        device
    ):
        self.params = params
        self.logger = logger
        self.device = device
        self.model = model
        self.loader = loader
        self.datasets = datasets
        
        (graph_node,graph_node_x,graph_edge,graph_cell,graph_Index) = next(iter(loader))
        
        (graph_node,graph_node_x,graph_edge,graph_cell,graph_Index) = datasets.datapreprocessing(
            graph_node=graph_node.to(self.device),
            graph_node_x=graph_node_x.to(self.device),
            graph_edge=graph_edge.to(self.device),
            graph_cell=graph_cell.to(self.device),
            graph_Index=graph_Index.to(self.device),
        )
        
        self.graph_node = graph_node
        self.graph_node_x = graph_node_x
        self.graph_edge = graph_edge
        self.graph_cell = graph_cell
        self.graph_Index = graph_Index
        
        self.graph_node.backup_x = self.graph_node.x[:,self.params.node_phi_size:].clone()
        
        # 初始化为tensor，用于自回归
        self.uvp_node = self.graph_node.x[:,0:self.params.node_phi_size].clone()
        self.uvp_cell = self.graph_cell.x[:,0:self.params.node_phi_size].clone()
        
        self.uvp_cell_new = torch.zeros_like(self.uvp_cell)
        self.uvp_node_new = torch.zeros_like(self.uvp_node)

    def train(self, n_iterations):
        
        t1_start = time.time()

        self.model.train() 
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            max_iter=1000,  # 减少每次step的最大内部迭代次数，避免过度优化
            history_size=100,  # 减少历史信息存储
            tolerance_grad=1e-6,  # 放宽梯度容忍度
            tolerance_change=1e-8,  # 放宽变化容忍度
            line_search_fn='strong_wolfe'
        )

        self.niter = 0
        
        # LBFGS主循环 - 自回归式训练
        for epoch in range(n_iterations):
            epoch_start = time.time()
            
            # 重置内部迭代计数器
            inner_iter_count = 0
            
            def closure():
                nonlocal inner_iter_count
                inner_start = time.time()
                    
                self.optimizer.zero_grad()
                
                self.graph_node.x = torch.cat((self.uvp_node.detach(),self.graph_node.backup_x), dim=-1)
                self.graph_cell.x[:,0:self.params.node_phi_size] = self.uvp_cell.detach()
                self.graph_node.norm_uvp = self.params.norm_uvp
                self.graph_node.norm_global = self.params.norm_global
                
                (
                    loss_cont,
                    loss_mom_x,
                    loss_mom_y,
                    loss_press,
                    uvp_node_new,
                    uvp_cell_new,
                ) = self.model(
                    graph_node=self.graph_node,
                    graph_node_x=self.graph_node_x,
                    graph_edge=self.graph_edge,
                    graph_cell=self.graph_cell,
                    graph_Index=self.graph_Index,
                    is_training=True,
                )

                ''' back up exclude LSFD'''
                loss_batch = (
                    self.params.loss_press * loss_press
                    + self.params.loss_cont * loss_cont
                    + self.params.loss_mom * loss_mom_x
                    + self.params.loss_mom * loss_mom_y
                )
                loss_batch_clamped = torch.clamp(loss_batch, min=1e-10, max=1e10)
                loss = torch.mean(torch.log(loss_batch_clamped))
                
                ''' back up exclude LSFD'''
                
                # compute gradients
                loss.backward()
                
                # 保存当前的输出用于下一轮迭代
                self.uvp_cell_new = uvp_cell_new.detach()
                self.uvp_node_new = uvp_node_new.detach()
                    
                inner_iter_count += 1
                self.niter += 1
                print(f"Epoch {epoch}, Inner iter {inner_iter_count}, Total iter {self.niter}, Loss: {loss.item():.6e}, Time: {time.time() - inner_start:.2f}s")
                
                return loss
            
            # 执行LBFGS步骤
            try:
                self.optimizer.step(closure)
            except Exception as e:
                print(f"Error in LBFGS step at epoch {epoch}: {e}")

            # 可视化
            if epoch % 10 == 0 or epoch == n_iterations - 1:  # 每10次迭代可视化一次

                graph_list = Batch.to_data_list(self.graph_cell)
                if len(graph_list) > 0:
                    plot_graph = graph_list[0].cpu()  # 取第一个case
                    
                    # 使用最新的求解结果进行可视化
                    plot_graph.x[:,0:3] = self.uvp_cell_new.cpu()
                    
                    # 获取mesh信息来构建文件名
                    mesh = self.datasets.meta_pool[plot_graph.graph_index]
                    case_name = mesh["case_name"]
                    dt = mesh["dt"].squeeze().item()
                    source = mesh["source"].squeeze().item()
                    aoa = mesh["aoa"].squeeze().item()
                    
                    try:
                        Re = mesh["Re"].squeeze().item()
                    except:
                        Re = 0
                        print("Warning: No Re number in the mesh, set to 0")
                    
                    # 创建分组文件夹（每50次迭代一个文件夹）
                    save_dir_num = epoch // 50
                    saving_dir = f"{self.logger.saving_path}/traing_results/LBFGS_iter_{save_dir_num*50}-{(save_dir_num+1)*50}"
                    os.makedirs(saving_dir, exist_ok=True)
                    
                    # 构建文件名，参照Graph_loader.py的格式
                    file_name = f"{saving_dir}/iter_{epoch:06d}_{case_name}_Re={Re:.2f}_dt={dt:.3f}_source={source:.2f}_aoa={aoa:.2f}"
                    
                    # 使用datasets的export_to_tecplot方法进行可视化
                    self.datasets.export_to_tecplot(
                        mesh, 
                        uvp=plot_graph.x[:,0:3],
                        datalocation="cell",
                        file_name=file_name
                    )
                    
                    print(f"Visualization saved: {file_name}")

            # 输出此epoch的处理时间
            print(f"Epoch {epoch} completed in {time.time() - epoch_start:.2f} seconds")

            # 保存模型状态
            if epoch % 10 == 0 or epoch == n_iterations - 1:
                _ = self.logger.save_state(
                    model=self.model,
                    optimizer=None,
                    scheduler=None,
                    index=str(epoch % 3),
                )
                print(f"Model state saved at epoch {epoch}")
             
        print(f"Training completed in {time.time() - t1_start:.2f} seconds")
        
if __name__=="__main__":
    
    # configurate parameters
    params = get_param.params()
    seed = int(datetime.datetime.now().timestamp())
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_per_process_memory_fraction(0.8, params.on_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' >>> 单独设置参数 >>> '''
    params.dataset_dir = "datasets/lid_driven_cavity/lid_driven_cavity_101x101"
    params.integrator = "imex"
    params.norm_global = False # 默认为False，表示不对全局条件进行归一化
    params.batch_size = 1 # LBFGS专用：设置批次大小为1（单case求解）
    ''' <<< 单独设置参数 <<< '''
    
    # initialize Logger and load model / optimizer if according parameters were given
    logger = Logger(
        get_hyperparam(params),
        use_csv=True,
        use_tensorboard=False,
        params=params,
        copy_code=True,
        seed=seed,
    )

    # initialize Training Dataset
    start = time.time()
    datasets_factory = Graph_loader.DatasetFactory(
        params=params,
        dataset_dir=params.dataset_dir,
        state_save_dir=logger.saving_path,
        device=device,
    )

    # refresh dataset size
    params.dataset_size = datasets_factory.dataset_size

    # create dataset objetc - 使用batch_size=1确保每次只处理一个case
    datasets, loader, sampler = datasets_factory.create_datasets(
        batch_size=1, num_workers=0, pin_memory=False
    )

    end = time.time()
    print("Training traj has been loaded time consuming:{0}".format(end - start))

    # initialize fluid model
    model = Simulator(params)

    fluid_model = model.to(device)
    fluid_model.train()

    """ >>> load state from old date >>> """
    if (
        params.load_date_time is not None
        or params.load_index is not None
    ):
        logger.load_logger(datetime=params.load_date_time)
        params.load_date_time, params.load_index = logger.load_state(
            model=fluid_model,
            optimizer=None,
            scheduler=None,
            datetime=params.load_date_time,
            index=params.load_index,
            device=device,
        )
        params.load_index = int(params.load_index)
        print(f"loaded: {params.load_date_time}, {params.load_index}")
    params.load_index = 0 if params.load_index is None else params.load_index
    """ <<< load state from old date <<< """

    trainer = Trainer(
        params = params, 
        logger = logger,
        model = fluid_model, 
        loader = loader, 
        datasets = datasets, 
        device = device)

    # 使用LBFGS求解器进行迭代 - 先用较少的迭代次数测试
    n_iterations = 10  # 减少迭代次数进行测试
    trainer.train(n_iterations)
    
    print("Training completed")