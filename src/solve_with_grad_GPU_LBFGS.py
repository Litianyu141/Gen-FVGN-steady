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
        
        self.plot_order=0
        
        self.graph_node.x[:,0:3] = torch.cat((self.graph_node.pos,self.graph_node.pos[:,0:1]),dim=-1)
        self.graph_cell.x[:,0:3] = torch.cat((self.graph_cell.pos,self.graph_cell.pos[:,0:1]),dim=-1)
        
        self.uvp_node_new = 0
        self.uvp_cell_new = 0
        
    def train(self,n_epoch):
        
        t1_start = time.time()

        self.model.train() 
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            max_iter=self.params.n_epochs, 
            history_size=1000,
            tolerance_grad=-1,
            tolerance_change=-1,
            line_search_fn='strong_wolfe'
        )

        self.niter = 0
        
        def closure():
            inner_start = time.time()
                
            self.optimizer.zero_grad()
            
            ''' >>> please check src/Load_mesh/Graph_loader.py/->update_x_attr >>> '''
            self.graph_node.norm_uvp=params.norm_uvp
            self.graph_node.norm_global=params.norm_global
            ''' <<< please check src/Load_mesh/Graph_loader.py/->update_x_attr <<< '''
            
            (
                loss_cont,
                loss_mom_x,
                loss_mom_y,
                loss_press,
                uvp_node_new,
                uvp_cell_new,
            ) = fluid_model(
                graph_node=self.graph_node,
                graph_node_x=self.graph_node_x,
                graph_edge=self.graph_edge,
                graph_cell=self.graph_cell,
                graph_Index=self.graph_Index,
                is_training=True,
            )

            ''' back up exclude LSFD'''
            loss_batch = (
                params.loss_press * loss_press
                + params.loss_cont * loss_cont
                + params.loss_mom * loss_mom_x
                + params.loss_mom * loss_mom_y
            )

            loss = torch.mean(torch.log(loss_batch))
            ''' back up exclude LSFD'''
            
            # loss = loss_cont_nwise
            
            # compute gradients
            loss.backward()
            
            self.uvp_cell_new = uvp_cell_new.detach()
            self.uvp_node_new = uvp_node_new.detach()
            
            if self.niter % 1000 == 0:
                self.graph_node.x[:,0:3] = self.uvp_node_new.detach()
                self.graph_cell.x[:,0:3] = self.uvp_cell_new.detach()

                # plot the result            
                graph_list = Batch.to_data_list(self.graph_cell)
                plot_graph = graph_list[self.plot_order].cpu()
                self.datasets.export_to_tecplot(
                    self.datasets.meta_pool[plot_graph.graph_index], 
                    uvp = plot_graph.x[:,0:3],
                    datalocation="cell"
                )
                self.plot_order += 1
                self.plot_order = self.plot_order % params.batch_size
                
                self.graph_node.x[:,0:3] = torch.cat((self.graph_node.pos,self.graph_node.pos[:,0:1]),dim=-1)
                self.graph_cell.x[:,0:3] = torch.cat((self.graph_cell.pos,self.graph_cell.pos[:,0:1]),dim=-1)
                
            self.niter+=1
            print(f"Inner iter {self.niter} Iteration Loss: {loss.item()} completed in {time.time() - inner_start:.2f} seconds")
            
            return loss

        self.optimizer.step(closure)

        # 输出此epoch的处理时间
        print(f"Epoch {self.niter} completed in {time.time() - start:.2f} seconds")

        # save state after every 2 epoch
        _ =self.logger.save_state(
            model=self.model,
            optimizer=None,
            scheduler=None,
            index=str(self.niter % 3),
        )
                
        # plot all the results
        self.graph_node.x[:,0:3] = self.uvp_node_new.detach()
        self.graph_cell.x[:,0:3] = self.uvp_cell_new.detach()
        graph_list = Batch.to_data_list(self.graph_cell)
        for plot_graph in graph_list:
            plot_graph = plot_graph.cpu()
            self.datasets.export_to_tecplot(
                self.datasets.meta_pool[plot_graph.graph_index], 
                uvp = plot_graph.x[:,0:3],
                datalocation="cell"
            )    
             
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

    params.dataset_dir = "datasets/lid_driven/lid_driven_cavity_161x161 copy"
    params.integrator = "implicit"
    
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

    # create dataset objetc
    datasets, loader, sampler = datasets_factory.create_datasets(
        batch_size=params.batch_size, num_workers=0, pin_memory=False
    )

    end = time.time()
    print("Training traj has been loaded time consuming:{0}".format(end - start))

    # initialize fluid model
    model = Simulator(params)

    fluid_model = model.to(device)
    fluid_model.train()


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


    trainer = Trainer(
        params = params, 
        logger = logger,
        model = fluid_model, 
        loader = loader, 
        datasets = datasets, 
        device = device)

    trainer.train(1000)
    
    print("Training completed")