import argparse
import json
import os
import random
import itertools
import numpy as np

def str2bool(v):
    """
    'boolean type variable' for add_argument
    """
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

def params(load=None):
    if load is not None:
        parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')
        params = vars(parser.parse_args([]))
        with open(load+'/commandline_args.json', 'rt') as f:
            params.update(json.load(f))
        for k, v in params.items():
            parser.add_argument('--' + k, default=v)
        args = parser.parse_args([])
        return  args
    else:
        """
        return parameters for training / testing / plotting of models
        :return: parameter-Namespace
        """
        parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')

        # Training parameters
        parser.add_argument('--net', default="TransFVGN_v2", type=str, help='network to train (default: FVGN)', choices=["FVGN","TransFVGN_v1", "TransFVGN_v2"])
        parser.add_argument('--n_epochs', default=210000, type=int, help='number of epochs (after each epoch, the model gets saved)')
        parser.add_argument('--batch_size', default=8, type=int, help='batch size (default: 100)')
        parser.add_argument('--average_sequence_length', default=500, type=int, help='average sequence length in dataset=dataset_size/batch_size(default: 5000)')
        parser.add_argument('--dataset_size', default=100, type=int, help='size of dataset (default: 1000)')
        parser.add_argument('--lr', default=5e-5, type=float, help='learning rate of optimizer (default: 0.0001)')
        parser.add_argument('--lr_scheduler', default="fixlr", type=str, help='choose learing rate scheduler (default: coslr)',choices=['coslr','fix'])
        parser.add_argument('--on_gpu', default=1, type=int, help='set training on which gpu')

        # train strategy parameters
        parser.add_argument('--integrator', default="imex", type=str, help='integration scheme (explicit / implicit / imex) (default: imex)',choices=['explicit','implicit','imex'])
        # parser.add_argument('--dimless', default=True, type=str2bool, help='Injetcting noise on face length for prevention of overfiting (default:False)')
        # 现在默认一定用dimless, Now default of dimless is always True
        parser.add_argument('--norm_uvp', default=True, type=str2bool, help='Whether norm input uvp value at graph_node.x (default:False)')
        parser.add_argument('--norm_global', default=True, type=str2bool, help='Whether norm input condition (eq. Re, pde_theta) value at graph_node.x (default:False)')
        parser.add_argument('--ncn_smooth', default=True, type=str2bool, help='inteploting node value to cell and to node (default:False)')
        parser.add_argument('--conserved_form', default=True, type=str2bool, help='Use artifacial wall method, which is setting boudary node to zero which it was flow into the domain (default:False)')
        parser.add_argument('--residual_tolerance', default=1e-7, type=float, help='unsteady time stepping convergence tolerance')
        parser.add_argument('--max_inner_steps', default=20, type=int, help='unsteady time stepping convergence max iteration steps')
        parser.add_argument('--order', default="2nd", type=str, help='order of WLSQ', choices=["1st","2nd","3rd", "4th"])
        
        # loss parameters
        parser.add_argument('--loss_cont', default=6e4, type=float, help='loss factor for continuity equation')
        parser.add_argument('--loss_mom', default=5e4, type=float, help='loss factor for uv diffusion flux on face')
        parser.add_argument('--loss_press', default=1, type=float, help='whether use p=p_spec,is set 0. it means use p=p_spec (default:0)')

        # Load parameters
        parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
        parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
        parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: True)')
        
        #model parameters
        parser.add_argument('--hidden_size', default=128, type=int, help='hidden size of network (default: 20)')
        parser.add_argument('--message_passing_num', default=3, type=int, help='message passing layer number (default:15)')
        
        parser.add_argument('--node_phi_size', default=3, type=int, help='node encoder node_input_size (default: 2)')
        parser.add_argument('--node_input_size', default=12, type=int, help='node encoder node_input_size (default: 2)')
        parser.add_argument('--node_one_hot', default=5, type=int, help='edge one hot dimention (default: 9)')
        parser.add_argument('--node_output_size', default=3, type=int, help='edge decoder edge_output_size uvp on edge center(default: 8)')

        #dataset params
        parser.add_argument('--dataset_dir', default='datasets/balanced_datasets', type=str, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')	

        params = parser.parse_args([])

        return params

def get_hyperparam(params):
    return f"net {params.net}; hs {params.hidden_size};"

def generate_list(min_val, step, max_val):
    if min_val == step == max_val:
        return [max_val]
    else:
        # 使用linspace可以确保开始和结束的值都包括在内
        # 并根据步长计算必要的点数
        num_points = int((max_val - min_val) / step) + 1
        return list(np.linspace(min_val, max_val, num_points))
    
def generate_combinations(
    U_range=None, rho_range=None, mu_range=None, Re_max=None, Re_min=None, source_range=None, aoa_range=None, dt=None, L=None
):

    U_list = generate_list(*U_range)
    rho_list = generate_list(*rho_range)
    mu_list = generate_list(*mu_range)
    source_list = generate_list(*source_range)
    aoa_list = generate_list(*aoa_range)
    
    combinations = list(itertools.product(U_list, rho_list, mu_list, source_list, aoa_list))

    valid_combinations = []
    valid_Re_values = []
    for U, rho, mu, source, aoa_list in combinations:
        if rho==0.:
            rho=1.
        Re = (U*rho*L) / mu
        if (Re <= Re_max and Re>=Re_min):
            if dt == "1/Re":
                dt = 1/Re
            elif (type(dt) == float) or (type(dt) == int):
                pass
            else:
                raise ValueError("Wrong dt in BC.json, it should be float or int or 1/Re")
            
            valid_combinations.append([U, rho, mu, source, aoa_list, dt, L])
            valid_Re_values.append(Re)

    # 计算最小值，最大值和平均值
    if valid_Re_values:
        min_Re = min(valid_Re_values)
        max_Re = max(valid_Re_values)
        avg_Re = sum(valid_Re_values) / len(valid_Re_values)
        print(f"Total valid Re combination: {len(valid_Re_values)}个")
        print(f"Min valid Re num: {min_Re}")
        print(f"Max valid Re num: {max_Re}")
        print(f"Mean valid Re num: {avg_Re}")
        return valid_combinations
    
    else:
        raise ValueError("Wrong Re number generation checkout all params")

    
if __name__=='__main__':
    
    params_t,git_info = params()
    
    prefix="pf"
    
    if prefix=="cw":
        source_frequency_range = getattr(params_t, f"{prefix}_source_frequency")
        source_strength_range = getattr(params_t, f"{prefix}_source_strength")
        rho_range = getattr(params_t, f"{prefix}_rho")
        dt = getattr(params_t, f"{prefix}_dt")
        
        result = generate_combinations(source_frequency_range=source_frequency_range, source_strength_range=source_strength_range, rho_range=rho_range,dt=dt, eqtype="wave")
    else:
        U_range = getattr(params_t, f"{prefix}_inflow_range")
        rho_range = getattr(params_t, f"{prefix}_rho")
        mu_range = getattr(params_t, f"{prefix}_mu")
        source_range = getattr(params_t, f"{prefix}_source")
        aoa_range = getattr(params_t, f"{prefix}_aoa")
        Re_max = getattr(params_t, f"{prefix}_Re_max")
        Re_min = getattr(params_t, f"{prefix}_Re_min")
        dt = getattr(params_t, f"{prefix}_dt")
        L = getattr(params_t, f"{prefix}_L")

        result = generate_combinations(U_range, rho_range, mu_range, Re_max, Re_min, source_range, aoa_range, dt ,L=L, eqtype="fluid")
        
    # print(f"满足雷诺数 {Re_max} 限制的[U, rho, mu]组合是: {result}")