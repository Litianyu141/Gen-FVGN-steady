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
        params = vars(parser.parse_args())
        with open(load+'/commandline_args.json', 'rt') as f:
            params.update(json.load(f))
        for k, v in params.items():
            parser.add_argument('--' + k, default=v)
        args = parser.parse_args()
        return  args
    else:
        """
        return parameters for training / testing / plotting of models
        :return: parameter-Namespace
        """
        parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')

        # Training parameters
        parser.add_argument('--net', default="GN-Cell", type=str, help='network to train (default: GN-Cell)', choices=["GN-Cell","GN-Node"])
        parser.add_argument('--n_epochs', default=300000, type=int, help='number of epochs (after each epoch, the model gets saved)')
        parser.add_argument('--traj_length', default=600, type=int, help='dataset traj_length (default: cylinder 599)')
        parser.add_argument('--batch_size', default=8, type=int, help='batch size (default: 100)')
        parser.add_argument('--train_traj_length', default=1000, type=int, help='number of time steps to train  (default: 300)')
        parser.add_argument('--average_sequence_length', default=500, type=int, help='average sequence length in dataset=dataset_size/batch_size(default: 5000)')
        parser.add_argument('--iner_step', default=1, type=int, help='inner iter steps =  n_batches_per_epoch/average_sequence_length (default: 5000)')
        parser.add_argument('--dataset_size', default=5, type=int, help='size of dataset (default: 1000)')
        parser.add_argument('--all_on_gpu', default=False, type=str2bool, help='whether put all dataset on GPU')
        parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of optimizer (default: 0.0001)')
        parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')
        parser.add_argument('--rollout', default=False, type=str2bool, help='rolling out or not (turn off for debugging)')
        parser.add_argument('--on_gpu', default=0, type=int, help='set training on which gpu')
        
        # train strategy parameters
        parser.add_argument('--Noise_injection_factor', default=2e-2, type=float, help='factor for normal Noise distrubation,0 means No using Noise injection ,(default: 2e-2),choices=["0","Greater than 0"]')
        parser.add_argument('--integrator', default="implicit", type=str, help='integration scheme (explicit / implicit / imex) (default: imex)',choices=['explicit','implicit','imex'])
        parser.add_argument('--statistics_times', default=1, type=int, help='accumlate data statistics for normalization before backprapagation (default: 1)')
        parser.add_argument('--before_explr_decay_steps', default=150000, type=int, help='steps before using exp lr decay technique (default:12000)')
        parser.add_argument('--dimless', default=True, type=str2bool, help='Injetcting noise on face length for prevention of overfiting (default:False)')
        parser.add_argument('--scale_mesh', default=None, type=float, help='Injetcting noise on face length for prevention of overfiting (default:False)')
        parser.add_argument('--integrate_p', default=False, type=str2bool, help='Use WSLQ or divergence therom to calculate nabla P (default:False)')
        parser.add_argument('--ncn_smooth', default=False, type=str2bool, help='inteploting node value to cell and to node (default:False)')
        parser.add_argument('--prevent_reverse_flow', default=True, type=str2bool, help='Use artifacial wall method, which is setting boudary node to zero which it was flow into the domain (default:False)')
        parser.add_argument('--conserved_form', default=False, type=str2bool, help='Use artifacial wall method, which is setting boudary node to zero which it was flow into the domain (default:False)')
        parser.add_argument('--GG_convection', default=False, type=str2bool, help='Use divergence therom for convection term (default:False)')
        parser.add_argument('--training_flow_type', nargs='*', type=str, default=["pf", "ff", "cf","p"], help='trainging flow type(default: ["pf", "ff", "cf","p"])')
        parser.add_argument('--max_aoa', default=35, type=float, help='max angle of attack for farfield case')
        parser.add_argument('--flow_type', default="pipe_flow", type=str, help='timestep of fluid integrator',choices=['pipe_flow','cavity_flow'])
        parser.add_argument('--equation_state', default="steady_with_possion", type=str, help='determine whether use unsteady NS eq or possion eq',choices=['unsteady_only','steady_only','steady_with_possion','unsteady_with_possion'])
        
        # loss parameters
        parser.add_argument('--loss_cont', default=10, type=float, help='loss factor for continuity equation')
        parser.add_argument('--loss_cont_hat', default=0, type=float, help='loss factor for continuity equation')
        parser.add_argument('--loss_projection_method', default=0, type=float, help='use projection method to regularize pressure (default: 0)')
        parser.add_argument('--loss_mom', default=1, type=float, help='loss factor for uv diffusion flux on face')
        parser.add_argument('--loss', default='square', type=str, help='loss type to train network (default: square)',choices=['square'])
        parser.add_argument('--loss_multiplier', default=1, type=float, help='multiply loss / gradients (default: 1)')
        parser.add_argument('--pressure_open_bc', default=0.001, type=float, help='whether use p=p_spec,is set 0. it means use p=p_spec (default:0)')
        parser.add_argument('--projection_method', default=0, type=float, help='whether use p=p_spec,is set 0. it means use p=p_spec (default:0)')
        
        # Pipe Flow Fluid parameters
        group1 = parser.add_argument_group('pipe_flow', 'Pipe Flow Fluid parameters')
        group1.add_argument('--pf_inflow_range', nargs='*', type=float, default=[0.1, 0.001, 0.51], help='training normal inflow range [start,step,end](default: [1, 0.1, 2])')
        group1.add_argument('--pf_rho', nargs='*', type=float, default=[1, 1, 1], help='trainging rho range [start,step,end] (default: [0.01, 0.01, 1])')
        group1.add_argument('--pf_mu', nargs='*', type=float, default=[0.001, 0.001, 0.001], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group1.add_argument('--pf_source', nargs='*', type=float, default=[0, 0, 0], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group1.add_argument('--pf_aoa', nargs='*', type=float, default=[0, 0, 0], help='training aoa range [start,step,end](default: [0.001, 0.001, 0.01])')
        group1.add_argument('--pf_dt', default=0.075, type=float, help='timestep of fluid integrator')
        group1.add_argument('--pf_L', default=0.1, type=float, help='characteristic length')
        group1.add_argument('--pf_Re_max', default=1000, type=float, help='max Re number')
        group1.add_argument('--pf_Re_min', default=1, type=float, help='min Re number')

        # Farfield Flow Fluid parameters
        group2 = parser.add_argument_group('farfield_flow', 'Farfield Flow Fluid parameters')
        group2.add_argument('--ff_inflow_range', nargs='*', type=float, default=[1, 0.01, 5], help='training normal inflow range [start,step,end](default: [1, 0.1, 2])')
        group2.add_argument('--ff_rho', nargs='*', type=float, default=[1, 1, 1], help='trainging rho range [start,step,end] (default: [0.01, 0.01, 1])')
        group2.add_argument('--ff_mu', nargs='*', type=float, default=[0.001, 0.001, 0.001], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group2.add_argument('--ff_source', nargs='*', type=float, default=[0, 0, 0], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group2.add_argument('--ff_aoa', nargs='*', type=float, default=[0, 0, 0], help='training aoa range [start,step,end](default: [0.001, 0.001, 0.01])')
        group2.add_argument('--ff_dt', default=0.05, type=float, help='timestep of fluid integrator')
        group2.add_argument('--ff_L', default=1, type=float, help='characteristic length')
        group2.add_argument('--ff_Re_max', default=1.1e4, type=float, help='max Re number')
        group2.add_argument('--ff_Re_min', default=1e3, type=float, help='min Re number')

        # Cavity Flow Fluid parameters
        group3 = parser.add_argument_group('cavity_flow', 'Cavity Flow Fluid parameters')
        group3.add_argument('--cf_inflow_range', nargs='*', type=float, default=[1, 0.01, 4], help='training normal inflow range [start,step,end](default: [1, 0.1, 2])')
        group3.add_argument('--cf_rho', nargs='*', type=float, default=[1, 1, 1], help='trainging rho range [start,step,end] (default: [0.01, 0.01, 1])')
        group3.add_argument('--cf_mu', nargs='*', type=float, default=[0.01, 0.01, 0.01], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group3.add_argument('--cf_source', nargs='*', type=float, default=[0, 0, 0], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group3.add_argument('--cf_aoa', nargs='*', type=float, default=[0, 0, 0], help='training aoa range [start,step,end](default: [0.001, 0.001, 0.01])')
        group3.add_argument('--cf_dt', default=1, type=float, help='timestep of fluid integrator')
        group3.add_argument('--cf_L', default=1, type=float, help='characteristic length')
        group3.add_argument('--cf_Re_max', default=1e4, type=float, help='max Re number')
        group3.add_argument('--cf_Re_min', default=1e2, type=float, help='min Re number')
         
        # Possion equation parameters
        group4 = parser.add_argument_group('possion', 'Possion equation parameters')
        group4.add_argument('--p_inflow_range', nargs='*', type=float, default=[1, 0.1, 6], help='training normal inflow range [start,step,end](default: [1, 0.1, 2])')
        group4.add_argument('--p_rho', nargs='*', type=float, default=[0,0,0], help='trainging rho range [start,step,end] (default: [0.01, 0.01, 1])')
        group4.add_argument('--p_mu', nargs='*', type=float, default=[0.1, 0.1, 0.1], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group4.add_argument('--p_source', nargs='*', type=float, default=[0, 1, 6], help='training mu range [start,step,end](default: [0.001, 0.001, 0.01])')
        group4.add_argument('--p_aoa', nargs='*', type=float, default=[0, 0, 0], help='training 0 range [start,step,end](default: [0.001, 0.001, 0.01])')
        group4.add_argument('--p_dt', default=1, type=float, help='timestep of fluid integrator')
        group4.add_argument('--p_L', default=1, type=float, help='characteristic length')
        group4.add_argument('--p_Re_max', default=1e12, type=float, help='max Re number')
        group4.add_argument('--p_Re_min', default=0, type=float, help='min Re number')
        
        # Cavity Wave parameters
        group5 = parser.add_argument_group('cavity_wave', 'Cavity Wave parameters')
        group5.add_argument('--cw_source_frequency', nargs='*', type=float, default=[100, 10, 1000], help='training source_frequency range [start,step,end](default: [1, 0.1, 2])')
        group5.add_argument('--cw_source_strength', nargs='*', type=float, default=[1, 1, 10], help='training source_strength range [start,step,end](default: [0.001, 0.001, 0.01])')
        group5.add_argument('--cw_rho', nargs='*', type=float, default=[1, 1, 1], help='trainging rho range [start,step,end] (default: [0.01, 0.01, 1])')
        group5.add_argument('--cw_dt', default=0.01, type=float, help='timestep of fluid integrator')
        
        # Load parameters
        parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
        parser.add_argument('--load_index', default=None , type=int, help='index of run to load (default: None)')
        parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: True)')
        parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
        
        #model parameters
        parser.add_argument('--hidden_size', default=64, type=int, help='hidden size of network (default: 20)')
        parser.add_argument('--message_passing_num', default=9, type=int, help='message passing layer number (default:15)')
        parser.add_argument('--node_input_size', default=9, type=int, help='node encoder node_input_size (default: 2)')
        parser.add_argument('--edge_input_size', default=6, type=int, help='edge encoder edge_input_size, include edge center pos (x,y) (default: 3)')
        parser.add_argument('--cell_input_size', default=13, type=int, help='cell encoder cell_input_size, include uvp (default: 3)')
        parser.add_argument('--node_one_hot', default=16, type=int, help='edge one hot dimention (default: 9)')
        parser.add_argument('--node_normlizer_input_size', default=13, type=int, help='edge normlizer edge_input_size (default: 2)')
        parser.add_argument('--edge_one_hot', default=0, type=int, help='edge one hot dimention (default: 9)')
        parser.add_argument('--edge_normlizer_input_size', default=3, type=int, help='edge normlizer edge_input_size (default: 2)')
        parser.add_argument('--cell_one_hot', default=0, type=int, help='cell one hot dimention (default: 9)')
        parser.add_argument('--cell_normlizer_input_size', default=2, type=int, help='cell normlizer edge_input_size (default: 2)')	
        parser.add_argument('--cell_target_input_size', default=2, type=int, help='cell normlizer cell_target_input_size (default: 2)')
        parser.add_argument('--cell_p_target_input_size', default=1, type=int, help='cell pressure normlizer cell_p_target_input_size (default: 1)')
        parser.add_argument('--face_flux_target_input_size', default=3, type=int, help='face_flux_target_input_size include uvp (default: 3)')
        parser.add_argument('--node_output_size', default=3, type=int, help='edge decoder edge_output_size uvp on edge center(default: 8)')
        parser.add_argument('--edge_output_size', default=3, type=int, help='edge decoder edge_output_size uvp on edge center(default: 8)')
        parser.add_argument('--cell_output_size', default=3, type=int, help='cell decoder cell_output_size uvp on cell center(default: 1)')
        parser.add_argument('--face_length_size', default=1, type=int, help='face_length_normlizer input_size (default: 1)')
        parser.add_argument('--cell_area_size', default=1, type=int, help='cell_area_normlizer input_size_size (default: 1)')
        parser.add_argument('--unv_size', default=2, type=int, help='unv_normlizer input_size_size (default: 1)')
        parser.add_argument('--drop_out', default=False, type=str2bool, help='using dropout technique in message passing layer(default: True)')
        parser.add_argument('--attention', default=False, type=str2bool, help='using dropout technique in message passing layer(default: True)')
        parser.add_argument('--multihead', default=1, type=int, help='using dropout technique in message passing layer(default: True)')

        #dataset params
        parser.add_argument('--dataset_type', default='tf', type=str, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
        parser.add_argument('--dataset_dir', default='/mnt/c/Users/DOOMDUKE2-lab/Desktop/Dataset_Grad_rec_test/cavity/tri/converted_dataset/tf', type=str, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')

          #git information
        if True:
            import git
            currentdir = os.getcwd()
            repo = git.Repo(currentdir)
            current_branch=repo.active_branch.name
            commits = list(repo.iter_commits(current_branch, max_count=5))   
            parser.add_argument('--git_branch', default=current_branch, type=str, help='current running code`s git branch')
            parser.add_argument('--git_messages', default=commits[0].message, type=str, help='current running code`s git messages')
            parser.add_argument('--git_commit_dates', default=str(commits[0].authored_datetime), type=str, help='current running code`s git commit date')
            params = parser.parse_args()
            
            # git information
            git_info = {"git_branch":params.git_branch,
            "git_commit_dates":params.git_commit_dates}
            # parse parameters
            
            return params,git_info
        else:
            parser.add_argument('--git_branch', default="FVGN-pde-jtedu smaller tanh factor,test no prevent oversmooth still normalize,lr on bc=1e-2", type=str, help='current running code git branch')
            parser.add_argument('--git_commit_dates', default="March 14th, 2023 10:56 PM", type=str, help='current running code git commit date')	
            params = parser.parse_args()
            git_info = {"git_branch":params.git_branch,
            "git_commit_dates":params.git_commit_dates}
            return params,git_info

def get_hyperparam(params):
    return f"net {params.net}; hs {params.hidden_size}; training_flow_type {'_'.join(params.training_flow_type)};"

def generate_list(min_val, step, max_val):
    if min_val == step == max_val:
        return [max_val]
    else:
        # 使用linspace可以确保开始和结束的值都包括在内
        # 并根据步长计算必要的点数
        num_points = int((max_val - min_val) / step) + 1
        return list(np.linspace(min_val, max_val, num_points))
    
def generate_combinations(U_range, rho_range, mu_range, Re_max, Re_min, source_range, aoa_range, dt, L=1):
    
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
        Re = (U * rho * L) / mu
        if (Re <= Re_max and Re>=Re_min):
            valid_combinations.append([U, rho, mu, source, aoa_list, dt])
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
    
    U_range = getattr(params_t, f"{prefix}_inflow_range")
    rho_range = getattr(params_t, f"{prefix}_rho")
    mu_range = getattr(params_t, f"{prefix}_mu")
    source_range = getattr(params_t, f"{prefix}_source")
    aoa_range = getattr(params_t, f"{prefix}_aoa")
    Re_max = getattr(params_t, f"{prefix}_Re_max")
    Re_min = getattr(params_t, f"{prefix}_Re_min")
    dt = getattr(params_t, f"{prefix}_dt")
    L = getattr(params_t, f"{prefix}_L")

    result = generate_combinations(U_range, rho_range, mu_range, Re_max, Re_min, source_range, aoa_range, dt ,L=L)
    # print(f"满足雷诺数 {Re_max} 限制的[U, rho, mu]组合是: {result}")