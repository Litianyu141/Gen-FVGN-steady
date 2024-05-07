import torch
import argparse
from torch.optim import AdamW
import torch_geometric.transforms as T
import numpy as np
from FVMmodel.GNNSolver import GenFVGN, manual_eval
import os
from dataset import Load_mesh
from utils import get_param, utilities
from utils.utilities import extract_cylinder_boundary_mask, extract_cylinder_boundary
from utils.get_param import get_hyperparam
from utils.Logger import Logger, t_step
import time
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader as torch_geometric_DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from math import ceil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
import pandas as pd
from Extract_mesh import write_tec
from circle_fit import hyper_fit
from torch_scatter import scatter_add
import csv
import traceback


def loss_function(x):
    return torch.pow(x, 2)


def plot_residuals(filename, rollout_index, re=None):
    # Load data
    data = pd.read_csv(filename)

    plt.figure(figsize=(10, 6))

    plt.semilogy(
        data["continuity_equation_residuals"],
        label="Continuity Equation Residuals",
        color="red",
    )
    plt.semilogy(
        data["x_momentum_residuals"], label="X Momentum Residuals", color="green"
    )
    plt.semilogy(
        data["y_momentum_residuals"], label="Y Momentum Residuals", color="blue"
    )

    plt.title("Residuals")
    plt.xlabel("Epoch")
    plt.ylabel("Residual")

    ax = plt.gca()  # Get the current axes instance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show plot
    saving_dir = os.path.dirname(filename)
    plt.savefig(
        f"{saving_dir}/re{str(re)}_mu{str(mu)}_{str(rollout_time_length)}steps_{rollout_index}_finetune({str(args.finetune)})_residuals.png",
        dpi=300,
    )
    plt.show()


def cal_relonyds_number(graph_node, graph_edge):
    graph_node = graph_node.clone().cpu()
    graph_edge = graph_edge.clone().cpu()

    target_on_edge = (
        torch.index_select(graph_node.y, 0, graph_node.edge_index[0])
        + torch.index_select(graph_node.y, 0, graph_node.edge_index[1])
    ) / 2.0
    face_type = graph_edge.x[:, 0].view(-1)
    _batch_edge = graph_edge.batch[[face_type == utilities.NodeType.INFLOW]]
    Inlet = target_on_edge[face_type == utilities.NodeType.INFLOW][:, 0]
    face_length = graph_edge.x[:, 1][face_type == utilities.NodeType.INFLOW]
    total_u = torch.sum(Inlet * face_length)
    top = torch.max(graph_node.pos[:, 1]).numpy()
    bottom = torch.min(graph_node.pos[:, 1]).cpu().numpy()
    left = torch.min(graph_node.pos[:, 0]).numpy()
    right = torch.max(graph_node.pos[:, 0]).numpy()
    mean_u = total_u / (top - bottom)

    mean_u_list = []
    Re_num_list = []
    total_u = scatter_add(Inlet * face_length, _batch_edge, dim=0)
    graph_node_list = Batch.to_data_list(graph_node)
    for index in range(len(graph_node_list)):
        graph = graph_node_list[index]
        node_type = graph.node_type[:, 0].view(-1)
        top = torch.max(graph.pos[:, 1]).numpy()
        bottom = torch.min(graph.pos[:, 1]).cpu().numpy()
        mean_u = total_u[index] / (top - bottom)
        mean_u_list.append(mean_u)

        boundary_pos = graph.pos[node_type == utilities.NodeType.WALL_BOUNDARY].numpy()
        cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1).numpy()
        cylinder_not_mask = np.logical_not(cylinder_mask)
        cylinder_mask = np.where(
            (
                (boundary_pos[:, 1] > bottom)
                & (boundary_pos[:, 1] < top)
                & (boundary_pos[:, 0] > left)
                & (boundary_pos[:, 0] < right)
            ),
            cylinder_mask,
            cylinder_not_mask,
        )

        cylinder_pos = torch.from_numpy(boundary_pos[cylinder_mask])

        xc, yc, R, _ = hyper_fit(np.asarray(cylinder_pos))

        L0 = R * 2.0

        Re_num_list.append((mean_u * L0 * 1.0) / 0.001)

    return Re_num_list, mean_u_list


def write_result(
    fluid_zone_list=None,
    result_dir=None,
    rollout_time_length=None,
    with_boundary=False,
    rho=None,
    mu=None,
    source=None,
    dt=None,
    whether_has_boundary=None,
    rollout_index=None,
):
    for fluid_zone_index, fluid_zone in enumerate(fluid_zone_list):
        graph_node = fluid_zone["graph_node"]
        graph_edge = fluid_zone["graph_edge"]
        graph_cell = fluid_zone["graph_cell"]

        saving_path = f"{result_dir}/NO.{rollout_index}_mean_u{mean_u}_rho_{rho}_mu_{mu}_source{source}_dt_{dt}_aoa_{aoa}_{str(rollout_time_length)}steps_finetune({str(args.finetune)}).dat"

        # extrac interior zone
        interior_zone = {"name": "Fliud", "rho": rho, "mu": mu, "dt": dt}

        interior_zone["node|X"] = graph_node.pos[:, 0:1].to("cpu").unsqueeze(0).numpy()
        interior_zone["node|Y"] = graph_node.pos[:, 1:2].to("cpu").unsqueeze(0).numpy()
        interior_zone["node|U"] = (
            fluid_zone["predicted_node_uvp"][:, :, 0:1].to("cpu").numpy()
        )
        interior_zone["node|V"] = (
            fluid_zone["predicted_node_uvp"][:, :, 1:2].to("cpu").numpy()
        )
        interior_zone["node|P"] = (
            fluid_zone["predicted_node_uvp"][:, :, 2:3].to("cpu").numpy()
        )
        interior_zone["cells_node"] = (
            graph_node.face.to("cpu").transpose(0, 1).unsqueeze(0).numpy()
        )
        interior_zone["cells_index"] = graph_cell.face.to("cpu").T.unsqueeze(0).numpy()
        interior_zone["face_node"] = (
            graph_node.edge_index.to("cpu").transpose(0, 1).unsqueeze(0).numpy()
        )
        interior_zone["neighbour_cell"] = (
            graph_cell.edge_index.to("cpu").T.unsqueeze(0).numpy()
        )

        if with_boundary and whether_has_boundary:
            cylinder_node_mask, cylinder_face_mask = extract_cylinder_boundary_mask(
                graph_node=graph_node, graph_edge=graph_edge, graph_cell=graph_cell
            )

            if cylinder_node_mask is not None:
                boundary_zone = extract_cylinder_boundary(
                    fluid_zone,
                    cylinder_node_mask,
                    cylinder_face_mask,
                    graph_node,
                    graph_edge,
                    graph_cell,
                    rho=rho,
                    mu=mu,
                    dt=dt,
                )
                write_zone = [interior_zone, boundary_zone]
            else:
                boundary_zone = None

        else:
            write_zone = [interior_zone, None]

        write_tec.write_tecplotzone_test(
            saving_path,
            datasets=write_zone,
            time_step_length=interior_zone["node|U"].shape[0],
        )


def store_predicted_uvp(
    predicted_node_uvp=None,
    predicted_face_uvp=None,
    predicted_cell_uvp=None,
    predicted_zone=None,
    graph_node=None,
    graph_edge=None,
    graph_cell=None,
    end=False,
    save_last=False,
):
    if not save_last:
        predicted_zone["predicted_node_uvp_list"].append(predicted_node_uvp)
        # predicted_zone["predicted_edge_uvp_list"].append(predicted_face_uvp)
        # predicted_zone["predicted_cell_uvp_list"].append(predicted_cell_uvp)
    else:
        if end:
            predicted_zone["predicted_node_uvp_list"].append(predicted_node_uvp)
            # predicted_zone["predicted_edge_uvp_list"].append(predicted_face_uvp)
            # predicted_zone["predicted_cell_uvp_list"].append(predicted_cell_uvp)
    if end:
        """
        shape: [batch_index,rollout_time_length,node/edge/cell_num,channal_num]
        """
        predicted_zone["predicted_node_uvp_list"] = torch.stack(
            predicted_zone["predicted_node_uvp_list"], dim=0
        )
        # predicted_zone["predicted_edge_uvp_list"] = torch.stack(
        #     predicted_zone["predicted_edge_uvp_list"], dim=0
        # )
        # predicted_zone["predicted_cell_uvp_list"] = torch.stack(
        #     predicted_zone["predicted_cell_uvp_list"], dim=0
        # )

        split_predicted_zone = []

        batch_node = graph_node.batch.to(predicted_node_uvp.device)
        batch_edge = graph_edge.batch.to(predicted_node_uvp.device)
        batch_cell = graph_cell.batch.to(predicted_node_uvp.device)

        num_graph = graph_node.num_graphs

        graph_cell_sets = Batch.to_data_list(graph_cell)
        graph_edge_sets = Batch.to_data_list(graph_edge)
        graph_node_sets = Batch.to_data_list(graph_node)

        for i in range(num_graph):
            current_zone = {"zonename": "Fluid"}
            mask_node = batch_node == i
            mask_edge = batch_edge == i
            mask_cell = batch_cell == i
            current_zone["predicted_node_uvp"] = predicted_zone[
                "predicted_node_uvp_list"
            ][:, mask_node, :]
            # current_zone["predicted_face_uvp"] = predicted_zone[
            #     "predicted_edge_uvp_list"
            # ][:, mask_edge, :]
            # current_zone["predicted_cell_uvp"] = predicted_zone[
            #     "predicted_cell_uvp_list"
            # ][:, mask_cell, :]
            # current_zone["data_target_uvp_node"] = graph_node.data_target_on_node[mask_node,:,:]
            current_zone["graph_node"] = graph_node_sets[i]
            current_zone["graph_edge"] = graph_edge_sets[i]
            current_zone["graph_cell"] = graph_cell_sets[i]

            split_predicted_zone.append(current_zone)

        predicted_zone = split_predicted_zone

    return predicted_zone


@torch.no_grad()
def rollout_without_fintune(
    fluid_model=None,
    dataset=None,
    rollout_index=0,
    result_dir=None,
    rollout_start_step=None,
    rollout_time_length=None,
    rho=None,
    mu=None,
    source=None,
    aoa=None,
    dt=None,
    optimizer=None,
    with_boundary=False,
    args=None,
):
    predicted_zone = {
        "predicted_node_uvp_list": [],
        "predicted_edge_uvp_list": [],
        "predicted_cell_uvp_list": [],
    }

    (
        graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_cell_x,
        whether_has_boundary,
        origin_mesh_file_location,
    ) = loader.get_specific_data([rollout_index])
    
    (
        graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_cell_x,
    ) = dataset.datapreprocessing(
        graph_node=graph_node.cuda(),
        graph_node_x=graph_node_x.cuda(),
        graph_edge=graph_edge.cuda(),
        graph_cell=graph_cell.cuda(),
        graph_cell_x=graph_cell_x.cuda(),
        dimless=params.dimless,
    )
    
    saving_dir = f"{result_dir}/mean_u{mean_u}_mu{str(mu)}_{str(rollout_time_length)}steps_finetune({str(args.finetune)})_{rollout_index}"

    os.makedirs(saving_dir, exist_ok=True)

    # do not reset the env during inference
    dataset._set_reset_env_flag(flag=False)

    # save origin mesh file path
    with open(f"{saving_dir}/origin_mesh_path.txt", "w") as mesh_path_file:
        mesh_path_file.write(origin_mesh_file_location[0])

    # resdiual monitor
    with open(
        f"{saving_dir}/NO.{rollout_index}_mean_u{mean_u}_rho_{rho}_mu_{mu}_source{source}_dt_{dt}_aoa_{aoa}_{str(rollout_time_length)}steps_finetune({str(args.finetune)})_residuals.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "continuity_equation_residuals",
                "x_momentum_residuals",
                "y_momentum_residuals",
            ]
        )

        for epoch in range(rollout_start_step, rollout_time_length):
            last_iteration = epoch == rollout_time_length - 1

            # forwarding the model,graph_old`s cell and edge attr has been normalized but without model upadte
            (
                predicted_node_uvp,
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y
            ) = fluid_model(
            graph_cell=graph_cell,
            graph_node_x=graph_node_x,
            graph_edge=graph_edge,
            graph_node=graph_node,
            graph_cell_x=graph_cell_x,
            params=params,
            )


            predicted_zone = store_predicted_uvp(
                predicted_node_uvp=predicted_node_uvp.detach(),
                predicted_zone=predicted_zone,
                graph_node=graph_node.detach(),
                graph_edge=graph_edge.detach(),
                graph_cell=graph_cell.detach(),
                end=last_iteration,
                save_last=args.save_last,
            )

            continuity_equation_residuals = abs(loss_cont.detach().mean().item())
            x_momentum_residuals = abs(loss_momtentum_x.detach().mean().item())
            y_momentum_residuals = abs(loss_momtentum_y.detach().mean().item())

            writer.writerow(
                [
                    continuity_equation_residuals,
                    x_momentum_residuals,
                    y_momentum_residuals,
                ]
            )

            if not last_iteration:
                graph_node.x = predicted_node_uvp.detach()
                (
                    graph_node,
                    graph_node_x,
                    graph_edge,
                    graph_cell,
                    graph_cell_x,
                ) = dataset.create_next_graph(
                    graph_node.detach(),
                    graph_node_x.detach(),
                    graph_edge.detach(),
                    graph_cell.detach(),
                    graph_cell_x.detach(),
                )

    plot_residuals(
        f"{saving_dir}/NO.{rollout_index}_mean_u{mean_u}_rho_{rho}_mu_{mu}_source{source}_dt_{dt}_aoa_{aoa}_{str(rollout_time_length)}steps_finetune({str(args.finetune)})_residuals.csv",
        rollout_index={rollout_index},
        re=graph_node.pde_theta_node[0, -1].cpu().numpy(),
    )

    write_result(
        fluid_zone_list=predicted_zone,
        result_dir=saving_dir,
        rollout_time_length=rollout_time_length,
        with_boundary=with_boundary,
        rho=rho,
        mu=mu,
        source=source,
        dt=dt,
        whether_has_boundary=whether_has_boundary,
        rollout_index=rollout_index,
    )


def rollout_with_fintune(
    fluid_model=None,
    dataset=None,
    rollout_index=0,
    result_dir=None,
    rollout_start_step=None,
    rollout_time_length=None,
    rho=None,
    mu=None,
    source=None,
    aoa=None,
    dt=None,
    optimizer=None,
    with_boundary=False,
    args=None,
):
    predicted_zone = {
        "predicted_node_uvp_list": [],
        "predicted_edge_uvp_list": [],
        "predicted_cell_uvp_list": [],
    }

    (
        graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_cell_x,
        whether_has_boundary,
        origin_mesh_file_location,
    ) = loader.get_specific_data([rollout_index])
    
    (
        graph_node,
        graph_node_x,
        graph_edge,
        graph_cell,
        graph_cell_x,
    ) = dataset.datapreprocessing(
        graph_node=graph_node.cuda(),
        graph_node_x=graph_node_x.cuda(),
        graph_edge=graph_edge.cuda(),
        graph_cell=graph_cell.cuda(),
        graph_cell_x=graph_cell_x.cuda(),
        dimless=params.dimless,
    )

    saving_dir = f"{result_dir}/mean_u{mean_u}_mu{str(mu)}_{str(rollout_time_length)}steps_finetune({str(args.finetune)})_{rollout_index}"

    os.makedirs(saving_dir, exist_ok=True)

    # do not reset the env during inference
    dataset._set_reset_env_flag(flag=False)

    # save origin mesh file path
    with open(f"{saving_dir}/origin_mesh_path.txt", "w") as mesh_path_file:
        mesh_path_file.write(origin_mesh_file_location[0])

    # resdiual monitor
    with open(
        f"{saving_dir}/NO.{rollout_index}_mean_u{mean_u}_rho_{rho}_mu_{mu}_source{source}_dt_{dt}_aoa_{aoa}_{str(rollout_time_length)}steps_finetune({str(args.finetune)})_residuals.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "continuity_equation_residuals",
                "x_momentum_residuals",
                "y_momentum_residuals",
            ]
        )

        for epoch in range(rollout_start_step, rollout_time_length):
            last_iteration = epoch == rollout_time_length - 1

            # forwarding the model,graph_old`s cell and edge attr has been normalized but without model upadte
            (
                predicted_node_uvp,
                loss_cont,
                loss_momtentum_x,
                loss_momtentum_y,
                loss_pressure_outlet,
                projection_method,
            ) = fluid_model(
                graph_cell=graph_cell,
                graph_node_x=graph_node_x,
                graph_edge=graph_edge,
                graph_node=graph_node,
                graph_cell_x=graph_cell_x,
                params=params,
            )

            """ loss on boundary"""
            loss_cont_L2 = global_mean_pool(
                loss_function(loss_cont), batch=graph_cell.batch.cuda()
            ).view(-1)
            try:
                loss_projection_method_L2 = global_mean_pool(
                    loss_function(projection_method), batch=graph_cell.batch.cuda()
                ).view(-1)
            except:
                loss_projection_method_L2 = 0.0
            try:
                loss_pressure_outlet_L2 = global_mean_pool(
                    loss_function(loss_pressure_outlet), batch=graph_cell.batch.cuda()
                ).view(-1)
            except:
                loss_pressure_outlet_L2 = 0.0

            loss_mom_L2 = global_mean_pool(
                loss_function(loss_momtentum_x), batch=graph_cell.batch.cuda()
            ).view(-1) + global_mean_pool(
                loss_function(loss_momtentum_y), batch=graph_cell.batch.cuda()
            ).view(
                -1
            )

            loss = (
                params.loss_cont * loss_cont_L2
                + params.loss_mom * loss_mom_L2
                + params.loss_projection_method * loss_projection_method_L2
                + params.pressure_open_bc * loss_pressure_outlet_L2
            )

            loss = params.loss_multiplier * torch.mean(torch.log(loss))

            try:
                # reset gradients
                optimizer.zero_grad()

                # compute gradients
                loss.backward()

                # perform optimization step
                optimizer.step()

            except Exception as e:
                print("Error: loss.backward() failed")
                print("Exception message:\n", e)
                print("Traceback:\n", traceback.format_exc())
                pass

            predicted_zone = store_predicted_uvp(
                predicted_node_uvp=predicted_node_uvp.detach(),
                predicted_zone=predicted_zone,
                graph_node=graph_node.detach(),
                graph_edge=graph_edge.detach(),
                graph_cell=graph_cell.detach(),
                end=last_iteration,
                save_last=args.save_last,
            )

            continuity_equation_residuals = abs(loss_cont.detach().mean().item())
            x_momentum_residuals = abs(loss_momtentum_x.detach().mean().item())
            y_momentum_residuals = abs(loss_momtentum_y.detach().mean().item())

            writer.writerow(
                [
                    continuity_equation_residuals,
                    x_momentum_residuals,
                    y_momentum_residuals,
                ]
            )

            if not last_iteration:
                graph_node.x = predicted_node_uvp.detach()
                (
                    graph_node,
                    graph_node_x,
                    graph_edge,
                    graph_cell,
                ) = dataset.create_next_graph(
                    graph_node.detach(),
                    graph_node_x.detach(),
                    graph_edge.detach(),
                    graph_cell.detach(),
                )

    plot_residuals(
        f"{saving_dir}/NO.{rollout_index}_mean_u{mean_u}_rho_{rho}_mu_{mu}_source{source}_dt_{dt}_aoa_{aoa}_{str(rollout_time_length)}steps_finetune({str(args.finetune)})_residuals.csv",
        rollout_index={rollout_index},
        re=graph_node.pde_theta_node[0, -1].cpu().numpy(),
    )

    write_result(
        fluid_zone_list=predicted_zone,
        result_dir=saving_dir,
        rollout_time_length=rollout_time_length,
        with_boundary=with_boundary,
        rho=rho,
        mu=mu,
        source=source,
        dt=dt,
        whether_has_boundary=whether_has_boundary,
        rollout_index=rollout_index,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    # configurate parameters
    def str2bool(v):
        """
        'boolean type variable' for add_argument
        """
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("boolean value expected.")

    parser = argparse.ArgumentParser(description="Implementation of MeshGraphNets")
    parser.add_argument("--gpu", type=int, default=1, help="gpu number: 0 or 1")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/doomduke/GEP-FVGN/Logger/net GN-Cell; hs 64; training_flow_type pf_ff_cf_p;/2024-01-02-13:17:08-nodewlsq-steady-final-lr=1e-6/states/1460.state",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="test batch size at once forward"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--rollout_num", type=int, default=1)
    parser.add_argument("--rollout_start_step", type=int, default=0)
    parser.add_argument("--rollout_time_length", type=int, default=500)
    parser.add_argument(
        "--cavit_equation",
        type=str,
        default="cavity_flow",
        choices=["cavity_flow", "possion", "cavity_wave"],
    )
    parser.add_argument(
        "--spec_u_rho_mu_comb",
        nargs="*",
        type=float,
        default=[2.5, 0, 0.1, 5, 0, 1],
        help="mean_u, rho, mu ,source,aoa and dt [mean_u,rho,mu,source,dt](default: [0.1, 1, 0.001, 0, 0, 1])",
    )
    parser.add_argument(
        "--farfield_bc_type",
        type=str,
        default="uniform_velocity_field",
        choices=["uniform_velocity_field", "parabolic_velocity_field"],
    )
    parser.add_argument(
        "--rollout_index",
        type=int,
        default=907,
    )
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--finetune", type=bool, default=False)

    args = parser.parse_args()
    params = get_param.params(os.path.split(args.model_dir)[0])
    params.rollout = True
    params.cavit_equation = args.cavit_equation
    
    # gpu devices
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_one_hot = params.edge_one_hot
    cell_one_hot = params.cell_one_hot

    # initialize flow parameters
    rollout_time_length = args.rollout_time_length
    mean_u, rho, mu, source, aoa, dt = args.spec_u_rho_mu_comb
    print("fluid parameters rho:{0} mu:{1} dt:{2}".format(rho, mu, dt))

    # initialize Logger and load model / optimizer if according parameters were given
    logger = Logger(
        get_hyperparam(params), use_csv=False, use_tensorboard=False, copy_code=False
    )
    params.load_index = 0 if params.load_index is None else params.load_index


    # initialize Training Dataset
    start = time.time()
    datasets_factory = Load_mesh.DatasetFactory(
        params=params,
        is_training=False,
        split="train",
        dataset_dir=params.dataset_dir4,
        spec_u_rho_mu_comb=args.spec_u_rho_mu_comb,
        inflow_bc_type="parabolic_velocity_field",
        device=device,
    )

    # refresh dataset size
    params.dataset_size = datasets_factory.dataset_size

    # create dataset objetc
    dataset, loader, sampler = datasets_factory.create_datasets(
        batch_size=params.batch_size, num_workers=0, pin_memory=False
    )

    end = time.time()
    print("traj has been loaded time consuming:{0}".format(end - start))

    # initialize fluid model
    fluid_model = GenFVGN(
        message_passing_num=params.message_passing_num,
        node_input_size=params.node_input_size + params.node_one_hot,
        edge_input_size=params.edge_input_size + params.edge_one_hot,
        cell_input_size=params.cell_input_size + params.cell_one_hot,
        node_output_size=params.node_output_size,
        edge_output_size=params.edge_output_size,
        cell_output_size=params.cell_output_size,
        drop_out=params.drop_out,
        attention=params.attention,
        MultiHead=params.multihead,
        hidden_size=params.hidden_size,
        normlizer_steps=25 * ceil(params.dataset_size / params.batch_size),
        device=device,
    )

    fluid_model.load_checkpoint(device=device, is_training=False, ckpdir=args.model_dir)
    fluid_model = fluid_model.to(device)
    optimizer = AdamW(fluid_model.parameters(), lr=1e-4)
    # fluid_model.eval()

    result_dir = f"{os.path.split(os.path.split(args.model_dir)[0])[0]}/rollout_with_{args.split}_dataset"
    nepochs = os.path.split(args.model_dir)[1].split(".")[0]
    dates = params.git_commit_dates.replace(":", "_").replace("+", "_")

    result_dir = f"{result_dir}/epochs_{nepochs}"
    os.makedirs(result_dir, exist_ok=True)

    rollout_start_step = args.rollout_start_step
    roll_outs = []
    last_time = time.time()


    result_dir = f"{result_dir}/rollout_index_{args.rollout_index}"
    os.makedirs(result_dir, exist_ok=True)

    if args.finetune:
        fluid_model.train()
        rollout_with_fintune(
            fluid_model=fluid_model,
            dataset=dataset,
            rollout_index=args.rollout_index,
            result_dir=result_dir,
            rollout_start_step=rollout_start_step,
            rollout_time_length=rollout_time_length,
            rho=rho,
            mu=mu,
            source=source,
            aoa=aoa,
            dt=dt,
            optimizer=optimizer,
            with_boundary=True,
            args=args,
        )

    else:
        fluid_model.eval()
        rollout_without_fintune(
            fluid_model=fluid_model,
            dataset=dataset,
            rollout_index=args.rollout_index,
            result_dir=result_dir,
            rollout_start_step=rollout_start_step,
            rollout_time_length=rollout_time_length,
            rho=rho,
            mu=mu,
            source=source,
            aoa=aoa,
            dt=dt,
            optimizer=optimizer,
            with_boundary=True,
            args=args,
        )
