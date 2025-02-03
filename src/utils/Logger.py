import os
import time
import torch
import numpy as np
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from statsmodels.nonparametric.smoothers_lowess import lowess
from Post_process import to_vtk
import csv

# import warnings
from natsort import natsorted
import pickle
import json
import shutil
import fnmatch


# import utilities
class Logger:
    def __init__(
        self,
        name,
        head="Logger",
        datetime=None,
        use_csv=False,
        use_tensorboard=False,
        params=None,
        saving_path=None,
        copy_code=False,
        seed=None,
    ):
        """
        Logger logs metrics to CSV files / tensorboard
        :name: logging name (e.g. model name / dataset name / ...)
        :datetime: date and time of logging start (useful in case of multiple runs). Default: current date and time is picked
        :use_csv: log output to csv files (needed for plotting)
        :use_tensorboard: log output to tensorboard
        """
        self.head = head
        self.name = name
        self.params = params
        self.log_item = {}

        if datetime:
            self.datetime = datetime
        else:
            self.datetime = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if saving_path is not None:
            self.saving_path = saving_path
        else:
            self.saving_path = os.getcwd() + f"/{head}/{name}/{self.datetime}"

        if seed is not None:
            os.makedirs(f"{self.saving_path}/seed_{seed}", exist_ok=True)

        source_valid_file_path = os.path.split(os.path.split(__file__)[0])[0]
        target_valid_file_path = f"{self.saving_path}/source"
        
        if copy_code:
            os.makedirs(f"{self.saving_path}/source", exist_ok=True)
            whitelist = [
                "*.py",
                "*.txt",
                "*.ipynb",
                "BC_encoder",
                "Extract_mesh",
                "FVMmodel",
                "Load_mesh",
                "Utils",
                "Post_process",
            ]
            self.copy_code_and_folders(
                source_valid_file_path, target_valid_file_path, whitelist
            )

        self.use_tensorboard = use_tensorboard

        self.use_csv = use_csv
        if use_csv:
            self.csv_file_path = f"{self.saving_path}/Loss_monitor.dat"
            self.headers = ["epoch"]
            self._write_header_to_csv(self.headers)

        if use_tensorboard:
            directory = self.saving_path + "/tensorboard"
            os.makedirs(directory, exist_ok=True)
            self.writer = SummaryWriter(directory)

    def copy_code_and_folders(self, src_dir, dst_dir, whitelist=None):
        """
        Copy files and folders from src_dir to dst_dir based on the whitelist.
        """
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for item in os.listdir(src_dir):
            src_item = os.path.join(src_dir, item)

            if os.path.isdir(src_item) and item in whitelist:
                # Copy directory if it's in the whitelist
                shutil.copytree(src_item, os.path.join(dst_dir, item))

            elif os.path.isfile(src_item):
                for pattern in whitelist:
                    if fnmatch.fnmatch(item, pattern):
                        # Copy file if it matches a pattern in the whitelist
                        shutil.copy2(src_item, os.path.join(dst_dir, item))
                        break

    def _write_header_to_csv(self, headers):
        header_line = "Variables=" + " ".join(f'"{header}"' for header in headers)
        with open(self.csv_file_path, "w") as file:
            file.write(header_line + "\n")

    def _update_csv_headers(self, new_headers):
        with open(self.csv_file_path, "r") as file:
            lines = file.readlines()

        header_line = "Variables=" + " ".join(f'"{header}"' for header in new_headers)
        lines[0] = header_line + "\n"

        with open(self.csv_file_path, "w") as file:
            file.writelines(lines)

    def add_log_item(self, item: str, value, index=None):
        if item not in self.log_item:
            self.log_item[item] = [value]
        else:
            self.log_item[item].append(value)

    def log(self, item, value, index):
        if self.use_csv:
            self.add_log_item(item, value, index)

            current_headers = ["epoch"] + list(self.log_item.keys())
            if set(current_headers) != set(self.headers):
                self.headers = current_headers
                self._update_csv_headers(self.headers)

            with open(self.csv_file_path, "a") as file:
                row = [index]
                for _, v in self.log_item.items():
                    row.append(v[-1])
                row_string = " ".join(str(item) for item in row)
                file.write(row_string + "\n")

        if self.use_tensorboard:
            self.writer.add_scalar(item, value, index)

    def log_histogram(self, item, values, index):
        """
        log index values-histogram couple for specific item to tensorboard
        :item: string describing item (e.g. "training_loss","test_loss")
        :values: values to log
        :index: index (e.g. batchindex / epoch)
        """
        if self.use_tensorboard:
            self.writer.add_histogram(item, values, index)

    def log_model_gradients(self, item, model, index):
        """
        log index model-gradients-histogram couple for specific item to tensorboard
        :item: string describing model item (e.g. "encoder","discriminator")
        :values: values to log
        :index: index (e.g. batchindex / epoch)
        """
        if self.use_tensorboard:
            params = [p for p in model.parameters()]
            if len(params) != 0:
                gradients = torch.cat(
                    [p.grad.view(-1) for p in params if p.grad is not None]
                )
                self.writer.add_histogram(f"{item}_grad_histogram", gradients, index)
                self.writer.add_scalar(f"{item}_grad_norm2", gradients.norm(2), index)

    def plot(self, res_dict=None, data_index=None, split="train"):
        """
        plot item metrics
        :item: item
        :log: logarithmic scale. Default: False
        :smoothing: smoothing of metric. Default: 0.025
        :ylim: y-axis limits [lower,upper]
        """

        if split == "train":
            res_saving_dir = f"{self.saving_path}/traing_results/{data_index}.vtu"
        else:
            res_saving_dir = f"{self.saving_path}/valid_case/{data_index}.vtu"

        os.makedirs(os.path.dirname(res_saving_dir), exist_ok=True)
        if "cells_node" in res_dict:
            to_vtk.write_to_vtk(res_dict, res_saving_dir)
        else:
            to_vtk.write_point_cloud_to_vtk(res_dict, res_saving_dir)

    def save_state(self, model, optimizer, scheduler, index="final"):
        """
        saves state of model and optimizer
        :model: model to save (if list: save multiple models)
        :optimizer: optimizer (if list: save multiple optimizers)
        :index: index of state to save (e.g. specific epoch)
        """
        os.makedirs(self.saving_path + "/states", exist_ok=True)
        path = self.saving_path + "/states"

        with open(path + "/commandline_args.json", "wt") as f:
            json.dump({**vars(self.params)}, f, indent=4, ensure_ascii=False)

        model.save_checkpoint(path + "/{}.state".format(index), optimizer, scheduler)
        return path + "/{}.state".format(index)

    def save_dict(self, dic, index="final"):
        """
        saves dictionary - helpful to save the population state of an evolutionary optimization algorithm
        :dic: dictionary to store
        :index: index of state to save (e.g. specific evolution)
        """
        os.makedirs(
            "{}/{}/{}/states".format(self.head, self.name, self.datetime), exist_ok=True
        )
        path = "{}/{}/{}/states/{}.dic".format(self.head, self.name, self.datetime, index)
        with open(path, "wb") as f:
            pickle.dump(dic, f)
            
    def save_model_state_subprocess(self, fluid_model, optimizer, lr_scheduler, index):
        """
        子进程中调用的函数，用于调用 logger.save_state
        """
        model_saving_path = self.save_state(
            model=fluid_model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            index=index,
        )
        
        return model_saving_path
    
    def load_state(
        self,
        model,
        optimizer,
        scheduler,
        datetime=None,
        index=None,
        continue_datetime=False,
        device=None,
    ):
        """
        loads state of model and optimizer
        :model: model to load (if list: load multiple models)
        :optimizer: optimizer to load (if list: load multiple optimizers; if None: don't load)
        :datetime: date and time from run to load (if None: take latest folder)
        :index: index of state to load (e.g. specific epoch) (if None: take latest index)
        :continue_datetime: flag whether to continue on this run. Default: False
        :return: datetime, index (helpful, if datetime / index wasn't given)
        """

        if datetime is None:
            for _, dirs, _ in os.walk("{}/{}/".format(self.head, self.name)):
                datetime = sorted(dirs)[-1]
                if datetime == self.datetime:
                    datetime = sorted(dirs)[-2]
                break

        if continue_datetime:
            # CODO: remove generated directories...
            os.rmdir()
            self.datetime = datetime

        if index is None:
            for _, _, files in os.walk(
                "{}/{}/{}/states/".format(self.head, self.name, datetime)
            ):
                index = os.path.splitext(natsorted(files)[-1])[0]
                break

        path = "{}/{}/{}/states/{}.state".format(self.head, self.name, datetime, index)

        model.load_checkpoint(
            optimizer=optimizer, scheduler=scheduler, ckpdir=path, device=device
        )

        return datetime, index

    def load_dict(self, dic, datetime=None, index=None, continue_datetime=False):
        """
        loads state of model and optimizer
        :dic: (empty) dictionary to fill with state information
        :datetime: date and time from run to load (if None: take latest folder)
        :index: index of state to load (e.g. specific epoch) (if None: take latest index)
        :continue_datetime: flag whether to continue on this run. Default: False
        :return: datetime, index (helpful, if datetime / index wasn't given)
        """

        if datetime is None:
            for _, dirs, _ in os.walk("{}/{}/".format(self.head, self.name)):
                datetime = sorted(dirs)[-1]
                if datetime == self.datetime:
                    datetime = sorted(dirs)[-2]
                break

        if continue_datetime:
            # CODO: remove generated directories...
            os.rmdir()
            self.datetime = datetime

        if index is None:
            for _, _, files in os.walk(
                "{}/{}/{}/states/".format(self.head, self.name, datetime)
            ):
                index = os.path.splitext(natsorted(files)[-1])[0]
                break

        path = "{}/{}/{}/states/{}.dic".format(self.head, self.name, datetime, index)
        with open(path, "rb") as f:
            state = pickle.load(f)

        for key in state.keys():
            dic[key] = state[key]

        return datetime, index

    def load_logger(self, datetime=None, load=False, saving_path=None):
        """
        copy older tensorboard logger to new dir
        :datetime: date and time from run to load (if None: take latest folder)
        """

        if datetime is None:
            for _, dirs, _ in os.walk("{}/{}/".format(self.head, self.name)):
                datetime = sorted(dirs)[-1]
                if datetime == self.datetime:
                    datetime = sorted(dirs)[-2]
                break
     
        if load:
            cwd = os.getcwd()
            path = "{}/{0}/{1}/tensorboard/".format(self.head, self.name, datetime)
            for _, _, files in os.walk(path):
                for file in files:
                    older_tensorboard_n = file
                    older_tensorboard = path + older_tensorboard_n

                    newer_tensorboard = (
                        cwd
                        + "/{0}/{1}/{2}/tensorboard/".format(
                            self.head, self.name, self.datetime
                        )
                        + older_tensorboard_n
                    )
                    shutil.copyfile(older_tensorboard, newer_tensorboard)
                break

            if os.path.exists(newer_tensorboard):
                print(
                    "older tensorboard aleady been copied to {0}".format(
                        newer_tensorboard
                    )
                )


t_start = 0


def t_step():
    """
    returns delta t from last call of t_step()
    """
    global t_start
    t_end = time.perf_counter()
    delta_t = t_end - t_start
    t_start = t_end
    return delta_t
