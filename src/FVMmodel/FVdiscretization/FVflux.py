import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import torch
import torch.nn as nn
from Utils.utilities import (
    decompose_and_trans_node_attr_to_cell_attr_graph,
    copy_geometric_data,
    NodeType,
    calc_cell_centered_with_node_attr,
    calc_node_centered_with_cell_attr,
)
from torch_geometric.data import Data
import numpy as np
from torch_scatter import scatter_add, scatter_mean
import matplotlib.pyplot as plt
from FVMmodel.FVdiscretization.FVInterpolation import Interplot

class FV_flux(Interplot):
    def __init__(self):
        """
        FV_flux class for finite volume flux calculation.
        Inherits from Interplot. Used as a base for flux-related operations.
        """
        super(FV_flux, self).__init__()
        self.plotted = False
