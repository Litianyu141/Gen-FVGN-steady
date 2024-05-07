from torch import tensor, zeros, ones, sum, max, sqrt, float
from torch.nn import Module, Parameter

# class Normalizer(nn.Module):
#     def __init__(self, size, max_accumulations=12e+4, std_epsilon=1e-8, affine=False,name='Normalizer', device='cuda'):
#         super(Normalizer, self).__init__()
#         self.device=device
#         self.name=name
#         self.setted = False
#         self.affine = affine
#         self._max_accumulations = self.register_buffer('_max_accumulations', torch.as_tensor(max_accumulations).to(device))
#         self._std_epsilon = self.register_buffer('_std_epsilon', torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device))
#         self._acc_count = self.register_buffer('_acc_count', torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device))
#         self._num_accumulations = self.register_buffer('_num_accumulations', torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device))
#         self._acc_sum = self.register_buffer('_acc_sum', torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))
#         self._acc_sum_squared = self.register_buffer('_acc_sum_squared', torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))
#         self._var = self.register_buffer('_var', torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))
#         self._mean_value = self.register_buffer('_mean_value', torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))


#         self._max_accumulations = torch.as_tensor(max_accumulations).to(device)
#         self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
#         self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
#         self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
#         self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
#         self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
#         self._var = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
#         self._mean_value = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)

#     def forward(self, batched_data, accumulate=True):
#         """Normalizes input data and accumulates statistics, and face_length has been normalized by torch`s BN"""
#         if self.name == '_cell_area_normalizer':
#             return torch.sigmoid(batched_data)
#         else:
#             if self.training:
#                 if accumulate:
#                 # stop accumulating after a million updates, to prevent accuracy issues
#                     if self._num_accumulations < self._max_accumulations:
#                         self._accumulate(batched_data.detach())
#                     return ((batched_data - self._mean()) / torch.sqrt(self._var_with_epsilon()))
#                 else:
#                     return ((batched_data - self._mean(batched_data)) / torch.sqrt(self._var_with_epsilon(batched_data)))
#             else:
#                 return ((batched_data - self._mean()) / torch.sqrt(self._var_with_epsilon()))

#     def inverse(self, normalized_batch_data):
#         """Inverse transformation of the normalizer."""
#         return ((normalized_batch_data * torch.sqrt(self._var_with_epsilon())) + self._mean())

#     def set_num_accumulations(self, num_accumulations):
#         """set num of accumulations."""
#         if not self.setted:
#             self._num_accumulations = num_accumulations
#             self.setted = True

#     def _accumulate(self, batched_data):
#         """Function to perform the accumulation of the batch_data statistics."""
#         count = batched_data.shape[0]
#         data_sum = torch.sum(batched_data, axis=0, keepdims=True)
#         squared_data_sum = torch.sum(batched_data**2, axis=0, keepdims=True)

#         self._acc_sum += data_sum
#         self._acc_sum_squared += squared_data_sum
#         self._acc_count += count
#         self._num_accumulations += 1
#         '''momentum way to stastic variance'''
#         self._var = 0.95*self._var+0.05*self._var_with_epsilon(batched_data)
#         self._mean_value = 0.95*self._mean_value+0.05*self._mean(batched_data)

#     def _mean(self,batched_data=None):
#         if batched_data is not None:
#             safe_count = batched_data.shape[0]
#             return torch.sum(batched_data, axis=0, keepdims=True)/safe_count
#         else:
#             safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
#             return self._acc_sum / safe_count

#     def _var_with_epsilon(self,batched_data=None):
#         # if batched_data is not None:

#         #     safe_count = batched_data.shape[0]
#         #     var = torch.sum(batched_data**2, axis=0, keepdims=True) / safe_count - torch.mean(batched_data,dim=0)**2
#         #     '''we use x-dir velocity var and pressure var as the total Pressure variance'''
#         #     # if self.name =='_face_uvp_flux_output_normalizer':
#         #     #     var[:,2] = var[:,0]+var[:,2]
#         #     return torch.maximum(var, self._std_epsilon)
#         # else:
#         #     safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
#         #     var = self._acc_sum_squared / safe_count - self._mean()**2
#         #     '''we use x-dir velocity var and pressure var as the total Pressure variance'''
#         #     # if self.name =='_face_uvp_flux_output_normalizer':
#         #     #     var[:,2] = var[:,0]+var[:,2]
#             var = torch.ones_like(self._std_epsilon)
#             return torch.maximum(var, self._std_epsilon)

#     def get_variable(self):

#         dict = {'_max_accumulations':self._max_accumulations,
#         '_std_epsilon':self._std_epsilon,
#         '_acc_count': self._acc_count,
#         '_num_accumulations':self._num_accumulations,
#         '_acc_sum': self._acc_sum,
#         '_acc_sum_squared':self._acc_sum_squared,
#         '_var':self._var,
#         '_mean_value':self._mean_value,
#         'name':self.name,
#         'beta':self.beta
#         }

#         return dict


class Normalizer(Module):
    def __init__(self, size, max_accumulations=10**7, epsilon=1e-8, device=None):
        """
        Online normalization module

        size: feature dimension
        max_accumulation: maximum number of batches
        epsilon: std cutoff for constant variable
        device: pytorch device
        """

        super(Normalizer, self).__init__()

        self.max_accumulations = max_accumulations
        self.epsilon = epsilon

        # self.register_buffer('acc_count', tensor(0, dtype=float, device=device))
        # self.register_buffer('num_accumulations', tensor(0, dtype=float, device=device))
        # self.register_buffer('acc_sum', zeros(size, dtype=float, device=device))
        # self.register_buffer('acc_sum_squared', zeros(size, dtype=float, device=device))

        self.register_buffer("acc_count", tensor(1.0, dtype=float, device=device))
        self.register_buffer(
            "num_accumulations", tensor(1.0, dtype=float, device=device)
        )
        self.register_buffer("acc_sum", zeros(size, dtype=float, device=device))
        self.register_buffer("acc_sum_squared", zeros(size, dtype=float, device=device))

    def forward(self, batched_data, accumulate=True):
        """
        Updates mean/standard deviation and normalizes input data

        batched_data: batch of data
        accumulate: if True, update accumulation statistics
        """
        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data)

        return (batched_data - self._mean().to(batched_data.device)) / self._std().to(
            batched_data.device
        )

    def inverse(self, normalized_batch_data):
        """
        Unnormalizes input data
        """

        return normalized_batch_data * self._std().to(
            normalized_batch_data.device
        ) + self._mean().to(normalized_batch_data.device)

    def _accumulate(self, batched_data):
        """
        Accumulates statistics for mean/standard deviation computation
        """
        count = tensor(batched_data.shape[0]).float()
        data_sum = sum(batched_data, dim=0)
        squared_data_sum = sum(batched_data**2, dim=0)

        self.acc_sum += data_sum.to(self.acc_sum.device)
        self.acc_sum_squared += squared_data_sum.to(self.acc_sum_squared.device)
        self.acc_count += count.to(self.acc_count.device)
        self.num_accumulations += 1

    def _mean(self):
        """
        Returns accumulated mean
        """
        safe_count = max(self.acc_count, tensor(1.0).float())

        return self.acc_sum / safe_count

    def _std(self):
        """
        Returns accumulated standard deviation
        """
        safe_count = max(self.acc_count, tensor(1.0).float())
        std = sqrt(self.acc_sum_squared / safe_count - self._mean() ** 2)

        std[std < self.epsilon] = 1.0

        return std
