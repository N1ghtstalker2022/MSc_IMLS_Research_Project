import torch
import math
import torch_geometric

class SDPAConv (torch.nn.Module):
    r"""Class for implementing Sphere Directional and Position-Aware convolution
    """
    def __init__(self,  in_channels, out_channels, kernel_size, node_dim=0, bias=True):
        super(SDPAConv, self).__init__()

        assert node_dim >= 0
        self.node_dim = node_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = torch.nn.Parameter(torch.Tensor(kernel_size, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        # Took it from torch.nn.Conv2d()

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # torch.nn.init.xavier_uniform_(self.weight, gain=2.)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        # Took it from torch_geometric.nn.ChebConv
        # torch_geometric.nn.inits.glorot(self.weight)
        # torch_geometric.nn.inits.zeros(self.bias)


    def forward(self, x, neighbors_indices, neighbors_weights, valid_index=None):
        assert (self.kernel_size-1)==neighbors_weights.size(1), "size does not match"

        out = torch.matmul(x, self.weight[0])

        # test_out = torch.zeros(x.size(), dtype=x.dtype)
        # for k in range(neighbors_weights.size(1)):
        #     test_out += torch.mul(neighbors_weights.narrow(dim=1, start=k, length=1), x.index_select(self.node_dim, neighbors_indices[:, k]))
        # print("test_out finished")

        for k in range(1, self.kernel_size):
            col = k-1
            if valid_index is None:
                s = torch.mul(neighbors_weights.narrow(dim=1, start=col, length=1), x.index_select(self.node_dim, neighbors_indices[:, col]))   # or I could use neighbors_weights[:,col].view(-1, 1)
                out += torch.matmul(s, self.weight[k])
            else:
                valid_rows = valid_index[:, col]
                s = torch.mul(neighbors_weights[valid_rows, col].view(-1, 1), x.index_select(self.node_dim, neighbors_indices[valid_rows, col]))
                out[:, valid_rows, :] += torch.matmul(s, self.weight[k])


        if self.bias is not None:
            out += self.bias

        return out

