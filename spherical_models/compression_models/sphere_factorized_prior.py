import math
import torch
from spherical_models.compression_models import SphereCompressionModel
from spherical_models import SphereGaussianConditional
from compressai.models.utils import update_registered_buffers
from spherical_models import SLB_Downsample, SLB_Upsample
import numpy as np


class SphereFactorizedPrior(SphereCompressionModel):
    r"""Scale Hyperprior model for spherical images_equirectangular.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, conv_name='SDPAConv', skip_conn_aggr='none', pool_func='avg_pool',
                 unpool_func='pixel_shuffle', **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        ####################### g_a #######################
        self.g_a = torch.nn.ModuleList()
        self.g_a.append(SLB_Downsample(conv_name, 3, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                       activation="GDN",
                                       pool_func=pool_func, pool_size_sqrt=2))
        # self.g_a.append(SLB_Downsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
        #                                activation="GDN",
        #                                pool_func=pool_func, pool_size_sqrt=2))
        # self.g_a.append(SLB_Downsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
        #                                activation="GDN",
        #                                pool_func=pool_func, pool_size_sqrt=2))
        # For the last layer there is no GDN anymore:
        self.g_a.append(SLB_Downsample(conv_name, N, M, hop=2, skip_conn_aggr=skip_conn_aggr,
                                       pool_func=pool_func, pool_size_sqrt=2))

        ####################### g_s #######################
        self.g_s = torch.nn.ModuleList()
        self.g_s.append(SLB_Upsample(conv_name, M, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     activation="GDN", activation_args={"inverse": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        # self.g_s.append(SLB_Upsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
        #                              activation="GDN", activation_args={"inverse": True},
        #                              unpool_func=unpool_func, unpool_size_sqrt=2))
        # self.g_s.append(SLB_Upsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
        #                              activation="GDN", activation_args={"inverse": True},
        #                              unpool_func=unpool_func, unpool_size_sqrt=2))
        # For the last layer there is no GDN anymore:
        self.g_s.append(SLB_Upsample(conv_name, N, 3, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     unpool_func=unpool_func, unpool_size_sqrt=2))

        ###################################################
        self.N = int(N)
        self.M = int(M)

        self._computeResOffset()

    def _computeResOffset(self):
        # compute convolution resolution offset
        g_a_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.g_a]))
        self._g_a_offset = [self.g_a[0].get_conv_input_res_offset()]
        self._g_a_offset.extend(
            [self.g_a[i].get_conv_input_res_offset() + g_a_output[i - 1] for i in range(1, len(self.g_a))])

        g_s_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.g_s]))
        g_s_output = [res + g_a_output[-1] for res in g_s_output]
        self._g_s_offset = [self.g_s[0].get_conv_input_res_offset() + g_a_output[-1]]
        self._g_s_offset.extend(
            [self.g_s[i].get_conv_input_res_offset() + g_s_output[i - 1] for i in range(1, len(self.g_s))])

    def get_resOffset(self):
        a = set(self._g_a_offset + self._g_s_offset)
        return set(self._g_a_offset + self._g_s_offset)

    def forward(self, x, dict_index, dict_weight, res, patch_res=None,
                dict_valid_index=None):  # x is a tensor of size [batch_size, num_nodes, num_features]

        data_res = res if patch_res is None else (res, patch_res)
        ########### apply g_a ###########
        y = x
        for i in range(len(self.g_a)):
            conv_res = type(data_res)(np.add(data_res, self._g_a_offset[i]))
            y = self.g_a[i](y, dict_index[conv_res], dict_weight[conv_res],
                            valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
        # print("applying g_a")
        # print("y.mean()=", y.mean(), "x.mean()=", x.mean())
        # print("y.max()=", y.max(), "y.min()=", y.min())
        # print("x.max()=", x.max(), "x.min()=", x.min())
        y_hat, y_likelihoods = self.entropy_bottleneck(y)

        ########### apply g_s ###########
        x_hat = y_hat
        for i in range(len(self.g_s)):
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[i]))
            x_hat = self.g_s[i](x_hat, dict_index[conv_res], dict_weight[conv_res],
                                valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        # print("applying g_s")
        # print("x_hat.mean()=", x_hat.mean(), "y_hat.mean()=", y_hat.mean())
        # print("x_hat.max()=", x_hat.max(), "x_hat.min()=", x_hat.min())
        # print("y_hat.max()=", y_hat.max(), "y_hat.min()=", y_hat.min())
        # with torch.no_grad():
        #     print("input/out mean ratio=", x.mean()/x_hat.mean())

        return {
            'x_hat': x_hat,
            'likelihoods': {
                'y': y_likelihoods,
            },
        }

    def compress(self, x, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):

        data_res = res if patch_res is None else (res, patch_res)
        ########### apply g_a ###########
        y = x
        for i in range(len(self.g_a)):
            conv_res = type(data_res)(np.add(data_res, self._g_a_offset[i]))
            y = self.g_a[i](y, dict_index[conv_res], dict_weight[conv_res],
                            valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[1]}

    def decompress(self, strings, shape, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)

        data_res = res if patch_res is None else (res, patch_res)

        ########### apply g_s ###########
        x_hat = y_hat
        for i in range(len(self.g_s)):
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[i]))
            x_hat = self.g_s[i](x_hat, dict_index[conv_res], dict_weight[conv_res],
                                valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
        x_hat = x_hat.clamp_(0, 1)
        return {"x_hat": x_hat}


if __name__ == '__main__':
    ssh = SphereFactorizedPrior(64, 128, "SDPAConv", "sum", "max_pool", "nearest")
    print(ssh.get_resOffset())
