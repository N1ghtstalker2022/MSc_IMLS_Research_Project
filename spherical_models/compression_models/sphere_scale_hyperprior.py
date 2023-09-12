import math
import torch
from spherical_models.compression_models import SphereCompressionModel
from spherical_models import SphereGaussianConditional
from compressai.models.utils import update_registered_buffers
from spherical_models import SLB_Downsample, SLB_Upsample
import numpy as np

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class SphereScaleHyperprior(SphereCompressionModel):
    r"""Scale Hyperprior model for spherical images_equirectangular.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    def __init__(self, N, M, conv_name, skip_conn_aggr, pool_func, unpool_func, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)


        ####################### g_a #######################
        self.g_a = torch.nn.ModuleList()
        self.g_a.append(SLB_Downsample(conv_name, 3, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                        activation="GDN",
                                        pool_func=pool_func, pool_size_sqrt=2))
        self.g_a.append(SLB_Downsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                       activation="GDN",
                                       pool_func=pool_func, pool_size_sqrt=2))
        self.g_a.append(SLB_Downsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                       activation="GDN",
                                       pool_func=pool_func, pool_size_sqrt=2))
        # For the last layer there is no GDN anymore:
        self.g_a.append(SLB_Downsample(conv_name, N, M, hop=2, skip_conn_aggr=skip_conn_aggr,
                                       pool_func=pool_func, pool_size_sqrt=2))

        ####################### g_s #######################
        self.g_s = torch.nn.ModuleList()
        self.g_s.append(SLB_Upsample(conv_name, M, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     activation="GDN", activation_args={"inverse":True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        self.g_s.append(SLB_Upsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     activation="GDN", activation_args={"inverse": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        self.g_s.append(SLB_Upsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     activation="GDN", activation_args={"inverse": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        # For the last layer there is no GDN anymore:
        self.g_s.append(SLB_Upsample(conv_name, N, 3, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     unpool_func=unpool_func, unpool_size_sqrt=2))

        ####################### h_a #######################
        self.h_a = torch.nn.ModuleList()
        # effective hop=1 => num_conv = 1, and there is no downsampling
        self.h_a.append(SLB_Downsample(conv_name, M, N, hop=1, skip_conn_aggr=None,
                                       activation="ReLU", activation_args={"inplace": True}))
        self.h_a.append(SLB_Downsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                       activation="ReLU", activation_args={"inplace": True},
                                       pool_func=pool_func, pool_size_sqrt=2))
        # For the last layer there is no ReLu anymore:
        self.h_a.append(SLB_Downsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                       pool_func=pool_func, pool_size_sqrt=2))

        ####################### h_s #######################
        self.h_s = torch.nn.ModuleList()
        self.h_s.append(SLB_Upsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     activation="ReLU", activation_args={"inplace": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        self.h_s.append(SLB_Upsample(conv_name, N, N, hop=2, skip_conn_aggr=skip_conn_aggr,
                                     activation="ReLU", activation_args={"inplace": True},
                                     unpool_func=unpool_func, unpool_size_sqrt=2))
        # effective hop=1 => num_conv = 1, and there is no Upsampling
        self.h_s.append(SLB_Upsample(conv_name, N, M, hop=1, skip_conn_aggr=None,
                                     activation="ReLU", activation_args={"inplace": True}))

        ###################################################
        self.gaussian_conditional = SphereGaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self._computeResOffset()


    def _computeResOffset(self):
        # compute convolution resolution offset
        g_a_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.g_a]))
        self._g_a_offset = [self.g_a[0].get_conv_input_res_offset()]
        self._g_a_offset.extend([self.g_a[i].get_conv_input_res_offset()+g_a_output[i-1] for i in range(1, len(self.g_a))])

        h_a_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.h_a]))
        h_a_output = [res+g_a_output[-1] for res in h_a_output]
        self._h_a_offset = [self.h_a[0].get_conv_input_res_offset() + g_a_output[-1]]
        self._h_a_offset.extend([self.h_a[i].get_conv_input_res_offset()+h_a_output[i-1] for i in range(1, len(self.h_a))])

        h_s_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.h_s]))
        h_s_output = [res+h_a_output[-1] for res in h_s_output]
        self._h_s_offset = [self.h_s[0].get_conv_input_res_offset()+h_a_output[-1]]
        self._h_s_offset.extend([self.h_s[i].get_conv_input_res_offset() + h_s_output[i - 1] for i in range(1, len(self.h_s))])

        assert h_s_output[-1] == g_a_output[-1], "resolutions do not match"

        g_s_output = list(np.cumsum([layerBlock.get_output_res_offset() for layerBlock in self.g_s]))
        g_s_output = [res + g_a_output[-1] for res in g_s_output]
        self._g_s_offset = [self.g_s[0].get_conv_input_res_offset() + g_a_output[-1]]
        self._g_s_offset.extend([self.g_s[i].get_conv_input_res_offset() + g_s_output[i - 1] for i in range(1, len(self.g_s))])


    def get_resOffset(self):
        return set(self._g_a_offset + self._h_a_offset + self._h_s_offset + self._g_s_offset)

    def forward(self, x, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):     # x is a tensor of size [batch_size, num_nodes, num_features]

        data_res = res if patch_res is None else (res, patch_res)
        ########### apply g_a ###########
        y = x
        for i in range(len(self.g_a)):
            # why offset?
            conv_res = type(data_res)(np.add(data_res, self._g_a_offset[i]))
            y = self.g_a[i](y, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
        # print("applying g_a")
        # print("y.mean()=", y.mean(), "x.mean()=", x.mean())
        # print("y.max()=", y.max(), "y.min()=", y.min())
        # print("x.max()=", x.max(), "x.min()=", x.min())
        ########### apply h_a ###########
        z = torch.abs(y)
        for i in range(len(self.h_a)):
            conv_res = type(data_res)(np.add(data_res, self._h_a_offset[i]))
            z = self.h_a[i](z, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
        # print("applying h_a")
        # print("z.mean()=", z.mean(), "torch.abs(y).mean()=", torch.abs(y).mean())
        # print("z.max()=", z.max(), "z.min()=", z.min())
        # print("torch.abs(y).max()=", torch.abs(y).max(), "torch.abs(y).min()=", torch.abs(y).min())
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        ########### apply h_s ###########
        scales_hat = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            scales_hat = self.h_s[i](scales_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        # print("applying h_s")
        # print("scales_hat.mean()=", scales_hat.mean(), "z_hat.mean()=", z_hat.mean())
        # print("scales_hat.max()=", scales_hat.max(), "scales_hat.min()=", scales_hat.min())
        # print("z_hat.max()=", z_hat.max(), "z_hat.min()=", z_hat.min())
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)

        ########### apply g_s ###########
        x_hat = y_hat
        for i in range(len(self.g_s)):
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[i]))
            x_hat = self.g_s[i](x_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

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
                'z': z_likelihoods
            },
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict)
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):

        data_res = res if patch_res is None else (res, patch_res)
        ########### apply g_a ###########
        y = x
        for i in range(len(self.g_a)):
            conv_res = type(data_res)(np.add(data_res, self._g_a_offset[i]))
            y = self.g_a[i](y, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        ########### apply h_a ###########
        z = torch.abs(y)
        for i in range(len(self.h_a)):
            conv_res = type(data_res)(np.add(data_res, self._h_a_offset[i]))
            z = self.h_a[i](z, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[1])

        ########### apply h_s ###########
        scales_hat = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            scales_hat = self.h_s[i](scales_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[1]}

    def decompress(self, strings, shape, dict_index, dict_weight, res, patch_res=None, dict_valid_index=None):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        data_res = res if patch_res is None else (res, patch_res)
        ########### apply h_s ###########
        scales_hat = z_hat
        for i in range(len(self.h_s)):
            conv_res = type(data_res)(np.add(data_res, self._h_s_offset[i]))
            scales_hat = self.h_s[i](scales_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)

        ########### apply g_s ###########
        x_hat = y_hat
        for i in range(len(self.g_s)):
            conv_res = type(data_res)(np.add(data_res, self._g_s_offset[i]))
            x_hat = self.g_s[i](x_hat, dict_index[conv_res], dict_weight[conv_res], valid_index=dict_valid_index[conv_res] if dict_valid_index is not None else None)
        x_hat = x_hat.clamp_(0, 1)
        return {"x_hat": x_hat}



if __name__ == '__main__':
    ssh = SphereScaleHyperprior(128, 192, "SDPAConv", "sum", "max_pool", "nearest")
    print(ssh.get_resOffset())
