import torch
import torch_geometric
from spherical_models import SDPAConv, SphereSkipConnection, SpherePixelShuffle, SphereGDN
from projects.deep_video_compression.utils import common_function as util_common
from projects.deep_video_compression.utils import healpix as hp_utils


class SLB_Downsample(torch.nn.Module):
    r"""Spherical Layer Block for Downsampling consists of:
     one or several convolutions (with desired aggregation of conv outputs) +
     optional non-linearity on the output of conv +
     optional down sampling

    Args:

    """

    def __init__(self,
                 conv_name,
                 in_channels,
                 out_channels,
                 bias=True,
                 hop=1,
                 skip_conn_aggr=None,
                 activation=None,
                 activation_args=dict(),
                 pool_func=None,
                 pool_size_sqrt=1):
        super().__init__()

        # 1- Setting convolution
        self.node_dim = 1
        self.list_conv = torch.nn.ModuleList()
        num_conv = hop if conv_name in ["GraphConv", "SDPAConv"] else 1
        if skip_conn_aggr == 'cat':
            out_channels //= num_conv

        if conv_name == "ChebConv":
            conv = getattr(torch_geometric.nn, conv_name)
            self.list_conv.append(
                conv(in_channels=in_channels, out_channels=out_channels, K=hop + 1, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['TAGConv',
                           'SGConv']:  # the graph convolutions in torch_geometric which need number of hop as input
            conv = getattr(torch_geometric.nn, conv_name)
            self.list_conv.append(
                conv(in_channels=in_channels, out_channels=out_channels, K=hop, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['GraphConv']:  # These convolutions don't accept number of hops as input
            conv = getattr(torch_geometric.nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, aggr='mean', bias=bias,
                                       node_dim=self.node_dim))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels,
                                                            aggr='mean', bias=bias, node_dim=self.node_dim) for _ in
                                                       range(
                                                           num_conv - 1)]))  # Maybe later not all of them has aggr as argument
        elif conv_name == "SDPAConv":
            conv = SDPAConv
            n_firstHopNeighbors = 8
            n_neighbors = util_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=1)
            self.list_conv.append(
                conv(in_channels=in_channels, out_channels=out_channels, kernel_size=n_neighbors + 1, bias=bias,
                     node_dim=self.node_dim))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels,
                                                            kernel_size=n_neighbors + 1, bias=bias,
                                                            node_dim=self.node_dim) for _ in range(num_conv - 1)]))
        else:
            raise ValueError('Convolution is not defined')

        self.in_channels = in_channels
        assert len(self.list_conv) == num_conv, "list conv must be equal to num_conv"
        # Setting aggregation of convolution results
        self.out_channels = out_channels
        if num_conv > 1 and skip_conn_aggr not in ["non", "none"]:
            self.skipconn = SphereSkipConnection(skip_conn_aggr)
            if skip_conn_aggr == "cat":
                self.out_channels *= num_conv
        else:
            self.register_parameter('skipconn', None)

        self.conv_out_channels = self.out_channels
        # 2- Setting nonlinearity
        if activation is None:
            self.register_parameter('activation', None)
        elif activation in ["GDN"]:
            self.activation = SphereGDN(self.out_channels, **activation_args)
        else:
            self.activation = getattr(torch.nn, activation)(**activation_args)

        # 3- Setting Downsampling
        self.pool_size_sqrt = pool_size_sqrt
        self.pool_size = pool_size_sqrt * pool_size_sqrt
        assert (self.pool_size == 1 and pool_func is None) or (
                self.pool_size > 1 and pool_func is not None), "pool_func and pool_size must match."
        if pool_func is None or self.pool_size == 1:
            self.register_parameter('pool', None)
        elif pool_func == 'max_pool':
            self.pool = getattr(torch.nn, "MaxPool3d")(kernel_size=(1, self.pool_size, 1))
        elif pool_func == "avg_pool":
            self.pool = getattr(torch.nn, "AvgPool3d")(kernel_size=(1, self.pool_size, 1))
        elif pool_func == "stride":
            self.pool = "stride"
        else:
            raise ValueError('Pooling is not defined')

    def forward(self, x, index, weight, valid_index=None,
                mapping=None):  # x is a tensor of size [batch_size, num_nodes, num_features]
        device = x.device
        index = index.to(device)
        weight = weight.to(device)
        valid_index = valid_index.to(device) if valid_index is not None else None
        xs = []
        for conv in self.list_conv:
            if conv.__class__.__name__ == "SDPAConv":
                x = conv(x, neighbors_indices=index, neighbors_weights=weight, valid_index=valid_index)
            else:
                x = conv(x, edge_index=index, edge_weight=weight)
            xs += [x] if self.pool != "stride" else [x.index_select(self.node_dim,
                                                                    torch.arange(0, x.size(self.node_dim),
                                                                                 step=self.pool_size, device=x.device))]
        x = self.skipconn(xs) if self.skipconn is not None else xs[-1]

        if mapping is not None:
            mapping = mapping.to(device)
            x = x.index_select(self.node_dim, mapping)

        if self.activation is not None:
            x = self.activation(x)

        if self.pool not in [None, "stride"]:

            x = torch.squeeze(self.pool(torch.unsqueeze(x, dim=0)), dim=0)

        return x

    def get_conv_input_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct images_equirectangular for the "input of the conv".

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        return 0

    def get_output_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct images_equirectangular for the "output of the module".

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        if self.pool is None:
            return 0

        # Otherwise the unpooling is Upsampling
        return hp_utils.healpix_getResolutionDownsampled(0, self.pool_size_sqrt)


class SLB_Upsample(torch.nn.Module):
    r"""Spherical Layer Block for Upsampling sists of:
     one or several convolutions (with desired aggregation of conv outputs) +
     optional non-linearity on the output of conv +
     optional up-sampling

    Args:

    """

    def __init__(self,
                 conv_name,
                 in_channels,
                 out_channels,
                 bias=True,
                 hop=1,
                 skip_conn_aggr=None,
                 activation=None,
                 activation_args=dict(),
                 unpool_func=None,
                 unpool_size_sqrt=1):
        super().__init__()

        self.node_dim = 1

        # 1- Setting up upsampling
        self.unpool_size_sqrt = unpool_size_sqrt
        self.unpool_size = unpool_size_sqrt * unpool_size_sqrt
        assert (self.unpool_size == 1 and unpool_func is None) or (
                self.unpool_size > 1 and unpool_func is not None), "unpool_func and unpool_size must match."
        if unpool_func is None or self.unpool_size == 1:
            self.register_parameter('unpool', None)
        elif unpool_func in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            self.unpool = getattr(torch.nn, "Upsample")(scale_factor=(self.unpool_size, 1), mode=unpool_func)
        elif unpool_func == "pixel_shuffle":
            self.unpool = SpherePixelShuffle(self.unpool_size_sqrt, self.node_dim)
            out_channels *= self.unpool_size
        else:
            raise ValueError('Unpooling is not defined')

        # 2- Setting convolution
        self.list_conv = torch.nn.ModuleList()
        num_conv = hop if conv_name in ["GraphConv", "SDPAConv"] else 1
        if skip_conn_aggr == 'cat':
            out_channels //= num_conv

        if conv_name == "ChebConv":
            conv = getattr(torch_geometric.nn, conv_name)
            self.list_conv.append(
                conv(in_channels=in_channels, out_channels=out_channels, K=hop + 1, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['TAGConv',
                           'SGConv']:  # the graph convolutions in torch_geometric which need number of hop as input
            conv = getattr(torch_geometric.nn, conv_name)
            self.list_conv.append(
                conv(in_channels=in_channels, out_channels=out_channels, K=hop, bias=bias, node_dim=self.node_dim))
        elif conv_name in ['GraphConv']:  # These convolutions don't accept number of hops as input
            conv = getattr(torch_geometric.nn, conv_name)
            self.list_conv.append(conv(in_channels=in_channels, out_channels=out_channels, aggr='mean', bias=bias,
                                       node_dim=self.node_dim))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels,
                                                            aggr='mean', bias=bias, node_dim=self.node_dim) for _ in
                                                       range(
                                                           num_conv - 1)]))  # Maybe later not all of them has aggr as argument
        elif conv_name == "SDPAConv":
            conv = SDPAConv
            n_firstHopNeighbors = 8
            n_neighbors = util_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=1)
            self.list_conv.append(
                conv(in_channels=in_channels, out_channels=out_channels, kernel_size=n_neighbors + 1, bias=bias,
                     node_dim=self.node_dim))
            self.list_conv.extend(torch.nn.ModuleList([conv(in_channels=out_channels, out_channels=out_channels,
                                                            kernel_size=n_neighbors + 1, bias=bias,
                                                            node_dim=self.node_dim) for _ in range(num_conv - 1)]))
        else:
            raise ValueError('Convolution is not defined')

        self.in_channels = in_channels
        assert len(self.list_conv) == num_conv, "list conv must be equal to num_conv"
        # Setting aggregation of convolution results
        self.out_channels = out_channels
        if num_conv > 1 and skip_conn_aggr not in ["non", "none"]:
            self.skipconn = SphereSkipConnection(skip_conn_aggr)
            if skip_conn_aggr == "cat":
                self.out_channels *= num_conv
        else:
            self.register_parameter('skipconn', None)
        self.conv_out_channels = self.out_channels
        if unpool_func == "pixel_shuffle":
            self.out_channels //= self.unpool_size

        # 3- Setting nonlinearity
        if activation is None:
            self.register_parameter('activation', None)
        elif activation in ["GDN"]:
            self.activation = SphereGDN(self.out_channels, **activation_args)
        else:
            self.activation = getattr(torch.nn, activation)(**activation_args)

    def forward(self, x, index, weight, valid_index=None,
                mapping=None):  # x is a tensor of size [batch_size, num_nodes, num_features]
        device = x.device

        if mapping is not None:
            raise NotImplementedError("Not implemented")

        # Note for unpooling:
        # if unpooling is Upsample the order is: Upsample then Convolution
        # if unpooling is SpherePixelShuffle the order is: Convolution then SpherePixelShuffle
        if self.unpool is not None and self.unpool.__class__.__name__ == "Upsample":
            x = torch.squeeze(self.unpool(torch.unsqueeze(x, dim=0)), dim=0)

        index = index.to(device)
        weight = weight.to(device)
        valid_index = valid_index.to(device) if valid_index is not None else None
        xs = []
        for conv in self.list_conv:
            if conv.__class__.__name__ == "SDPAConv":
                x = conv(x, neighbors_indices=index, neighbors_weights=weight, valid_index=valid_index)
            else:
                x = conv(x, edge_index=index, edge_weight=weight)
            xs += [x]
        x = self.skipconn(xs) if self.skipconn is not None else xs[-1]

        if self.unpool is not None and self.unpool.__class__.__name__ == "SpherePixelShuffle":
            x = self.unpool(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def get_conv_input_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct images_equirectangular for the "input of the conv".
        For example, if we use Upsampling, since first the upsampling is applied and then convolution, for unpool_size_sqrt=2
        it returns 1 because conv is appliad on upsampled images_equirectangular.
        For pixel shuffling, since pixel shuffling is applied after convolution, the function return 0 no matter of unpool_size_sqrt

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        if self.unpool is None:
            return 0

        # There is an unpooling
        if self.unpool.__class__.__name__ == "SpherePixelShuffle":
            return 0

        # Otherwise the unpooling is Upsampling
        return hp_utils.healpix_getResolutionUpsampled(0, self.unpool_size_sqrt)

    def get_output_res_offset(self):
        r"""
        Show the offset of the healpix resolution of struct images_equirectangular for the "output of the module".

        Returns
        -------
        Integer that shows the offset resolution for the convolution of
        """
        if self.unpool is None:
            return 0

        # Otherwise the unpooling is Upsampling
        return hp_utils.healpix_getResolutionUpsampled(0, self.unpool_size_sqrt)


if __name__ == '__main__':
    import healpy as hp
    import healpix_graph_loader
    import healpix_sdpa_struct_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resolution = 2
    patch_resolution = 2
    patch_id = 5
    nside = hp.order2nside(resolution)  # == 2 ** sampling_resolution
    nPix = hp.nside2npix(nside)
    use_geodesic = True

    folder = "../GraphData"
    cutGraphForPatchOutside = True
    weight_type = "gaussian"
    K = 1  # Number of hops

    conv_name = "SDPAConv"  # SDPAConv, 'ChebConv', 'TAGConv', 'SGConv', GraphConv
    unpool_func = "nearest"  # 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', pixel_shuffle
    scale_factor = 2

    if conv_name == "SDPAConv":
        loader = healpix_sdpa_struct_loader.HealpixSdpaStructLoader(weight_type=weight_type,
                                                                    use_geodesic=use_geodesic,
                                                                    use_4connectivity=False,
                                                                    normalization_method="sym",
                                                                    cutGraphForPatchOutside=cutGraphForPatchOutside,
                                                                    load_save_folder=folder)
        struct_data = loader.getStruct(resolution, K, patch_resolution, patch_id)
        # struct_sdpa = sdpa_loader.getStruct(resolution, K)
        index_downsample = struct_data[0]
        weight_downsample = struct_data[1]
        nodes = struct_data[3]
        if unpool_func == "pixel_shuffle":
            index_upsample = index_downsample
            weight_upsample = weight_downsample
        else:
            struct_data = loader.getStruct(hp_utils.healpix_getResolutionUpsampled(resolution, scale_factor), K,
                                           hp_utils.healpix_getResolutionUpsampled(patch_resolution, scale_factor),
                                           patch_id)
            # struct_graph = graph_loader.getGraph(sampling_res=resolution)
            index_upsample = struct_data[0]
            weight_upsample = struct_data[1]
    else:
        loader = healpix_graph_loader.HealpixGraphLoader(weight_type=weight_type,
                                                         use_geodesic=use_geodesic,
                                                         use_4connectivity=False,
                                                         load_save_folder=folder)

        n_hop_graph = 0 if cutGraphForPatchOutside else K
        struct_data = loader.getGraph(sampling_res=resolution, patch_res=patch_resolution, num_hops=n_hop_graph,
                                      patch_id=patch_id)
        # struct_graph = graph_loader.getGraph(sampling_res=resolution)
        index_downsample = struct_data[0]
        weight_downsample = struct_data[1]
        nodes = struct_data[2]
        if unpool_func == "pixel_shuffle":
            index_upsample = index_downsample
            weight_upsample = weight_downsample
        else:
            struct_data = loader.getGraph(
                sampling_res=hp_utils.healpix_getResolutionUpsampled(resolution, scale_factor),
                patch_res=hp_utils.healpix_getResolutionUpsampled(patch_resolution, scale_factor),
                num_hops=n_hop_graph, patch_id=patch_id)
            # struct_graph = graph_loader.getGraph(sampling_res=resolution)
            index_upsample = struct_data[0]
            weight_upsample = struct_data[1]

    B = 4  # batch size
    in_channels = 2
    out_channels = 10
    data_th = torch.randn(B, nPix, in_channels)
    data_th = data_th.index_select(dim=1, index=nodes)

    print("data_th.size()=", data_th.size())

    slb_down = SLB_Downsample(conv_name, in_channels, out_channels,
                              bias=True, hop=2,
                              skip_conn_aggr="sum",
                              activation="GDN",
                              pool_func="max_pool", pool_size_sqrt=scale_factor
                              )

    slb_down2 = SLB_Downsample(conv_name, out_channels, out_channels,
                               bias=True, hop=2,
                               skip_conn_aggr="sum",
                               activation="GDN",
                               pool_func="max_pool", pool_size_sqrt=scale_factor
                               )

    print(slb_down)
    out_down = slb_down(data_th, index_downsample, weight_downsample)
    out_down = slb_down2(out_down, index_downsample, weight_downsample)
    print("out_down.size()=", out_down.size())

    # TODO: Check the same for SLB_Upsample
    slb_up = SLB_Upsample(conv_name, in_channels, out_channels,
                          bias=True, hop=2,
                          skip_conn_aggr="sum",
                          activation="GDN", activation_args={"inverse": True},
                          unpool_func=unpool_func, unpool_size_sqrt=scale_factor
                          )

    print(slb_up)
    out_up = slb_up(data_th, index_upsample, weight_upsample)
    print("out_up.size()=", out_up.size())
