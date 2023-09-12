import os
import torch
import numpy as np
import healpy as hp
from utils import healpix as hp_utils


class HealpixSdpaStructLoader:
    def __init__(self, weight_type, use_geodesic, use_4connectivity, normalization_method, cutGraphForPatchOutside,
                 load_save_folder=None):

        self.weight_type = weight_type
        self.use_geodesic = use_geodesic
        self.use_4connectivity = use_4connectivity
        self.isNest = True
        self.folder = load_save_folder
        self.normalization_method = normalization_method
        self.cutGraph = cutGraphForPatchOutside
        if self.folder:
            os.makedirs(self.folder, exist_ok=True)

    def getStruct(self, sampling_res, num_hops, patch_res=None, patch_id=None):

        if (num_hops is None) or (num_hops <= 0):
            num_hops = 1

        if self.folder:
            filename = "sdpa_{}_{}_{}_{}_{}_{}".format(self.weight_type, self.normalization_method, self.use_geodesic,
                                                       self.use_4connectivity, sampling_res, num_hops)
            if patch_res:
                filename = filename + "_{}_{}_{}".format(patch_res, patch_id, self.cutGraph)
            filename += ".pth"
            file_address = os.path.join(self.folder, filename)
            if os.path.isfile(file_address):
                # print("Loading file {}".format(file_address))
                data_dict = torch.load(file_address)
                index = data_dict.get("index", None)
                weight = data_dict.get("weight", None)
                valid_neighbors = data_dict.get("mask_valid", None)
                if patch_res is None:
                    return index, weight, valid_neighbors
                nodes = data_dict.get("nodes", None)
                mapping = data_dict.get("mapping", None)
                return index, weight, valid_neighbors, nodes, mapping

        if patch_res is None:
            nside = hp.order2nside(sampling_res)  # == 2 ** sampling_resolution
            nPix = hp.nside2npix(nside)
            pixel_id = np.arange(0, nPix, dtype=int)

            index, weight, valid_neighbors = hp_utils.k_hop_healpix_weightmatrix(resolution=sampling_res,
                                                                                 weight_type=self.weight_type,
                                                                                 use_geodesic=self.use_geodesic,
                                                                                 use_4=self.use_4connectivity,
                                                                                 nodes_id=pixel_id,
                                                                                 dtype=np.float32,
                                                                                 nest=self.isNest,
                                                                                 num_hops=num_hops,
                                                                                 )
            # print("weights before=", weight[:10,:])
            # print("valid neighbor before=", valid_neighbors[:10, :])
            index, weight = self.__normalize(index, weight, valid_neighbors, self.normalization_method)
            # print("after=", weight[:10, :])
            # print("valid neighbor after=", valid_neighbors[:10, :])
            # index = torch.from_numpy(index)
            # weight = torch.from_numpy(weight)
            # valid_neighbors = torch.from_numpy(valid_neighbors)
            # index[~valid_neighbors] = 0
            # weight[~valid_neighbors] = 0
            if self.folder:
                print("Saving file {}".format(file_address))
                torch.save({"index": index, "weight": weight, "mask_valid": valid_neighbors}, file_address)

            return index, weight, valid_neighbors

        # for Patch based, we temporary deactivate normalization for the whole images_equirectangular because we want to have the normalization per patch
        tmp_norm = self.normalization_method
        self.normalization_method = "non"
        index, weight, valid_neighbors = self.getStruct(sampling_res=sampling_res, num_hops=num_hops)
        self.normalization_method = tmp_norm  # return back to the original normalization

        n_patches, nPix_per_patch = self.getPatchesInfo(sampling_res, patch_res)
        assert patch_id >= 0 and patch_id < n_patches, "patch_id={} is not in valid range [0, {})".format(patch_id,
                                                                                                          n_patches)

        # https://github.com/rusty1s/pytorch_geometric/issues/1205
        # https://github.com/rusty1s/pytorch_geometric/issues/973
        interested_nodes = torch.arange(nPix_per_patch * patch_id, nPix_per_patch * (patch_id + 1), dtype=torch.long)

        if self.cutGraph:
            index = index.narrow(dim=0, start=nPix_per_patch * patch_id, length=nPix_per_patch).detach().clone()
            weight = weight.narrow(dim=0, start=nPix_per_patch * patch_id, length=nPix_per_patch).detach().clone()
            valid_neighbors = (index >= nPix_per_patch * patch_id) & (
                        index < nPix_per_patch * (patch_id + 1)).detach().clone()
            index -= nPix_per_patch * patch_id

            nodes = interested_nodes
            mapping = None
        else:
            tmp_valid = valid_neighbors.narrow(dim=0, start=nPix_per_patch * patch_id,
                                               length=nPix_per_patch).clone().detach()
            nodes, inv = index.narrow(dim=0, start=nPix_per_patch * patch_id, length=nPix_per_patch)[tmp_valid].unique(
                return_inverse=True)
            mapping = (nodes.unsqueeze(1) == interested_nodes).nonzero()[:, 0]
            index = index.index_select(dim=0, index=nodes)
            weight = weight.index_select(dim=0, index=nodes)
            valid_neighbors = torch.zeros(len(nodes), valid_neighbors.size(1), dtype=torch.bool)
            valid_neighbors[mapping, :] = tmp_valid
            index[valid_neighbors] = inv

        # print("before=", weight[:10, :])
        # print("valid neighbor before=", valid_neighbors[:10, :])
        index, weight = self.__normalize(index, weight, valid_neighbors, self.normalization_method)
        # print("after=", weight[:10, :])
        # print("valid neighbor after=", valid_neighbors[:10, :])
        # index[~valid_neighbors] = 0
        # weight[~valid_neighbors] = 0

        if self.folder:
            print("Saving file {}".format(file_address))
            torch.save({"index": index,
                        "weight": weight,
                        "mask_valid": valid_neighbors,
                        "nodes": nodes,
                        "mapping": mapping},
                       file_address)
        return index, weight, valid_neighbors, nodes, mapping

    def __normalize(self, index, weight, valid_neighbors, normalization_method):
        assert normalization_method in ['non', 'sym', "sym8", 'sym_neighbors',
                                        'global_directional_avg'], 'normalization_method not defined'

        if not isinstance(index, torch.Tensor):
            index = torch.from_numpy(index)
        if not isinstance(weight, torch.Tensor):
            weight = torch.from_numpy(weight)
        if not isinstance(valid_neighbors, torch.Tensor):
            valid_neighbors = torch.from_numpy(valid_neighbors)

        index[~valid_neighbors] = 0
        weight[~valid_neighbors] = 0

        if normalization_method == "non":
            return index, weight

        if normalization_method == "sym":
            weight.div_(weight.sum(dim=1, keepdim=True))
        elif normalization_method == "sym8":
            weight.div_(weight.sum(dim=1, keepdim=True))
            weight *= 8
        elif normalization_method == "sym_neighbors":
            n_neighbors = valid_neighbors.sum(dim=1, keepdim=True)
            weight.div_(weight.sum(dim=1, keepdim=True))
            weight.mul_(n_neighbors)
        elif normalization_method == "global_directional_avg":
            for col in range(weight.shape[1]):
                weight_col = weight[:, col]
                weight_col.div_(weight_col.sum())
                if self.weight_type == "distance":
                    weight_col = 2. - weight_col
                    raise NotImplementedError("Not sure about it")

        return index, weight

    def getPatchesInfo(self, sampling_res, patch_res):
        # assert patch_res <= sampling_res, "patch_res can not be greater than sampling_res"
        nside = hp.order2nside(sampling_res)  # == 2 ** sampling_resolution

        if patch_res is None or patch_res < 0:  # Negative value means that the whole sphere is desired
            return 1, hp.nside2npix(nside)

        patch_width = hp.order2nside(patch_res)
        nPix_per_patch = patch_width * patch_width
        nside_patch = nside // patch_width
        n_patches = hp.nside2npix(nside_patch)
        return n_patches, nPix_per_patch

    def getLayerStructUpsampling(self, scaling_factor_upsampling, hop_upsampling, resolution, patch_resolution=None,
                                 patch_id=None, inputHopFromDownsampling=None):
        # print("starting unsampling graph construction", flush=True)
        assert len(scaling_factor_upsampling) == len(
            hop_upsampling), "list size for scaling factor and hop numbers must be equal"
        nconv_layers = len(scaling_factor_upsampling)
        list_sampling_res_conv, list_patch_res_conv = [[None] * nconv_layers for i in range(2)]
        list_sampling_res_conv[0] = resolution
        list_patch_res_conv[0] = patch_resolution

        patching = False
        if all(v is not None for v in [patch_resolution, patch_id]) and (patch_resolution > 0):
            patching = True

        for l in range(1, nconv_layers):
            list_sampling_res_conv[l] = hp_utils.healpix_getResolutionUpsampled(list_sampling_res_conv[l - 1],
                                                                                scaling_factor_upsampling[l - 1])
            if patching:
                list_patch_res_conv[l] = hp_utils.healpix_getResolutionUpsampled(list_patch_res_conv[l - 1],
                                                                                 scaling_factor_upsampling[l - 1])

        highest_sampling_res = hp_utils.healpix_getResolutionUpsampled(list_sampling_res_conv[-1],
                                                                       scaling_factor_upsampling[-1])
        if patching:
            highest_patch_res = hp_utils.healpix_getResolutionUpsampled(list_patch_res_conv[-1],
                                                                        scaling_factor_upsampling[-1])

        list_index, list_weight, list_mapping_upsampling = [[None] * nconv_layers for i in range(3)]

        K = hop_upsampling.copy()
        if inputHopFromDownsampling is not None:
            K[0] += inputHopFromDownsampling

        if not patching:
            l_first = next(
                (i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]),
                -1) + 1
            aggregated_K = np.sum(K[
                                  l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
            index, weight, _ = self.getStruct(sampling_res=list_sampling_res_conv[-1], num_hops=aggregated_K)
            list_index[l_first], list_weight[l_first] = index, weight
            for l in reversed(range(nconv_layers - 1)):
                if list_sampling_res_conv[l] != list_sampling_res_conv[l + 1]:
                    l_first = next(
                        (i for i in reversed(range(l + 1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]),
                        -1) + 1
                    aggregated_K = np.sum(K[
                                          l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                    index, weight, _ = self.getStruct(sampling_res=list_sampling_res_conv[l], num_hops=aggregated_K)

                    list_index[l_first], list_weight[l_first] = index, weight

            return {"list_sampling_res": list_sampling_res_conv, "list_index": list_index, "list_weight": list_weight,
                    "output_sampling_res": highest_sampling_res}

        if self.cutGraph:  # cutting the graph in the patch part. This means that border nodes lose their connectivity with outside of the patch
            l_first = next(
                (i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]),
                -1) + 1
            aggregated_K = np.sum(K[
                                  l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
            index, weight, _, _, _ = self.getStruct(sampling_res=list_sampling_res_conv[-1], num_hops=aggregated_K,
                                                    patch_res=list_patch_res_conv[-1], patch_id=patch_id)
            list_index[l_first], list_weight[l_first] = index, weight
            for l in reversed(range(nconv_layers - 1)):
                if list_sampling_res_conv[l] != list_sampling_res_conv[l + 1]:
                    l_first = next(
                        (i for i in reversed(range(l + 1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]),
                        -1) + 1
                    aggregated_K = np.sum(K[
                                          l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                    index, weight, _, _, _ = self.getStruct(sampling_res=list_sampling_res_conv[l],
                                                            num_hops=aggregated_K, patch_res=list_patch_res_conv[l],
                                                            patch_id=patch_id)
                    list_index[l_first], list_weight[l_first] = index, weight

            return {"list_sampling_res": list_sampling_res_conv, "list_patch_res": list_patch_res_conv,
                    "list_index": list_index, "list_weight": list_weight,
                    "output_sampling_res": highest_sampling_res, "output_patch_res": highest_patch_res}

        # TODO: This part has not been checked for bugs
        l_first = next(
            (i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]),
            -1) + 1
        aggregated_K = np.sum(
            K[l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
        index, weight, _, nodes, mapping = self.getStruct(sampling_res=list_sampling_res_conv[-1],
                                                          num_hops=aggregated_K, patch_res=list_patch_res_conv[-1],
                                                          patch_id=patch_id)

        if highest_sampling_res != list_sampling_res_conv[-1]:
            n_bitshit = 2 * (highest_sampling_res - list_sampling_res_conv[-1])
            n_children = 1 << n_bitshit
            mapping = mapping << n_bitshit
            mapping = mapping.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
            mapping = mapping.flatten()
        list_mapping_upsampling[-1] = mapping
        list_index[l_first], list_weight[l_first] = index, weight

        for l in reversed(range(nconv_layers - 1)):
            if list_sampling_res_conv[l] != list_sampling_res_conv[l + 1]:
                n_bitshit = 2 * (list_sampling_res_conv[l + 1] - list_sampling_res_conv[l])
                parent_nodes = nodes >> n_bitshit
                parent_nodes = parent_nodes.unique()

                l_first = next(
                    (i for i in reversed(range(l + 1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]),
                    -1) + 1
                aggregated_K = np.sum(K[
                                      l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                index, weight, valid_neighbors = self.getStruct(sampling_res=list_sampling_res_conv[l],
                                                                num_hops=aggregated_K)

                index = index.index_select(0, parent_nodes)
                weight = weight.index_select(0, parent_nodes)
                valid_neighbors = valid_neighbors.index_select(0, parent_nodes)

                parent_nodes, inv = index[valid_neighbors].unique(return_inverse=True)
                index[valid_neighbors] = inv

                index[~valid_neighbors] = 0
                weight[~valid_neighbors] = 0

                n_children = 1 << n_bitshit
                generated_children_nodes_next_layer = parent_nodes << n_bitshit
                generated_children_nodes_next_layer = generated_children_nodes_next_layer.unsqueeze(1).repeat(1,
                                                                                                              n_children) + torch.arange(
                    n_children)
                generated_children_nodes_next_layer = generated_children_nodes_next_layer.flatten()
                mapping = (nodes.unsqueeze(1) == generated_children_nodes_next_layer).nonzero()[:, 1]

                nodes = parent_nodes

                list_mapping_upsampling[l] = mapping
                list_index[l_first], list_weight[l_first] = index, weight

        # print("ending unsampling graph construction", flush=True)
        return {"list_sampling_res": list_sampling_res_conv, "list_patch_res": list_patch_res_conv,
                "list_index": list_index, "list_weight": list_weight,
                "list_mapping": list_mapping_upsampling,
                "input_nodes": nodes,
                "output_sampling_res": highest_sampling_res, "output_patch_res": highest_patch_res}

    def getLayerStructs(self, scaling_factor_downsampling, hop_downsampling, scaling_factor_upsampling, hop_upsampling,
                        upsampled_resolution, patch_upsampled_resolution=None, patch_id=None):

        assert len(scaling_factor_downsampling) == len(
            hop_downsampling), "number of layers between scale factor and hops must be equal"
        nlayers_downsampling = len(scaling_factor_downsampling)

        assert len(scaling_factor_upsampling) == len(
            hop_upsampling), "number of layers between scale factor and hops must be equal"

        patching = False
        if all(v is not None for v in [patch_upsampled_resolution, patch_id]) and (patch_upsampled_resolution > 0):
            patching = True

        list_downsampling_res_conv, list_downsampling_patch_res_conv = [[None] * nlayers_downsampling for i in range(2)]
        list_downsampling_res_conv[0] = upsampled_resolution
        list_downsampling_patch_res_conv[0] = patch_upsampled_resolution

        for l in range(1, nlayers_downsampling):
            list_downsampling_res_conv[l] = hp_utils.healpix_getResolutionDownsampled(list_downsampling_res_conv[l - 1],
                                                                                      scaling_factor_downsampling[
                                                                                          l - 1])
            if patching:
                list_downsampling_patch_res_conv[l] = hp_utils.healpix_getResolutionDownsampled(
                    list_downsampling_patch_res_conv[l - 1], scaling_factor_downsampling[l - 1])

        lowest_sampling_res = hp_utils.healpix_getResolutionDownsampled(list_downsampling_res_conv[-1],
                                                                        scaling_factor_downsampling[-1])
        if patching:
            lowest_patch_res = hp_utils.healpix_getResolutionDownsampled(list_downsampling_patch_res_conv[-1],
                                                                         scaling_factor_downsampling[-1])

        list_index_downsampling, list_weight_downsampling, list_mapping_downsampling = [[None] * nlayers_downsampling
                                                                                        for i in range(3)]

        lowest_res_aggregated_hop = 0
        if list_downsampling_res_conv[-1] == lowest_sampling_res:
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if
                            list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            lowest_res_aggregated_hop = np.sum(hop_downsampling[l_first:])

        if not patching:
            dict_graphs = dict()
            dict_graphs["upsampling"] = self.getLayerStructUpsampling(scaling_factor_upsampling, hop_upsampling,
                                                                      lowest_sampling_res,
                                                                      inputHopFromDownsampling=lowest_res_aggregated_hop)
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if
                            list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            if list_downsampling_res_conv[-1] == lowest_sampling_res:
                index = dict_graphs["upsampling"]["list_index"][0]
                weight = dict_graphs["upsampling"]["list_weight"][0]
            else:
                aggregated_K = np.sum(hop_downsampling[
                                      l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                index, weight, _ = self.getStruct(sampling_res=list_downsampling_res_conv[-1], num_hops=aggregated_K)

            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
            for l in reversed(range(nlayers_downsampling - 1)):
                if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
                    l_first = next((i for i in reversed(range(l + 1)) if
                                    list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
                    aggregated_K = np.sum(hop_downsampling[
                                          l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                    index, weight, _ = self.getStruct(sampling_res=list_downsampling_res_conv[l], num_hops=aggregated_K)
                    list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight

            dict_graphs["downsampling"] = {"list_sampling_res": list_downsampling_res_conv,
                                           "list_index": list_index_downsampling,
                                           "list_weight": list_weight_downsampling}
            return dict_graphs

        if self.cutGraph:  # cutting the graph in the patch part. This means that border nodes lose their connectivity with outside of the patch
            dict_graphs = dict()
            dict_graphs["upsampling"] = self.getLayerStructUpsampling(scaling_factor_upsampling, hop_upsampling,
                                                                      lowest_sampling_res,
                                                                      patch_resolution=lowest_patch_res,
                                                                      patch_id=patch_id,
                                                                      inputHopFromDownsampling=lowest_res_aggregated_hop)
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if
                            list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            if list_downsampling_res_conv[-1] == lowest_sampling_res:
                index = dict_graphs["upsampling"]["list_index"][0]
                weight = dict_graphs["upsampling"]["list_weight"][0]
            else:
                aggregated_K = np.sum(hop_downsampling[
                                      l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                index, weight, _, _, _ = self.getStruct(sampling_res=list_downsampling_res_conv[-1],
                                                        num_hops=aggregated_K,
                                                        patch_res=list_downsampling_patch_res_conv[-1],
                                                        patch_id=patch_id)

            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
            for l in reversed(range(nlayers_downsampling - 1)):
                if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
                    l_first = next((i for i in reversed(range(l + 1)) if
                                    list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
                    aggregated_K = np.sum(hop_downsampling[
                                          l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                    index, weight, _, _, _ = self.getStruct(sampling_res=list_downsampling_res_conv[l],
                                                            num_hops=aggregated_K,
                                                            patch_res=list_downsampling_patch_res_conv[l],
                                                            patch_id=patch_id)
                    list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight

            _, nPixPerPatch = self.getPatchesInfo(upsampled_resolution, patch_upsampled_resolution)
            range_downsampling_input_to_patch = (int(patch_id * nPixPerPatch), int((patch_id + 1) * nPixPerPatch))

            dict_graphs["downsampling"] = {"list_sampling_res": list_downsampling_res_conv,
                                           "list_patch_res": list_downsampling_patch_res_conv,
                                           "list_index": list_index_downsampling,
                                           "list_weight": list_weight_downsampling,
                                           "range_downsampling_input_to_patch": range_downsampling_input_to_patch}
            return dict_graphs

        # TODO: This part has not been checked for bugs
        dict_graphs = dict()
        dict_graphs["upsampling"] = self.getLayerGraphUpsampling(scaling_factor_upsampling, hop_upsampling,
                                                                 lowest_sampling_res, patch_resolution=lowest_patch_res,
                                                                 patch_id=patch_id,
                                                                 inputHopFromDownsampling=lowest_res_aggregated_hop)

        # print("starting downsampling graph construction", flush=True)

        nodes = dict_graphs["upsampling"]["input_nodes"]
        index = dict_graphs["upsampling"]["list_index"][0]
        weight = dict_graphs["upsampling"]["list_weight"][0]

        _, nPixPerPatch = self.getPatchesInfo(lowest_sampling_res, lowest_patch_res)
        ind_start = (
                    nodes == patch_id * nPixPerPatch).nonzero().item()  # to find index of the node==patch_id*nPixPerPatch
        # Maybe later I can remove the next assert check.
        assert torch.all(torch.eq(nodes.narrow(dim=0, start=ind_start, length=nPixPerPatch),
                                  torch.arange(patch_id * nPixPerPatch, (patch_id + 1) * nPixPerPatch,
                                               dtype=nodes.dtype))), "patch nodes from upsampling must already contains last resolution patch nodes in a sorted order"
        range_downsampling_output_to_patch = (ind_start, ind_start + nPixPerPatch)

        if list_downsampling_res_conv[
            -1] == lowest_sampling_res:  # This means that last conv layer of downsampling has same size of first conv layer of upsampling
            l_first = next((i for i in reversed(range(nlayers_downsampling)) if
                            list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            list_mapping_downsampling[
                -1] = None  # This means that we are in the middle of layer so no mapping is needed
            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
        else:
            n_bitshit = 2 * (list_downsampling_res_conv[-1] - lowest_sampling_res)
            n_children = 1 << n_bitshit
            interested_nodes = nodes << n_bitshit
            interested_nodes = interested_nodes.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
            interested_nodes = interested_nodes.flatten()

            l_first = next((i for i in reversed(range(nlayers_downsampling)) if
                            list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
            aggregated_K = np.sum(hop_downsampling[
                                  l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
            index, weight, valid_neighbors = self.getStruct(sampling_res=list_downsampling_res_conv[-1],
                                                            num_hops=aggregated_K)

            index = index.index_select(0, interested_nodes)
            weight = weight.index_select(0, interested_nodes)
            valid_neighbors = valid_neighbors.index_select(0, interested_nodes)

            nodes, inv = index[valid_neighbors].unique(return_inverse=True)
            index[valid_neighbors] = inv
            mapping = (nodes.unsqueeze(1) == interested_nodes).nonzero()[:, 0]

            index[~valid_neighbors] = 0
            weight[~valid_neighbors] = 0

            interested_nodes = nodes
            list_mapping_downsampling[-1] = mapping
            list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight

        for l in reversed(range(nlayers_downsampling - 1)):
            if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
                n_bitshit = 2 * (list_downsampling_res_conv[l] - list_downsampling_res_conv[l + 1])
                n_children = 1 << n_bitshit
                nodes = nodes << n_bitshit
                interested_nodes = interested_nodes.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
                interested_nodes = interested_nodes.flatten()

                l_first = next((i for i in reversed(range(l + 1)) if
                                list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
                aggregated_K = np.sum(hop_downsampling[
                                      l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
                index, weight, valid_neighbors = self.getGraph(sampling_res=list_downsampling_res_conv[l],
                                                               num_hops=aggregated_K)

                index = index.index_select(0, interested_nodes)
                weight = weight.index_select(0, interested_nodes)
                valid_neighbors = valid_neighbors.index_select(0, interested_nodes)

                nodes, inv = index[valid_neighbors].unique(return_inverse=True)
                index[valid_neighbors] = inv
                mapping = (nodes.unsqueeze(1) == interested_nodes).nonzero()[:, 0]

                index[~valid_neighbors] = 0
                weight[~valid_neighbors] = 0

                interested_nodes = nodes

                list_mapping_downsampling[l] = mapping
                list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight

        _, nPixPerPatch = self.getPatchesInfo(upsampled_resolution, patch_upsampled_resolution)
        ind_start = (
                    nodes == patch_id * nPixPerPatch).nonzero().item()  # to find index of the node==patch_id*nPixPerPatch
        # Maybe later I can remove the next assert check.
        assert torch.all(torch.eq(nodes.narrow(dim=0, start=ind_start, length=nPixPerPatch),
                                  torch.arange(patch_id * nPixPerPatch, (patch_id + 1) * nPixPerPatch,
                                               dtype=nodes.dtype))), "patch nodes from upsampling must already contains last resolution patch nodes in a sorted order"
        range_downsampling_input_to_patch = (ind_start, ind_start + nPixPerPatch)

        # print("ending downsampling graph construction", flush=True)
        dict_graphs["downsampling"] = {"list_sampling_res": list_downsampling_res_conv,
                                       "list_patch_res": list_downsampling_patch_res_conv,
                                       "list_index": list_index_downsampling, "list_weight": list_weight_downsampling,
                                       "input_nodes": nodes, "list_mapping": list_mapping_downsampling,
                                       "range_downsampling_output_to_patch": range_downsampling_output_to_patch,
                                       "range_downsampling_input_to_patch": range_downsampling_input_to_patch}

        return dict_graphs
