"""it is copied from DeepSphere """
import numpy as np
import healpy as hp
import torch
from scipy import sparse
from . import common_function as utils_common
from . import graph as graph_utils
import networkx as nx
import cv2

def healpixTo12subImages(x, isNest):
    assert x.ndim == 1 or x.ndim==2, "The input should of one or 2 dimension"
    x = x.reshape((x.shape[0], -1))
    npix = x.shape[0]
    nside = hp.npix2nside(npix)
    imgs = np.empty((12,nside, nside, x.shape[1]), dtype=x.dtype)
    ix, iy, f = hp.pix2xyf(nside, np.arange(npix), nest=isNest)
    iy = nside-1-iy
    imgs[f, iy, ix, :] = x[:]
    # for i in range(x.shape[1]):
    #     imgs[f, iy, ix, i] = x[:, i]
    return imgs


def k_hop_healpix_weightmatrix(resolution=6, nest=True, nodes_id=None, dtype=np.float32, use_4=False, use_geodesic=True, weight_type="gaussian", num_hops=1):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling. It also returns the coords as the second output, and a matrix in which each row idx shows neighbors of node idx upto hop num_hops
        Parameters
        ----------
        resolution : int
            The healpix resolution order parameter, must be a power of 2, less than 2**30.
        nest : bool, optional
            if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
        nodes_id : list of int, optional
            List of node ids to use. This allows to build the graph from a part of
            the sphere only. If None, the default, the whole sphere is used.
        dtype : images_equirectangular-type, optional
            The desired images_equirectangular type of the weight matrix.
        use_4 : bool, optional
            Use 4-connected neighboring if it is true, othehrwise, use 8-connected neighboring
        weight_type : a string indicating weighting method

        use_geodesic : bool, optional
            True: use geodesic distance in weight calculation, otherwise use euclidean distance

        num_hops : int, optional
            The number of hops
        """
    nside = hp.order2nside(resolution)

    assert weight_type in ['gaussian', 'identity', 'inverse_distance', 'distance'], 'weight_type parameter must be gaussian, identity, inverse_distance, or distance'
    assert num_hops >=1, "num_hops must be greater than or equal to 1"

    set_requested_nodes_id = set(nodes_id)
    assert len(nodes_id) == len(set_requested_nodes_id), 'There should be no duplicate of the indices'

    if nodes_id is None:
        nodes_id = range(hp.nside2npix(nside))

    usefast = utils_common.checkConsecutive(nodes_id)   # If the user input is not consecutive nodes, we need to use a slower method.


    assert not use_4, "for the moment we only support 8 neighborhood"
    n_firstHopNeighbors = 4 if use_4 else 8

    x, y, z = hp.pix2vec(nside, nodes_id, nest=nest)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    if usefast:
        min_id = min(nodes_id)
        max_id = max(nodes_id)
    else:
        inverse_map = [np.nan] * (hp.nside2npix(nside))
        for index, i in enumerate(nodes_id):
            inverse_map[i] = index


    # SW(0), W(1), NW(2), N(3), NE(4), E(5), SE(6) and S(7)
    # The next array just indicates where to search at each pixel for the outer hop
    whereToSearchNeighbor = np.full(utils_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=num_hops - 1), 0, dtype=int)

    for a in range(1, n_firstHopNeighbors):
        for h in range(1, num_hops):
            if (a & 1): # if odd
                whereToSearchNeighbor[utils_common.sumOfAP(a=a, d=n_firstHopNeighbors, n=h)]=a
            else:
                whereToSearchNeighbor[utils_common.sumOfAP(a=a-1, d=n_firstHopNeighbors, n=h)+1:utils_common.sumOfAP(a=a+1, d=n_firstHopNeighbors, n=h)] = a

    all_node_ids = range(hp.nside2npix(nside))
    allNodes_firstHopNeighbors = hp.pixelfunc.get_all_neighbours(nside, all_node_ids, nest=nest)
    allNodes_firstHopNeighbors = allNodes_firstHopNeighbors.T


    (edge_index_allNodes, edge_weight_allNodes, keep_allNodes), _ = healpix_weightmatrix(resolution=resolution,
                                                                                          weight_type=weight_type,
                                                                                          use_geodesic=use_geodesic,
                                                                                          use_4=use_4,
                                                                                          nodes_id=all_node_ids,
                                                                                          dtype=dtype,
                                                                                          nest=nest,
                                                                                          keep_directional_edge=True
                                                                                          )
    assert np.array_equal(np.repeat(all_node_ids, n_firstHopNeighbors), edge_index_allNodes[0,:]), "edge_index_allNodes first row must be node ids repetitions"
    assert np.array_equal(np.reshape(edge_index_allNodes[1, :], (-1, n_firstHopNeighbors)), allNodes_firstHopNeighbors), "edge_index_allNodes second row must be allNodes_firstHopNeighbors"

    n_neighbors = utils_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=num_hops)
    hop_neighbors = np.full((len(nodes_id), n_neighbors), -1, dtype=allNodes_firstHopNeighbors.dtype)
    hop_weights = np.full((len(nodes_id), n_neighbors), 0, dtype=dtype)
    valid_neighbors = np.full((len(nodes_id), n_neighbors), False, dtype=bool)

    reshaped_allNodesWeights = np.reshape(edge_weight_allNodes, (-1, n_firstHopNeighbors))
    if usefast:
        hop_neighbors[:, 0:n_firstHopNeighbors] = allNodes_firstHopNeighbors[min_id:max_id+1, :]
        hop_weights[:, 0:n_firstHopNeighbors] = reshaped_allNodesWeights[min_id:max_id+1, :]

    else:
        hop_neighbors[:, 0:n_firstHopNeighbors] = allNodes_firstHopNeighbors[nodes_id, :]
        hop_weights[:, 0:n_firstHopNeighbors] = reshaped_allNodesWeights[nodes_id, :]


    edge_index_allNodes  = edge_index_allNodes[:, keep_allNodes]
    edge_weight_allNodes = edge_weight_allNodes[keep_allNodes]

    if num_hops == 1:
        # Remove pixels that are out of our indexes of interest (part of sphere).
        if usefast:
            valid_neighbors[:, 0:n_firstHopNeighbors] = (hop_neighbors <= max_id) & (hop_neighbors >= min_id)
            hop_neighbors[valid_neighbors] = hop_neighbors[valid_neighbors] - min_id
        else:
            valid_neighbors[:, 0:n_firstHopNeighbors] = np.isin(hop_neighbors, nodes_id)  # or simply using keep_nodes = [c in set_requested_nodes_id for c in corresponding_neighbors_id]
            hop_neighbors[valid_neighbors] = np.vectorize(lambda el: inverse_map[el])(hop_neighbors[valid_neighbors])
        outmask = np.logical_not(valid_neighbors)
        hop_neighbors[outmask] = 0
        hop_weights[outmask] = 0
        return hop_neighbors, hop_weights, valid_neighbors

    for row in range(len(nodes_id)):
        id = nodes_id[row]
        nodes_at_kth_hop, edge_mask = graph_utils.k_hop_nodes(id, num_hops=num_hops, edge_index=edge_index_allNodes, relabel_nodes=False,
                                                              num_nodes=None, flow='target_to_source')

        for k in range(1, num_hops):
            hop = k + 1  # change 0-based to one-based for hop number
            n_elemes_kth_hop = n_firstHopNeighbors + (hop - 1) * n_firstHopNeighbors
            curr_hop_first_element = utils_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=hop - 1)
            pre_hop_first_elem_ind = utils_common.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=hop - 2)


            ind = 0  # index which tracks elements in curr_hop_first_element:curr_hop_first_element+ n_elemes_kth_hop
            last_added_node_id_of_current_hop = -1
            for ind_pre_hop in range(pre_hop_first_elem_ind, curr_hop_first_element):
                # in healpix, neighboring pixels which are side by side of the desired pixel are at position 0:8:2 of the neighbors and the corner neighboring pixels are at location 1:8:2

                neighbor_id_pre_hop = hop_neighbors[row, ind_pre_hop]
                if (whereToSearchNeighbor[ind_pre_hop] & 1):  # if odd: in the corner. The corner pixels in pre hop are responsible to fill 3 pixels of the next hop
                    if neighbor_id_pre_hop < 0:
                        hop_neighbors[row, curr_hop_first_element + ind] = -1
                        hop_neighbors[row, curr_hop_first_element + ind + 1] = -1
                        hop_neighbors[row, curr_hop_first_element + ind + 2] = -1
                        ind += 3
                    elif last_added_node_id_of_current_hop < 0:  # currently no node has been added to the current hop
                        # Exception handling: in this case there are two pixels that are side by side to neighbor_id_pre_hop. We have to choose the one that is not in the neighbor of next hop_neighbors[row, ind_pre_hop+1] in the previous hop
                        next_neighbor_id_pre_hop = hop_neighbors[row, ind_pre_hop+1]
                        assert next_neighbor_id_pre_hop>=0, "this should exist otherwise I need another exception handling"
                        node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 0:8:2] if i in nodes_at_kth_hop[k] and i not in allNodes_firstHopNeighbors[next_neighbor_id_pre_hop, :] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        hop_neighbors[row, curr_hop_first_element + ind] = node_id_to_add
                        last_added_node_id_of_current_hop = node_id_to_add if node_id_to_add >= 0 else last_added_node_id_of_current_hop
                        ind += 1

                        if last_added_node_id_of_current_hop < 0:  # currently no node has been added to the current hop
                            node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 1:8:2] if i in nodes_at_kth_hop[k] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        else:
                            node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 1:8:2] if i in nodes_at_kth_hop[k] and i in allNodes_firstHopNeighbors[last_added_node_id_of_current_hop, 0:8:2] and i not in hop_neighbors[row,curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        hop_neighbors[row, curr_hop_first_element + ind] = node_id_to_add
                        last_added_node_id_of_current_hop = node_id_to_add if node_id_to_add >= 0 else last_added_node_id_of_current_hop
                        ind += 1

                        if last_added_node_id_of_current_hop < 0:  # currently no node has been added to the current hop
                            node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 0:8:2] if i in nodes_at_kth_hop[k] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        else:
                            node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 0:8:2] if i in nodes_at_kth_hop[k] and i in allNodes_firstHopNeighbors[last_added_node_id_of_current_hop, 0:8:2] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        hop_neighbors[row, curr_hop_first_element + ind] = node_id_to_add
                        last_added_node_id_of_current_hop = node_id_to_add if node_id_to_add >= 0 else last_added_node_id_of_current_hop
                        ind += 1

                    else:
                        node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 0:8:2] if i in nodes_at_kth_hop[k] and i in allNodes_firstHopNeighbors[last_added_node_id_of_current_hop, 0:8:2] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        hop_neighbors[row, curr_hop_first_element + ind] = node_id_to_add
                        last_added_node_id_of_current_hop = node_id_to_add if node_id_to_add >= 0 else last_added_node_id_of_current_hop
                        ind += 1

                        # cornoer pixel in outer hop
                        node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 1:8:2] if i in nodes_at_kth_hop[k] and i in allNodes_firstHopNeighbors[last_added_node_id_of_current_hop, 0:8:2] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        hop_neighbors[row, curr_hop_first_element + ind] = node_id_to_add
                        last_added_node_id_of_current_hop = node_id_to_add if node_id_to_add >= 0 else last_added_node_id_of_current_hop
                        ind += 1

                        node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 0:8:2] if i in nodes_at_kth_hop[k] and i in allNodes_firstHopNeighbors[last_added_node_id_of_current_hop, 0:8:2] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                        hop_neighbors[row, curr_hop_first_element + ind] = node_id_to_add
                        last_added_node_id_of_current_hop = node_id_to_add if node_id_to_add >= 0 else last_added_node_id_of_current_hop
                        ind += 1


                else:   # not in the corner
                    if neighbor_id_pre_hop < 0:
                        node_id_to_add = -1
                    elif last_added_node_id_of_current_hop < 0:   # currently no node has been added to the current hop
                        node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 0:8:2] if i in nodes_at_kth_hop[k] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element + ind]), -1)
                    else:
                        node_id_to_add = next((i for i in allNodes_firstHopNeighbors[neighbor_id_pre_hop, 0:8:2] if i in nodes_at_kth_hop[k] and i in allNodes_firstHopNeighbors[last_added_node_id_of_current_hop, 0:8:2] and i not in hop_neighbors[row, curr_hop_first_element: curr_hop_first_element+ind]), -1)

                    hop_neighbors[row, curr_hop_first_element + ind] = node_id_to_add
                    last_added_node_id_of_current_hop = node_id_to_add if node_id_to_add >= 0 else last_added_node_id_of_current_hop
                    ind += 1


        # further test, remove it later
        arr = hop_neighbors[row, :]
        arr = arr[arr != -1]
        assert set(arr) == set(np.concatenate(nodes_at_kth_hop)), "the two arrays are not equal for pixel id {}".format(id)

        corresponding_neighbors_id = hop_neighbors[row, :]
        # Remove pixels that are out of our indexes of interest (part of sphere).
        if usefast:
            keep_nodes = (corresponding_neighbors_id <= max_id) & (corresponding_neighbors_id >= min_id)
            corresponding_neighbors_id[keep_nodes] = corresponding_neighbors_id[keep_nodes] - min_id
            id = id - min_id
            subgraph_edge_index = edge_index_allNodes[:, edge_mask]
            keep_edges = (subgraph_edge_index[0,:] <= max_id) & (subgraph_edge_index[0,:] >= min_id) & (subgraph_edge_index[1,:] <= max_id) & (subgraph_edge_index[1,:] >= min_id)
            subgraph_edge_index = subgraph_edge_index[:, keep_edges]
            subgraph_edge_index = subgraph_edge_index - min_id

        else:
            keep_nodes = np.isin(corresponding_neighbors_id, nodes_id)  # or simply using keep_nodes = [c in set_requested_nodes_id for c in corresponding_neighbors_id]
            corresponding_neighbors_id[keep_nodes] = np.vectorize(lambda el: inverse_map[el])(corresponding_neighbors_id[keep_nodes])
            id = inverse_map[id]
            subgraph_edge_index = edge_index_allNodes[:, edge_mask]
            keep_edges = np.isin(subgraph_edge_index[0, :], nodes_id) & np.isin(subgraph_edge_index[1, :], nodes_id)
            subgraph_edge_index = subgraph_edge_index[:, keep_edges]
            subgraph_edge_index = np.vectorize(lambda el: inverse_map[el])(subgraph_edge_index)



        inv_keep_nodes = np.logical_not(keep_nodes)
        corresponding_neighbors_id[inv_keep_nodes] = 0    # note: this doesn't cause any problem that all non keep_nodes images_equirectangular refer to the first node, because we are setting the corresponding weight value to zero.
        corresponding_weights = hop_weights[row, :]
        corresponding_weights[inv_keep_nodes] = 0
        valid_neighbors[row, :] = keep_nodes


        subgraph_edge_weight = edge_weight_allNodes[edge_mask]
        subgraph_edge_weight = subgraph_edge_weight[keep_edges]
        indices = np.where(keep_nodes)[0]
        corresponding_weights[keep_nodes] = 1 # setting to one to be neutral in the multiplications
        for ind in indices:
            G = nx.Graph()
            G.add_edges_from(list(map(tuple, subgraph_edge_index.transpose())))
            for p in nx.all_shortest_paths(G, source=id, target=corresponding_neighbors_id[ind]):
                path_edges = np.array([p[:-1], p[1:]], dtype=subgraph_edge_index.dtype)
                ws = [subgraph_edge_weight[e] for e in range(subgraph_edge_index.shape[1]) if np.any(np.equal(path_edges, subgraph_edge_index[:, e].reshape(2,-1)).all(0))]
                corresponding_weights[ind] *= np.prod(ws)



    return hop_neighbors, hop_weights, valid_neighbors


def healpix_weightmatrix(resolution=6, nest=True, nodes_id=None, dtype=np.float32, use_4=False, use_geodesic=True, weight_type="gaussian", keep_directional_edge=False):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling. It also returns the coords as the second output
    Parameters
    ----------
    resolution : int
        The healpix resolution order parameter, must be a power of 2, less than 2**30.
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    nodes_id : list of int, optional
        List of node ids to use. This allows to build the graph from a part of
        the sphere only. If None, the default, the whole sphere is used.
    dtype : images_equirectangular-type, optional
        The desired images_equirectangular type of the weight matrix.
    use_4 : bool, optional
        Use 4-connected neighboring if it is true, othehrwise, use 8-connected neighboring
    weight_type : a string indicating weighting method

    use_geodesic : bool, optional
        True: use geodesic distance in weight calculation, otherwise use euclidean distance

    keep_directional_edge : bool, optional
        True: for each requested node id function returns all directional neighbors (SW, W, NW, N, NE, E, SE and S neighbours if use_4==False, and SW, NW, NE, SE if use_4==True),
        if some of nodes are not in the input list node_id, their corresponding element is -1
        False: function returns only nodes that are available in nodes_id
    """

    nside = hp.order2nside(resolution)


    assert weight_type in ['gaussian', 'identity', 'inverse_distance', 'distance'], 'weight_type parameter must be gaussian, identity, inverse_distance, or distance'
    set_requested_nodes_id = set(nodes_id)
    assert len(nodes_id) == len(set_requested_nodes_id), 'There should be no duplicates in the indices'


    if nodes_id is None:
        nodes_id = range(hp.nside2npix(nside))
    npix = len(nodes_id)  # Number of pixels.

    usefast = utils_common.checkConsecutive(nodes_id)  # If the user input is not consecutive nodes, we need to use a slower method.
    nodes_id = list(nodes_id)

    neighbors = hp.pixelfunc.get_all_neighbours(nside, nodes_id, nest=nest)
    if use_4:
        neighbors = neighbors[0:8:2]
        # neighbors = neighbors[1:8:2]
        corresponding_neighbors_id = neighbors.T.reshape((npix * 4))
        requested_node_id = np.repeat(nodes_id, 4)
    else:
        # Get the 7-8 neighbors.
        corresponding_neighbors_id = neighbors.T.reshape((npix * 8))
        requested_node_id = np.repeat(nodes_id, 8)

    # Remove pixels that are out of our indexes of interest (part of sphere).
    x, y, z = hp.pix2vec(nside, nodes_id, nest=nest)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    if usefast:
        min_id = min(nodes_id)
        max_id = max(nodes_id)
        keep = (corresponding_neighbors_id <= max_id) & (corresponding_neighbors_id >= min_id)
        requested_kept_node_coord_ind = requested_node_id[keep] - min_id
        corresponding_kept_neighbors_coord_ind = corresponding_neighbors_id[keep] - min_id
    else:
        keep = [c in set_requested_nodes_id for c in corresponding_neighbors_id]    # or simply using: keep = np.isin(corresponding_neighbors_id, nodes_id)
        inverse_map = [np.nan] * (hp.nside2npix(nside))
        for index, i in enumerate(nodes_id):
            inverse_map[i] = index

        # requested_kept_node_coord_ind = np.array([inverse_map[el] for el in requested_node_id], dtype=requested_node_id.dtype)
        # corresponding_kept_neighbors_coord_ind = np.full((corresponding_neighbors_id.shape), -1, dtype=corresponding_neighbors_id.dtype)
        # corresponding_kept_neighbors_coord_ind[keep] = [inverse_map[el] for el in corresponding_neighbors_id[keep]]
        requested_kept_node_coord_ind = [inverse_map[el] for el in requested_node_id[keep]]
        corresponding_kept_neighbors_coord_ind = [inverse_map[el] for el in corresponding_neighbors_id[keep]]


    # Compute Euclidean distances between neighbors.
    # Get the coordinates.

    distances = np.sqrt(np.sum((coords[requested_kept_node_coord_ind] - coords[corresponding_kept_neighbors_coord_ind])**2, axis=1))
    if use_geodesic:
        distances = 2. * np.arcsin(distances / 2.)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
    if weight_type == "gaussian":
        kernel_width = np.mean(distances)
        weights = np.exp(-(distances*distances) / (2 * kernel_width))
    elif weight_type == "identity":
        weights = np.ones(distances.shape, dtype=dtype)
    elif weight_type == "inverse_distance":
        # Similarity proposed by Renata & Pascal, ICCV 2017.
        # weights = 1 / distances
        weights = 1 / distances
    elif weight_type == "distance":
        weights = distances

    if keep_directional_edge:
        edge_index = np.full((2, corresponding_neighbors_id.shape[0]), -1, dtype=corresponding_neighbors_id.dtype)
        # edge_index[0, keep] = requested_kept_node_coord_ind
        edge_index[0, :] = requested_node_id
        edge_index[1, keep] = corresponding_kept_neighbors_coord_ind
        edge_weight = np.full((corresponding_neighbors_id.shape[0]), 0, dtype=weights.dtype)
        edge_weight[keep] = weights
        return (edge_index, edge_weight, keep), coords
    else:
        edge_index = np.vstack((requested_kept_node_coord_ind, corresponding_kept_neighbors_coord_ind))
        edge_weight = weights
        return (edge_index, edge_weight), coords

log2_dict= {2**1:1, 2**2:2, 2**3:3, 2**4:4, 2**5:5, 2**6:6, 2**7:7}

def healpix_getChildrenPixels(resolution, upsampling_width=2, nodes_id= None, nest=True):

    nside = hp.order2nside(resolution)

    assert utils_common.is_power2(upsampling_width), 'upsampling_width should be power of 2'

    if nodes_id is None:
        nodes_id = range(hp.nside2npix(nside))

    if not nest:
        nodes_id = hp.ring2nest(nside, nodes_id)

    log2_upsampling_width = log2_dict.get(upsampling_width, int(np.log2(upsampling_width)))  # advantage of get function: if downsampling_width does not exist in log2_dict,  then get() returns the value specified in the second argument (np.log2(downsampling_width))
    n_bitshift = 2 * log2_upsampling_width

    # In nested numbering scheme every two bits represent a pixel number at a given depth, i.e., by shifting off the last two bits one can find the parent pixel number of a pixel
    children_pixels = np.transpose(np.tile(np.left_shift(nodes_id, n_bitshift), (upsampling_width * upsampling_width, 1)))
    relative_id = np.arange(0, upsampling_width * upsampling_width, dtype=children_pixels.dtype)
    children_pixels += relative_id  # adding offsers of the indices through broadcasting


    if not nest:
        nside_upsampled = 1 << (resolution + log2_upsampling_width)
        children_pixels = hp.nest2ring(nside_upsampled, children_pixels)

    return children_pixels




def healpix_getParentPixel(resolution, downsampling_width=2, nodes_id= None, nest=True):

    nside = hp.order2nside(resolution)

    assert utils_common.is_power2(downsampling_width), 'downsampling_width should be power of 2'
    assert nside >= downsampling_width, 'downsampling width is higher than nside of healpix'

    if nodes_id is None:
        nodes_id = range(hp.nside2npix(nside))

    if not nest:
        nodes_id = hp.ring2nest(nside, nodes_id)


    log2_downsampling_width = log2_dict.get(downsampling_width, int(np.log2(downsampling_width))) # advantage of get function: if downsampling_width does not exist in log2_dict,  then get() returns the value specified in the second argument (np.log2(downsampling_width))

    # In nested numbering scheme every two bits represent a pixel number at a given depth, i.e., by shifting off the last two bits one can find the parent pixel number of a pixel
    n_bitshift = 2 * log2_downsampling_width
    parent_pixel = np.right_shift(nodes_id, n_bitshift)

    if not nest:
        nside_downsampled = 1 << (resolution - log2_downsampling_width)
        parent_pixel = hp.nest2ring(nside_downsampled, parent_pixel)

    return parent_pixel

def healpix_getResolutionDownsampled(resolution_curr=6, downsampling_width=2):

    assert utils_common.is_power2(downsampling_width), 'downsampling_width should be power of 2'
    log2_downsampling_width = log2_dict.get(downsampling_width, int(np.log2(downsampling_width)))  # advantage of get function: if downsampling_width does not exist in log2_dict,  then get() returns the value specified in the second argument (np.log2(downsampling_width))
    resolution_downSample = resolution_curr - log2_downsampling_width
    # assert resolution_downSample >= 0, 'downsampling width is higher than nside of healpix'
    return resolution_downSample

def healpix_getResolutionUpsampled(resolution_curr=6, upsampling_width=2):

    assert utils_common.is_power2(upsampling_width), 'upsampling_width should be power of 2'
    log2_upsampling_width = log2_dict.get(upsampling_width, int(np.log2(upsampling_width)))  # advantage of get function: if downsampling_width does not exist in log2_dict,  then get() returns the value specified in the second argument (np.log2(downsampling_width))
    resolution_upSample = resolution_curr + log2_upsampling_width
    return resolution_upSample

def healpix_downsamplingWeightMatrix(resolution_curr=6, downsampling_width=2, nest=True, nodes_id_curr=None, dtype=np.float32, use_4=False, use_geodesic=True, weight_type="gaussian"):

    nodes_id_downsampled = healpix_getParentPixel(resolution=resolution_curr, downsampling_width=downsampling_width, nodes_id=nodes_id_curr, nest=nest)

    log2_downsampling_width = log2_dict.get(downsampling_width, int(np.log2(downsampling_width)))  # advantage of get function: if downsampling_width does not exist in log2_dict,  then get() returns the value specified in the second argument (np.log2(downsampling_width))
    resolution_downSample = resolution_curr-log2_downsampling_width

    dict_nodes_id_downsampled_set = utils_common.get_indexesDuplicateItems(nodes_id_downsampled)

    nodes_id_downsampled_set = dict_nodes_id_downsampled_set.keys()

    weights, coords = healpix_weightmatrix(resolution=resolution_downSample, nest=nest, nodes_id=nodes_id_downsampled_set, dtype=dtype, use_4=use_4, use_geodesic=use_geodesic, weight_type=weight_type)
    return weights, coords, dict_nodes_id_downsampled_set


def sampleEquirectangularForHEALPix(equi_img, nside, interpolation, indexes=None, nest=True):
    device = equi_img.device
    if indexes is None:
        nPix = hp.nside2npix(nside)
        indexes = torch.arange(nPix, device=device)

    latitude, longitude = hp.pix2ang(nside=nside, nest=nest, ipix=indexes.cpu().numpy())
    latitude, longitude = torch.tensor(latitude, device=device), torch.tensor(longitude, device=device)

    return utils_common.sampleEquirectangular(equi_img, latitude, longitude, flip=True, interpolation=interpolation)



def xy2vec(x, y=None, flip=-1.):
    if y is None:
        x, y = x
    mask = np.asarray(x) ** 2 / 4.0 + np.asarray(y) ** 2 > 1.0
    if not mask.any():
        mask = np.ma.nomask
    if not hasattr(x, "__len__"):
        if mask is not np.ma.nomask:
            return np.nan, np.nan, np.nan
        else:
            s = np.sqrt((1 - y) * (1 + y))
            a = np.arcsin(y)
            z = 2.0 / np.pi * (a + y * s)
            phi = flip * np.pi / 2.0 * x / np.maximum(s, 1.0e-6)
            sz = np.sqrt((1 - z) * (1 + z))
            vec = sz * np.cos(phi), sz * np.sin(phi), z
            return vec
    else:
        w = mask == False
        vec = (
            np.zeros(x.shape) + np.nan,
            np.zeros(x.shape) + np.nan,
            np.zeros(x.shape) + np.nan,
        )
        s = np.sqrt((1 - y[w]) * (1 + y[w]))
        a = np.arcsin(y[w])
        vec[2][w] = 2.0 / np.pi * (a + y[w] * s)
        phi = flip * np.pi / 2.0 * x[w] / np.maximum(s, 1.0e-6)
        sz = np.sqrt((1 - vec[2][w]) * (1 + vec[2][w]))
        vec[0][w] = sz * np.cos(phi)
        vec[1][w] = sz * np.sin(phi)
        return vec
def mollw(map, xsize, bilinear_interpolation=True, isNest=True):
    assert map.ndim == 1 or map.ndim == 2, "The input should of one or 2 dimension"
    map = map.reshape((map.shape[0], -1))
    npix = map.shape[0]
    nside = hp.npix2nside(npix)

    xsize = int(xsize)
    ysize = xsize // 2
    xc, yc = (xsize - 1.0) / 2.0, (ysize - 1.0) / 2.0

    idx = np.outer(np.arange(ysize), np.ones(xsize))
    y = - (idx - yc) / yc
    idx = np.outer(np.ones(ysize), np.arange(xsize))
    x = 2.0 * (idx - xc) / xc
    mask = x ** 2 / 4.0 + y ** 2 > 1.0
    if not mask.any():
        mask = np.ma.nomask
    w = mask == False
    vec = xy2vec(np.asarray(x[w]), np.asarray(y[w]))

    img = np.zeros((x.shape[0], x.shape[1], map.shape[1]), np.uint8)
    if bilinear_interpolation:
        vec = np.asarray(vec).transpose()
        theta, phi = hp.vec2ang(vec)
        for i in range(map.shape[1]):
            values = hp.get_interp_val(map[:, i].astype(np.float64), theta, phi, nest=isNest)
            values = np.around(values)
            assert (values >= 0).all() and (values <= 255).all()
            img[w, i] = values.astype(img.dtype)

    else:
        pix = hp.vec2pix(nside, vec[0], vec[1], vec[2], nest=isNest)
        img[w, :] = map[pix, :]
    # for i in range(map.shape[1]):
    #     img[w, i] = map[pix, i]


    # img = img.astype(equi_img.dtype)
    return img

    # x = np.ma.array(x, mask=mask)
    # y = np.ma.array(y, mask=mask)
    #
    # if np.__version__ >= "1.1":
    #     matype = np.ma.core.MaskedArray
    # else:
    #     matype = np.ma.array
    # if type(x) is matype and x.mask is not np.ma.nomask:
    #     w = x.mask == False
    # else:
    #     w = slice(None)
    # img = np.zeros(x.shape, np.float64) - np.inf
    # vec = self.xy2vec(np.asarray(x[w]), np.asarray(y[w]))
    # vec = (R.Rotator(rot=rot, coord=self.mkcoord(coord))).I(vec)
    # pix = vec2pix_func(vec[0], vec[1], vec[2])
    # mpix = map[pix]
    # img[w] = mpix


def get_all_neighbours(resolution=6, nest=True, nodes_id=None, neighborhood_connection = 1):
    nside = hp.order2nside(resolution)
    n_cols = (2 * neighborhood_connection + 1) * (2 * neighborhood_connection + 1) - 1
    neighbors = np.full((nodes_id.shape[0], n_cols), -1, dtype=np.int)

    # first hop
    neighbors[:, 0:8] = hp.get_all_neighbours(nside, nodes_id).transpose()
    # for h in range(neighborhood_connection):
    raise NotImplementedError("get_all_neighbours not implemented")

    return neighbors

def get_pixelRegion(resolution, pixel_ids, nest):
    nside = hp.order2nside(resolution)
    npix = hp.nside2npix(nside)

    pixel_ring_ids = hp.nest2ring(nside, pixel_ids) if nest else pixel_ids
    npface = nside * nside
    ncap = (npface - nside) << 1
    regions = np.array(["equatorial_region"] * len(pixel_ring_ids))
    regions[pixel_ring_ids < ncap] = "north_polar_cap"
    regions[pixel_ring_ids >= (npix - ncap)] = "south_polar_cap"
    return regions

def get_regionPixelIds(resolution, region, nest):
    """
    retuns list of pixel ids in the requested region
    Args:
        resolution: int
            resolution of healpix to take sample from
        region: str
            region to take sample from: "north_polar_cap", "equatorial_region", "south_polar_cap"
        nest:
            True means nest indexing and false means ring indexing

    Returns:

    """
    nside = hp.order2nside(resolution)
    npix = hp.nside2npix(nside)
    npface = nside * nside
    ncap = (npface - nside) << 1 if nside > 1 else 4


    if region == "north_polar_cap":
        pixel_ring_ids = np.arange(0, ncap)
    elif region == "equatorial_region":
        pixel_ring_ids = np.arange(ncap, npix - ncap)
    elif region == "south_polar_cap":
        pixel_ring_ids = np.arange(npix - ncap, npix)
    else:
        raise ValueError("region is not defined")

    pixel_ids = hp.ring2nest(nside, pixel_ring_ids) if nest else pixel_ring_ids
    return pixel_ids

def get_n_ring(resolution):
    """
    Returns number of rings in the sampling
    Args:
        resolution: int
                resolution of healpix to take sample from
    Returns:
        nring: int
            number of rings in the sampling

    """
    nside = hp.order2nside(resolution)
    return 4*nside-1

def get_ring_info_small(resolution, ring, nest):
    """
        Returns useful information about a given ring of the map.
        Args:
            resolution: int
                resolution of healpix to take sample from
            ring: int
                the ring number (the number of the first ring is 1)
            nest:
                True means nest indexing and false means ring indexing

        Returns:
            startpix: int
            	the number of the first pixel in the ring (NOTE: if nest the number is in nest )
            ringpix: int
            	the number of pixels in the ring
            shifted: bool
            	if true, the center of the first pixel is not at phi=0

    """
    nside = hp.order2nside(resolution)
    npix = hp.nside2npix(nside)
    npface = nside * nside
    ncap = (npface - nside) << 1 if nside > 1 else 4

    assert ring >=1 and ring <=get_n_ring(resolution), "ring range must be between [1:number of rings]"
    if ring < nside:
        shifted = True
        ringpix = 4 * ring
        startpix = 2 * ring * (ring-1)
    elif ring < 3 * nside:
        shifted = ((ring - nside) & 1) == 0
        ringpix = 4 * nside
        startpix = ncap + (ring - nside) * ringpix
    else:
        shifted = True
        nr = 4 * nside - ring
        ringpix = 4 * nr
        startpix = npix - 2 * nr * (nr + 1)

    if nest:
        startpix = hp.ring2nest(nside, startpix)

    return startpix, ringpix, shifted


def get_ring_pixelIDs(resolution, ring, nest):
    """
    Returns pixel ids of ring ID ring
    Args:
        resolution: int
            resolution of healpix to take sample from
        ring: int
            the ring number (the number of the first ring is 1)
        nest:
            True means nest indexing and false means ring indexing

    Returns:
        pixelIDs: list
            list of pixel IDs in the ring
    """

    # returns ring info in ring scheme:
    nside = hp.order2nside(resolution)
    startpix, ringpix, shifted = get_ring_info_small(resolution, ring, False)
    pixelIDs = np.arange(startpix, startpix+ringpix)
    if nest:
        pixelIDs = hp.ring2nest(nside, pixelIDs)

    return pixelIDs