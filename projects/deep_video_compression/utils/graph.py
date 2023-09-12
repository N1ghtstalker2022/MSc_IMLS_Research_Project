import numpy as np

def k_hop_nodes(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None, flow='source_to_target'):

    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        dst, org = edge_index
    else:
        org, dst = edge_index

    node_mask = np.empty(num_nodes, dtype=bool)
    edge_mask = np.empty(len(dst), dtype=bool)

    subsets = [np.array([node_idx]).flatten()]
    nodes_at_kth_hop = [None] * num_hops


    for h in range(num_hops):
        node_mask.fill(False)
        node_mask[subsets[-1]] = True
        np.take(node_mask, axis=0, indices=dst, out=edge_mask)
        subsets.append(org[edge_mask])
        nodes_at_kth_hop[h] = list(set(subsets[-1]) - set(np.concatenate(subsets[:-1])))

    # adding the node itself to the set of nodes
    nodes = np.concatenate(([node_idx], np.concatenate(nodes_at_kth_hop)))
    # nodes = np.concatenate(nodes_at_kth_hop)
    node_mask.fill(False)
    node_mask[nodes] = True
    edge_mask = node_mask[org] & node_mask[dst]

    return nodes_at_kth_hop, edge_mask

def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    loop_index = np.arange(0, num_nodes, dtype=edge_index.dtype)
    loop_index = np.tile(loop_index, (2, 1))


    if edge_weight is not None:
        assert edge_weight.size == edge_index.shape[1]
        loop_weight = np.full((num_nodes,), fill_value, dtype=edge_weight.dtype)
        edge_weight = np.concatenate([edge_weight, loop_weight])

    edge_index = np.concatenate([edge_index, loop_index], axis=1)

    return edge_index, edge_weight

def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    row, col = edge_index

    mask = row != col

    if edge_weight is not None:
        assert edge_weight.size == edge_index.shape[1]
        inv_mask = ~mask

        loop_weight = np.full( (num_nodes, ), fill_value, dtype=None if edge_weight is None else edge_weight.dtype)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.size > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = np.concatenate([edge_weight[mask], loop_weight])

    loop_index = np.arange(0, num_nodes, dtype=row.dtype)
    loop_index = np.tile(loop_index, (2,1))
    edge_index = np.concatenate([edge_index[:, mask], loop_index], axis=1)

    return edge_index, edge_weight


def get_laplacian(edge_index, edge_weight=None, normalization=None, dtype=None,
                  num_nodes=None):
    r""" Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        dtype (torch.dtype, optional): The desired images_equirectangular type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    """

    assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = np.ones((edge_index.shape(1),), dtype=dtype)

    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    row, col = edge_index
    # the next two lines are equivalent of deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg = np.zeros((num_nodes,), dtype=edge_weight.dtype)
    np.add.at(deg, row, edge_weight)

    if normalization is None:
        # L = D - A.
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = np.concatenate([-edge_weight, deg], axis=0)
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.

        # deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # L = I - A_norm.
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight,
                                                 fill_value=1,
                                                 num_nodes=num_nodes)
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        edge_weight = deg_inv[row] * edge_weight

        # L = I - A_norm.
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight,
                                                 fill_value=1,
                                                 num_nodes=num_nodes)

    return edge_index, edge_weight

def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


def norm_GCNConv(edge_index, num_nodes=None, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = np.ones((edge_index.shape(1),), dtype=dtype)

    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    # the next two lines are equivalent of deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg = np.zeros((num_nodes,), dtype=edge_weight.dtype)
    np.add.at(deg, row, edge_weight)
    # deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def norm_ChebConv(edge_index, num_nodes=None, edge_weight=None, normalization='sym', lambda_max=None, dtype=None):

    if edge_weight is None:
        edge_weight = np.ones((edge_index.shape(1),), dtype=dtype)

    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    lambda_max = 2.0 if lambda_max is None else lambda_max
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                            normalization, dtype,
                                            num_nodes)

    edge_weight = (2.0 * edge_weight) / lambda_max
    edge_weight[edge_weight == float('inf')] = 0

    edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                             fill_value=-1,
                                             num_nodes=num_nodes)

    return edge_index, edge_weight

def norm_TAGConv(edge_index, num_nodes=None, edge_weight=None, dtype=None):
    if edge_weight is None:
        edge_weight = np.ones((edge_index.shape(1),), dtype=dtype)

    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    row, col = edge_index
    # the next two lines are equivalent of deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg = np.zeros((num_nodes,), dtype=edge_weight.dtype)
    np.add.at(deg, row, edge_weight)
    # deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]