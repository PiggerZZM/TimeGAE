import torch
import torch as th
from dgl import DGLError
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import init


class TimeGraphConv(nn.Module):
    r"""Graph convolutional layer for TimeGAE from `TimeGAE: A Multivariate Time Series Generation Method based
        on Graph Auto Encoder
     <link>`

    Mathematically it is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)} w_{ji} feat-extract(h_j^{(l)}))

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`w_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    (i.e.,  :math:`w_{ji} = edge_softmax(v_i^{(l)} \cdot v_j^{(l)})`,
            :math: `v_i^{(l)} = W_{left}h_i^{(l)}`,
            :math: `v_j^{(l)} = W_{right}h_j^{(l)}`
    )
    and :math:`\sigma` is an activation function.

    Parameters
    ----------
    seq_len : int
        Input time series seq_len.
    num_sensors : int
        Input time series num_sensors.
    feat_extract : str
        The time series feature extractor applied by TimeGAE.
        Default: ``transformer``.
    emb_size: int
        The embedding size of linear transformation.
        Default: ``16``
    num_heads: int
        When feat_extract is ``transformer``, it represents the number of multi-head attention.
        Default: ``2``
    num_layers: int
        The number of stacked time series feature extractors.
        Default: ``1``
    bidirectional: bool
        When feat_extract is ``rnn``, it refers to whether to use bidirectional rnn.
    dropout_ratio: float
        Dropout ratio of GRU-RNN, transformer.
        Don't use now.
    edge_dropout_ratio: float
        Edge dropout ratio.
        Default: ``0``
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Attributes
    ----------
    bias : torch.Tensor
        The learnable bias tensor.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------

    """

    def __init__(self,
                 seq_len,
                 num_sensors,
                 feat_extract="transformer",
                 emb_size=16,
                 num_heads=2,
                 num_layers=1,
                 bidirectional=False,
                 dropout_ratio=0,
                 edge_dropout_ratio=0,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(TimeGraphConv, self).__init__()
        self._seq_len = seq_len
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_sensors = num_sensors
        self._num_heads = num_heads
        self._bidirectional = bidirectional
        self._num_layers = num_layers
        self._feat_extract = feat_extract
        self._dropout_ratio = dropout_ratio
        self._edge_dropout_ratio = edge_dropout_ratio
        if self._edge_dropout_ratio != 0:
            self._edge_dropout_layer = nn.Dropout(self.edge_dropout_ratio)
        self._emb_size = emb_size
        self.bn = nn.BatchNorm1d(num_sensors * num_sensors)
        self.lookup_l = nn.Sequential(
            nn.Linear(self._seq_len, self._emb_size),
        )
        self.lookup_r = nn.Sequential(
            nn.Linear(self._seq_len, self._emb_size),
        )

        if self._feat_extract == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self._num_sensors, nhead=self._num_heads,
                                                       dropout=0.1)
            self.ts_feat_extractor = nn.TransformerEncoder(encoder_layer, num_layers=self._num_layers)
        elif self._feat_extract == "mlp":
            mlp_blocks = []
            for i in range(self._num_layers):
                mlp_blocks.append(nn.Linear(self._seq_len, self._seq_len))
                mlp_blocks.append(nn.ReLU())
            self.ts_feat_extractor = nn.Sequential(*mlp_blocks)
        elif self._feat_extract == "conv1d":
            conv1d_blocks = []
            for i in range(self._num_layers):
                conv1d_blocks.append(nn.Conv1d(self._num_sensors, self._num_sensors,
                                               kernel_size=3, stride=1, padding=1))
                conv1d_blocks.append(nn.ReLU())
            self.ts_feat_extractor = nn.Sequential(*conv1d_blocks)
        else:
            self.ts_feat_extractor = nn.GRU(self._num_sensors, self._num_sensors, num_layers=self._num_layers,
                                            bidirectional=self._bidirectional,
                                            dropout=self._dropout_ratio)

        if bias:
            self.bias = nn.Parameter(th.Tensor(self._seq_len))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=True):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : bool or torch.Tensor
            If a torch.Tensor is given, it represents the predefined edge weight of shape
            :math:`(num_edges, batch_size, 1)` or :math:`(num_edges, 1, 1)`
            If edge_weight is True, it represents that the model use learnable edge weight.
            If edge_weight is False or None, it represents that the model doesn't use edge weight.

        Returns
        -------
        rst: torch.Tensor
            The output feature
        graph_edge_weight: torch.Tensor or None
            The graph edge weight for visualization

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            feat_src, feat_dst = expand_as_pair(feat, graph)

            # Input shape of RNN and Transformer-Encoder: (seq_len, batch_size, num_sensors)
            # Input shape of Conv1d and MLP: (batch_size, num_sensors, seq_len)
            # feat_src.shape = (num_sensors, batch_size, seq_len)

            graph.srcdata["att_l"] = self.lookup_l(feat_src)
            graph.dstdata["att_r"] = self.lookup_r(feat_src)

            out = feat_src.transpose(0, 2)

            # out.shape = (seq_len, batch_size, num_sensors)
            seq_len, batch_size, num_sensors = out.shape

            if self._feat_extract in ["mlp", "conv1d"]:
                # required shape: (batch_size, num_sensors, seq_len)
                out = out.transpose(0, 1).transpose(1, 2)
                out = self.ts_feat_extractor(out)
                out = out.transpose(1, 2).transpose(0, 1)
            elif self._feat_extract == "rnn":
                out, _ = self.ts_feat_extractor(out)
                if self._bidirectional:
                    out_reshaped = out.view(seq_len, batch_size, 2, num_sensors)
                    out_forward = out_reshaped[:, :, 0, :].view(seq_len, batch_size, 1, num_sensors)
                    out_backward = out_reshaped[:, :, 1, :].view(seq_len, batch_size, 1, num_sensors)
                    out = ((out_forward + out_backward) / 2).squeeze(2)
            else:  # default setting feat_extract == transformer
                out = self.ts_feat_extractor(out)

            out = out.transpose(0, 2)
            # out.shape = (num_sensors, batch_size, seq_len)

            graph.srcdata['h'] = out

            if edge_weight:
                if isinstance(edge_weight, torch.Tensor):
                    # Predefined edge_weight
                    assert edge_weight.shape[0] == graph.number_of_edges()
                    graph.edata['_edge_weight'] = edge_weight
                else:
                    # Learnable edge_weight
                    graph.apply_edges(fn.u_dot_v('att_l', 'att_r', '_edge_weight'))

                if self._edge_dropout_ratio != 0:
                    graph.edata['_edge_weight'] = self.dropout_layer(graph.edata['_edge_weight'])

                graph.edata['_edge_weight_norm'] = edge_softmax(graph, graph.edata['_edge_weight'])
                # TODO: 这里还可以改进
                # 本质上是将某一维度的ts用其他维度线性表示，不应该只用正的权重
                # 用BN计算权重增大了训练难度，而且效果差别不大
                # (num_edges, batch_size, 1)
                # weights = graph.edata['_edge_weight'].transpose(0, 1)
                # norm_weights = self.bn(weights)
                # graph.edata['_edge_weight_norm'] = norm_weights.transpose(0, 1)

                aggregate_fn = fn.u_mul_e('h', '_edge_weight_norm', 'm')
            else:
                aggregate_fn = fn.copy_src('h', 'm')
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='rst'))

            rst = graph.dstdata['rst']

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            graph_edge_weight = None
            if edge_weight:
                graph_edge_weight = graph.edata['_edge_weight_norm']
            return rst, graph_edge_weight

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_seq_len}, out={_seq_len}, num_sensors={_num_sensors}, ' \
                  'feat_extract={_feat_extract}, num_layers={_num_layers}, edge_dropout_ratio={_edge_dropout_ratio}'
        if self._feat_extract == "transformer":
            summary += ', num_heads={_num_heads}'
        elif self._feat_extract == "rnn":
            summary += ', bidirectional={_bidirectional}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
