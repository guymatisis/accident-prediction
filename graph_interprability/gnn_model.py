import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import reset, uniform, zeros
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from typing import Union, Tuple, Callable


###################################################################
                        # BELOW COPIED FROM TRAVLENET
###################################################################                        
# class TRAVELConv(MessagePassing):
#     r"""
#     Args:
#         in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
#             derive the size from the first input(s) to the forward method.
#             A tuple corresponds to the sizes of source and target
#             dimensionalities.
#         out_channels (int): Size of each output sample.
#         nn (torch.nn.Module): Multiple layers of non-linear transformations 
#             that maps feature data of shape :obj:`[-1,
#             num_node_features + num_edge_features]` to shape
#             :obj:`[-1, new_dimension]`, *e.g.*, defined by
#             :class:`torch.nn.Sequential`.
#         aggr (string, optional): The aggregation scheme to use
#             (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
#             (default: :obj:`"add"`)
#         root_weight (bool, optional): If set to :obj:`False`, the layer will
#             not add the transformed root node features to the output.
#             (default: :obj:`True`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, in_channels: Union[int, Tuple[int, int]],
#                  out_channels: int, nn: Callable, aggr: str = 'add',
#                  root_weight: bool = False, bias: bool = True, **kwargs):
#         super(TRAVELConv, self).__init__(aggr=aggr, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.nn = nn
#         self.aggr = aggr

#         if isinstance(in_channels, int):
#             in_channels = (in_channels, in_channels)
            
#         self.in_channels_l = in_channels[0]

#         if root_weight:
#             self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
#         else:
#             self.register_parameter('root', None)

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.nn)
#         if self.root is not None:
#             uniform(self.root.size(0), self.root)
#         zeros(self.bias)


#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, size: Size = None) -> Tensor:
#         if isinstance(x, Tensor):
#             x: OptPairTensor = (x, x)

#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

#         x_r = x[1]
#         # if x_r is not None and self.root is not None:
#         #     out += torch.matmul(x_r, self.root)

#         if self.bias is not None:
#             out += self.bias
#         return out


#     def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
#         inputs = torch.cat([x_j, edge_attr], dim=1)
#         return self.nn(inputs)

#     def __repr__(self):
#         return '{}({}, {}, aggr="{}", nn={})'.format(self.__class__.__name__,
#                                                      self.in_channels,
#                                                      self.out_channels,
#                                                      self.aggr, self.nn)
        
###################################################################
                        # ABOVE COPIED FROM TRAVLENET
###################################################################                        
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class EdgeMLPConv(MessagePassing):
    def __init__(self, node_in_dim, node_out_dim, edge_dim):
        super().__init__(aggr="mean")  # or "add"

        self.mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_dim, node_out_dim),
            nn.ReLU(),
            nn.Linear(node_out_dim, node_out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # IMPORTANT: keyword for edge_index
        x = x.float()
        edge_attr = edge_attr.float()
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: (E, node_in_dim), edge_attr: (E, edge_dim)
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out):
        return aggr_out

class DiffPoolGNN(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim=64, embed_dim=128):
        super().__init__()

        # ----------- Sparse GNN (edge-aware) -----------
        self.gnn1 = EdgeMLPConv(in_node_dim, hidden_dim, in_edge_dim)

        self.gnn2 = EdgeMLPConv(hidden_dim, hidden_dim, in_edge_dim)

        # Assignment network for first pooling
        self.assign_gnn = EdgeMLPConv(hidden_dim, hidden_dim // 4, in_edge_dim)

        # ----------- Dense GNN after pooling -----------
        self.dense_gnn1 = DenseGCNConv(hidden_dim, hidden_dim)
        self.dense_gnn2 = DenseGCNConv(hidden_dim, hidden_dim)

        # Assignment for second pooling (to 1 supernode)
        self.assign_dense = nn.Linear(hidden_dim, 1)

        # ----------- MLP Head -----------
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)     # regression: total accidents
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # --------------------------
        # 1. Sparse GNN + EdgeConv
        # --------------------------
        x = F.relu(self.gnn1(x, edge_index, edge_attr))
        x = F.relu(self.gnn2(x, edge_index, edge_attr))

        # assignment scores for stage 1
        s_raw = self.assign_gnn(x, edge_index, edge_attr)       # (total_nodes, C)
        s_raw = torch.softmax(s_raw, dim=1)

        s, _ = to_dense_batch(s_raw, batch)                     # (B, N, C)

        # --------------------------
        # 2. Sparse -> Dense
        # --------------------------
        adj = to_dense_adj(edge_index, batch=batch)
        x_dense, mask = to_dense_batch(x, batch)

        # --------------------------
        # 3. DiffPool #1
        # --------------------------
        x1, adj1, _, _  = dense_diff_pool(x_dense, adj, s)

        # --------------------------
        # 4. Dense GNN Block (simple)
        # --------------------------
        x2 = F.relu(self.dense_gnn1(x1, adj1))
        x2 = F.relu(self.dense_gnn2(x2, adj1))

        # --------------------------
        # 5. DiffPool #2 (pool to 1)
        # --------------------------
        s2 = torch.softmax(self.assign_dense(x2), dim=1)  # shape B × N1 × 1
        x3, adj3, _, _ = dense_diff_pool(x2, adj1, s2)

        # final embedding (B, hidden_dim)
        g_emb = x3.squeeze(1)

        # --------------------------
        # 6. MLP Head
        # --------------------------
        return self.mlp(g_emb)
