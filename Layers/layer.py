import math
import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_cheby = 1
        self.weight = Parameter(torch.FloatTensor(in_features * self.k_cheby, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        Xt = chebyshev(adj, input, self.k_cheby)
        support = torch.mm(Xt, self.weight)
        #output = support
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def chebyshev(L, X, K):
    M, N = X.shape
    Xt = torch.empty((K, M, N))
    Xt[0, ...] = X
    if K > 1:
        Xt[1, ...] = torch.mm(L, X)
    for k in range(2, K):
        Xt[k, ...] = 2 * torch.mm(L, Xt[k-1, ...]) - Xt[k-2, ...]    
    Xt = Xt.permute(1,2,0)
    Xt = torch.reshape(Xt, [M, N*K])
    return Xt