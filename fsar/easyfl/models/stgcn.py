import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from easyfl.models.model import BaseModel
from easyfl.models.utils.tgcn import ConvTemporalGraphical
from easyfl.models.utils.graph import Graph


class Model(BaseModel):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.stgcn0 = st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0)
        self.stgcn1 = st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs)
        self.stgcn2 = st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs)
        self.stgcn3 = st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs)
        self.stgcn4 = st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs)
        self.stgcn5 = st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs)
        self.stgcn6 = st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs)
        self.stgcn7 = st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs)
        self.stgcn8 = st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs)
        self.stgcn9 = st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs)

        self.fc = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(10)
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.learnableB = nn.Parameter(copy.deepcopy(A))
        nn.init.constant_(self.learnableB, 1e-6)

        self.unshareC = nn.Parameter()

    def forward(self, x, mode='', start_level=-1, NM_for_train_model=None):

        if mode == 'train':
            if start_level == -1:
                N, C, T, V, M = x.size()
                x = x.permute(0, 4, 3, 1, 2).contiguous()
                x = x.view(N * M, V * C, T)
                x = x.type(torch.FloatTensor).cuda()
                x = self.data_bn(x)
                x = x.view(N, M, V, C, T)
                x = x.permute(0, 1, 3, 4, 2).contiguous()
                x = x.view(N * M, C, T, V)
                start_level += 1
            if start_level == 0:
                x, _ = self.stgcn0(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[0])
                start_level += 1
            if start_level == 1:
                x, _ = self.stgcn1(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[1])
                start_level += 1
            if start_level == 2:
                x, _ = self.stgcn2(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[2])
                start_level += 1
            if start_level == 3:
                x, _ = self.stgcn3(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[3])
                start_level += 1
            if start_level == 4:
                x, _ = self.stgcn4(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[4])
                start_level += 1
            if start_level == 5:
                x, _ = self.stgcn5(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[5])
                start_level += 1
            if start_level == 6:
                x, _ = self.stgcn6(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[6])
                start_level += 1
            if start_level == 7:
                x, _ = self.stgcn7(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[7])
                start_level += 1
            if start_level == 8:
                x, _ = self.stgcn8(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[8])
                start_level += 1
            if start_level == 9:
                x, _ = self.stgcn9(x, (self.A + self.learnableB + self.unshareC) * self.edge_importance[9])
                start_level += 1
            if start_level == 10:
                # global pooling
                N, M = NM_for_train_model[0], NM_for_train_model[1]
                x = F.avg_pool2d(x, x.size()[2:])
                x = x.view(N, M, -1).mean(dim=1)
                # prediction
                x = self.fc(x)
                x = x.view(x.size(0), -1)
            return x

        elif mode == 'global':
            N, C, T, V, M = x.size()
            N_for_train_model, M_for_train_model = N, M
            x = x.permute(0, 4, 3, 1, 2).contiguous()
            x = x.view(N * M, V * C, T)
            x = x.type(torch.FloatTensor).cuda()
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x0 = x.view(N * M, C, T, V)
            #
            x1, _ = self.stgcn0(x0, (self.learnableB) * self.edge_importance[0])
            x2, _ = self.stgcn1(x1, (self.learnableB) * self.edge_importance[1])
            x3, _ = self.stgcn2(x2, (self.learnableB) * self.edge_importance[2])
            x4, _ = self.stgcn3(x3, (self.learnableB) * self.edge_importance[3])
            x5, _ = self.stgcn4(x4, (self.learnableB) * self.edge_importance[4])
            x6, _ = self.stgcn5(x5, (self.learnableB) * self.edge_importance[5])
            x7, _ = self.stgcn6(x6, (self.learnableB) * self.edge_importance[6])
            x8, _ = self.stgcn7(x7, (self.learnableB) * self.edge_importance[7])
            x9, _ = self.stgcn8(x8, (self.learnableB) * self.edge_importance[8])
            x10, _ = self.stgcn9(x9, (self.learnableB) * self.edge_importance[9])
            #
            return [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, [N_for_train_model, M_for_train_model]]

        elif mode == 'test':
            N, C, T, V, M = x.size()
            x = x.permute(0, 4, 3, 1, 2).contiguous()
            x = x.view(N * M, V * C, T)
            x = x.type(torch.FloatTensor).cuda()
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x0 = x.view(N * M, C, T, V)
            #
            x1, _ = self.stgcn0(x0, (self.A + self.learnableB + self.unshareC) * self.edge_importance[0])
            x2, _ = self.stgcn1(x1, (self.A + self.learnableB + self.unshareC) * self.edge_importance[1])
            x3, _ = self.stgcn2(x2, (self.A + self.learnableB + self.unshareC) * self.edge_importance[2])
            x4, _ = self.stgcn3(x3, (self.A + self.learnableB + self.unshareC) * self.edge_importance[3])
            x5, _ = self.stgcn4(x4, (self.A + self.learnableB + self.unshareC) * self.edge_importance[4])
            x6, _ = self.stgcn5(x5, (self.A + self.learnableB + self.unshareC) * self.edge_importance[5])
            x7, _ = self.stgcn6(x6, (self.A + self.learnableB + self.unshareC) * self.edge_importance[6])
            x8, _ = self.stgcn7(x7, (self.A + self.learnableB + self.unshareC) * self.edge_importance[7])
            x9, _ = self.stgcn8(x8, (self.A + self.learnableB + self.unshareC) * self.edge_importance[8])
            x10, _ = self.stgcn9(x9, (self.A + self.learnableB + self.unshareC) * self.edge_importance[9])
            #
            # global pooling
            x = F.avg_pool2d(x10, x10.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)
            # prediction
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            #
            return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A