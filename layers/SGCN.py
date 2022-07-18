import torch
import torch.nn as nn
from torch import Tensor


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


class unit_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 mask_learning=False):
        super(unit_gcn, self).__init__()

        # ==========================================
        # number of nodes
        self.V = 22

        # the adjacency matrixes of the graph
        # self.A = Variable(
        #     A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of input channels
        self.in_channels = in_channels

        # number of output channels
        self.out_channels = out_channels

        # if true, use mask matrix to reweight the adjacency matrix
        self.mask_learning = mask_learning

        # number of adjacency matrix (number of partitions)
        self.num_A = A.size(0)

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn
        # ==========================================

        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1), dtype=torch.float) for i in range(self.num_A)
        ])

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(A.size()))
        if use_local_bn:
            self.bn = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn = nn.BatchNorm2d(self.out_channels, dtype=torch.float)

        self.act = nn.Mish()

        # initialize
        for conv in self.conv_list:
            conv_init(conv)

    def forward(self, x, A):

        x = x.permute(0, 3, 1, 2)

        N, C, T, V = x.size()
        A = A.cuda(x.get_device())

        # reweight adjacency matrix
        if self.mask_learning:
            A = A*self.mask
        # graph convolution
        for i, a in enumerate(A):

            xa = x.reshape(-1, V).mm(a).reshape(N, C, T, V)

            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y = y+self.conv_list[i](xa)

        # batch normalization
        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
            y = self.bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y.clone())

        # nonliner
        y = self.act(y.clone())

        y = y.clone().permute(0, 2, 3, 1)
        return y


class SGCN(nn.Module):
    def __init__(self, features_in, features_out, A) -> None:
        super().__init__()
        default_backbone = [(features_in, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, features_out, 2), (features_out, features_out, 1),(features_out, features_out, 1)]
        # , (128, 256, 2), (256, 256, 1), (256, 256, 1) , (256, 512, 2), (512, 512, 1), (512, 512, 1)
        # default_backbone = [(3,128,1)]
        self.conv_layers = nn.ModuleList([
            # unit_agcn(dim_in, dim_out, A)
            unit_gcn(dim_in, dim_out, A, mask_learning=True)
            for dim_in, dim_out, kernel_size in default_backbone
        ])

    def forward(self, x: Tensor, adjacency_matrix: Tensor) -> torch.Tensor:
        for l in self.conv_layers:
            x = l(x, adjacency_matrix)

        return x
