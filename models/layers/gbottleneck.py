import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.gconv import GConv, LocalGConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
        super(GBottleneck, self).__init__()

        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat, activation=activation)
                           for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        return x_out, x_hidden


class DeformationReasoning(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, adj_mat, sample_coord):
        super(DeformationReasoning, self).__init__()
        self.sample_coord = nn.Parameter(sample_coord, requires_grad=False)
        # TODO only local
        self.conv1 = LocalGConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = LocalGConv(in_features=hidden_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv3 = LocalGConv(in_features=hidden_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv4 = LocalGConv(in_features=hidden_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv5 = LocalGConv(in_features=hidden_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv6 = LocalGConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat, act=lambda x:x)

    def forward(self, inputs_feature):

        x1 = self.conv1(inputs_feature)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2) + x1
        x4 = self.conv4(x3)
        x5 = self.conv5(x4) + x3
        x6 = self.conv6(x5)
        score = F.softmax(10*x6, dim=2)
        delta_coord = torch.sum(score * self.sample_coord, dim=2)
        
        return delta_coord, score, x6
