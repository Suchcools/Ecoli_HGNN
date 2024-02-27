import torch
import torch.nn as nn
import torch.nn.functional as F
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.5, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 可学习参数 W，用于线性变换输入特征
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 可学习参数 a，用于计算注意力系数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Leaky ReLU 激活函数，用于引入非线性
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # 线性变换，得到节点特征表示，形状为 [N, out_features]
        N = h.size()[0] # 图的节点数

        # 构造注意力机制的输入，a_input 的形状为 [N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) # 5 5 12

        # 计算注意力系数，通过学习得到，形状为 [N, N]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 使用零向量对不相邻节点的注意力系数进行屏蔽
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 对注意力系数进行 softmax 归一化，然后使用 dropout 进行正则化
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 使用注意力系数对节点特征进行加权求和，得到更新后的节点表示 h_prime
        h_prime = torch.matmul(attention, h)

        # 如果设置了 concat 为 True，则在更新后的节点表示上使用激活函数 ELU，并返回
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
layer = GraphAttentionLayer(in_features=10, out_features=6)

adjacency_matrix = [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
]

# 转换为PyTorch张量
adj = torch.tensor(adjacency_matrix, dtype=torch.float32)
feat = torch.Tensor(5,10)

layer(feat,adj)