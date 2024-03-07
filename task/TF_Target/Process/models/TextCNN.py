# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self):
        datasetpath = './'
        self.model_name = 'TextCNN'
        self.class_list = list(range(0,10))              # 类别名单
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 200                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 30                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.002                                       # 学习率
        self.embed = 300 #self.embedding_pretrained.size(1)\
            #if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 5)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)

'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed),padding=2) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = x.view(x.shape[0],x.shape[1],-1)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # out = self.embedding(x[0])
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
