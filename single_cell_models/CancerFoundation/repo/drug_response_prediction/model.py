from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch


class UGCNN(nn.Module):
    def __init__(self, hidden_dims: List[int], input_dim: int, out_channels: int):
        super(UGCNN, self).__init__()
        dims = [input_dim] + hidden_dims + [out_channels]
        self.conv_list = nn.ModuleList([])
        for i in range(len(dims)-1):
            self.conv_list.extend(
                [
                    GCNConv(dims[i], dims[i+1])
                ]
            )
        self.norm_list = nn.ModuleList([])
        for i in range(len(dims)-1):
            self.norm_list.append(
                nn.Sequential(
                    BatchNorm(dims[i+1]),
                    nn.ReLU(dims[i+1]),
                    nn.Dropout(p=0.1)
                )
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for (conv, norm) in zip(self.conv_list, self.norm_list):
            x = norm(conv(x, edge_index))

        # Use global mean pooling for the entire graph
        x = global_mean_pool(x, batch)

        return x
        # return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, out_dim: int, input_dim: int, hidden_dim: int = 256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.relu(x)


class CombinedMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(CombinedMLP, self).__init__()
        self.combined_fc1 = nn.Linear(input_dim, 300)
        self.combined_conv1 = nn.Conv2d(1, 30, kernel_size=(1, 150))
        self.combined_conv2 = nn.Conv2d(30, 10, kernel_size=(1, 5))
        self.combined_conv3 = nn.Conv2d(10, 5, kernel_size=(1, 5))
        self.combined_pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.combined_pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.combined_pool3 = nn.MaxPool2d(kernel_size=(1, 3))
        self.final_fc = nn.Linear(30, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.final_dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor):
        x = self.combined_fc1(x)
        x = F.tanh(x)
        x = self.dropout(x)
        x = x.unsqueeze(1).unsqueeze(2)

        x = self.combined_conv1(x)

        x = F.relu(x)
        x = self.combined_pool1(x)
        x = self.combined_conv2(x)
        x = F.relu(x)
        x = self.combined_pool2(x)
        x = self.combined_conv3(x)
        x = F.relu(x)
        x = self.combined_pool3(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.final_dropout(x)

        return self.final_fc(x)
