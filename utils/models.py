# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : models.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/26 14:49
# @Description:

import torch
import torch.nn as nn

from typing import Tuple
from torch_geometric.nn import GINConv, global_mean_pool


class GINClassification(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
        )

        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
        )

        self.fc = nn.Linear(32, out_channels)

    def forward(self, x, edge_index, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        classification = self.fc(x)
        return classification, x
