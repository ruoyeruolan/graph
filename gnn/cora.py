# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : cora.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/21 16:17
# @Description: 

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

class GCN(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def main():
    dataset = Planetoid(
        '~/IdeaProjects/Projects/documents/pygeometric/data/planetoid', 
        name='Cora'
    )
    data = dataset[0]

    model = GCN(dataset.num_node_features, dataset.num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1, 101):
        
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)  # type: ignore
        loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])  # type: ignore
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)  # type: ignore
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum() # type: ignore
    acc = int(correct) / int(data.test_mask.sum()) # type: ignore
    print(f'Accuracy: {acc:.4f}')
