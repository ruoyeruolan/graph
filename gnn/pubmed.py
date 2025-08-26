# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : pubmed.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/22 15:03
# @Description: 

import torch
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import ClusterData, ClusterLoader

from typing import cast

class GCN(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, layer: str = 'GCN') -> None:
        super().__init__()

        print(f'Using {layer} layer for GCN model.')
        torch.manual_seed(12345)
        if layer == 'GCN':
            self.conv1 = GCNConv(in_channels, 16)
            self.conv2 = GCNConv(16, out_channels)
        elif layer == 'GAT':
            self.conv1 = GATConv(in_channels, 16)
            self.conv2 = GATConv(16, out_channels)
        else:
            raise ValueError(f'Layer {layer} not supported.')
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def main():

    dataset = Planetoid(
        '~/IdeaProjects/Projects/documents/pygeometric/data/planetoid', 
        name='PubMed', transform=NormalizeFeatures(), force_reload=True
    )
    # len(dataset[0])
    # Data(x=[19717, 500], edge_index=[2, 88648], y=[19717], train_mask=[19717], val_mask=[19717], test_mask=[19717])

    data_: Data = cast(Data, dataset[0])

    torch.manual_seed(12345)
    data = ClusterData(dataset[0], num_parts=32)
    loader = ClusterLoader(data, batch_size=32, shuffle=True)

    model = GCN(dataset.num_node_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):
        for batch in loader:

            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = torch.nn.CrossEntropyLoss()(pred[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step() 

            correct = pred[batch.test_mask].argmax(dim=1) == batch.y[batch.test_mask]
            acc = correct.sum().item() / batch.test_mask.sum().item()
            print(f'Epoch: {epoch}, Loss: {loss.item(): .4f}, Accuracy: {acc: .4f}')
    
    model.eval()
    with torch.no_grad():
        pred = model(data_.x, data_.edge_index).argmax(dim=1)
        correct = (pred[data_.test_mask] == data_.y[data_.test_mask]).sum()  # type: ignore
        acc = correct.item() / data_.test_mask.sum().item()
        print(f'Test Accuracy: {acc: .4f}')
    
    # No ClusterData
    model.train()
    for epoch in range(1, 101):

        optimizer.zero_grad()
        pred = model(data_.x, data_.edge_index)
        loss = torch.nn.CrossEntropyLoss()(pred[data_.train_mask], data_.y[data_.train_mask])  # type: ignore
        loss.backward()
        optimizer.step()

        correct = pred[data_.train_mask].argmax(dim=1) == data_.y[data_.train_mask] # type: ignore
        acc = correct.sum().item() / data_.train_mask.sum().item()
        print(f'Epoch: {epoch}, Loss: {loss.item(): .4f}, Accuracy: {acc: .4f}')

    model.eval()
    with torch.no_grad():
        pred = model(data_.x, data_.edge_index).argmax(dim=1)
        correct = (pred[data_.test_mask] == data_.y[data_.test_mask]).sum() # type: ignore
        acc = correct.item() / data_.test_mask.sum().item()
        print(f'Test Accuracy: {acc: .4f}')
