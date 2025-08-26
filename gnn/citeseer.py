# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : physics.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/23 14:52
# @Description: 


import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module
from torch_geometric.data.data import BaseData, Data

from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv


def preprocess() -> InMemoryDataset:

    dataset = Planetoid(
        '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid', 
        name='CiteSeer', transform= NormalizeFeatures(), force_reload=True
    )
    return dataset

class GCN(Module):

    def __init__(self, in_channels: int, out_channels: int, layer: str = 'GCN') -> None:
        super().__init__()

        layer_dict = {
            'GCN': GCNConv,
            'SAGE': SAGEConv,
            'GAT': GATConv,
            'GraphConv': GraphConv
        }

        torch.manual_seed(42)
        self.conv1 = layer_dict[layer](in_channels, 16)
        self.conv2 = layer_dict[layer](16, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        return x

def model_train(model: GCN, data: Data | BaseData, optimizer: torch.optim.Optimizer) -> torch.Tensor:

    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = torch.nn.CrossEntropyLoss()(pred[data.train_mask], data.y[data.train_mask])  # type: ignore
    loss.backward()
    optimizer.step()
    return loss.item()

def model_test(model: GCN, data: Data) -> None:

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = (pred[mask] == data.y[mask]).sum()  # type: ignore
            acc = int(correct) / int(mask.sum())
            accs.append(acc)
        print(f"Train Accuracy: {accs[0]:.4f}, Val Accuracy: {accs[1]:.4f}, Test Accuracy: {accs[2]:.4f}")

def main():

    dataset = preprocess()
    data = dataset[0]

    model = GCN(dataset.num_node_features, dataset.num_classes, layer='GCN')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):

        loss = model_train(model, data, optimizer)  # type: ignore
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    model_test(model, data) # type: ignore

