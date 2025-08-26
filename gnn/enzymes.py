# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : enzymes.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/25 14:41
# @Description:

import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from sklearn.model_selection import train_test_split

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, global_mean_pool


def preprocess() -> Tuple[Dataset, DataLoader, DataLoader]:
    dataset = TUDataset(
        "/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/tudatasets",
        name="ENZYMES",
    )

    dataset.shuffle()
    labels = np.array([data.y.item() for data in dataset])
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.1,  # 10% for test set
        stratify=labels,
        random_state=42,
    )

    train_loader = DataLoader(
        dataset=dataset[train_indices], batch_size=32, shuffle=True
    )  # type: ignore
    test_loader = DataLoader(
        dataset=dataset[test_indices], batch_size=32, shuffle=False
    )  # type: ignore
    return dataset, train_loader, test_loader


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layer: str = "GCN") -> None:
        super().__init__()

        torch.manual_seed(42)
        layer_dict = {"GCN": GCNConv, "Graph": GraphConv, "SAGE": SAGEConv}

        self.conv1 = layer_dict[layer](in_channels, 16)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.conv2 = layer_dict[layer](16, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        # self.conv3 = layer_dict[layer](32, 32)
        # self.bn3 = torch.nn.BatchNorm1d(32)

        self.lin = torch.nn.Linear(32, out_features=out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = F.relu(x)

        x = global_mean_pool(x, batch=batch)
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


def model_train(model: GCN, loader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.CrossEntropyLoss()(pred, data.y)
        loss.backward()
        optimizer.step()


def model_test(model: GCN, loader: DataLoader) -> float:
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
        return correct / total


def main(layer: str = "GCN"):
    dataset, train_loader, test_loader = preprocess()

    model = GCN(dataset.num_node_features, dataset.num_classes, layer=layer)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):
        model_train(model, train_loader, optimizer)

        train_acc = model_test(model, train_loader)
        test_acc = model_test(model, test_loader)
        print(f"Epoch: {epoch}, Train Acc: {train_acc: .4f}, Test Acc: {test_acc: .4f}")


if __name__ == "__main__":
    main("Graph")
    main("SAGE")
