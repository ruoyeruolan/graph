# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : mutag.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/23 19:12
# @Description:

import torch
import torch.nn.functional as F

from typing import Tuple
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GraphConv, global_add_pool


def preprocess() -> Tuple[Dataset, DataLoader, DataLoader]:
    dataset = TUDataset(
        "/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/tudatasets",
        name="MUTAG",
    )

    train, test = dataset[:150], dataset[150:]
    train_loader: DataLoader = DataLoader(dataset=train, batch_size=32, shuffle=True)  # type: ignore
    test_loader: DataLoader = DataLoader(dataset=test, batch_size=32, shuffle=False)  # type: ignore
    return dataset, train_loader, test_loader


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layer: str = "GCN") -> None:
        super().__init__()

        torch.manual_seed(42)
        if layer == "GCN":
            self.conv1 = GCNConv(in_channels, 16)
            self.conv2 = GCNConv(16, 32)
            self.conv3 = GCNConv(32, 32)
        elif layer == "Graph":
            self.conv1 = GraphConv(in_channels, 16)
            self.conv2 = GraphConv(16, 32)
            self.conv3 = GraphConv(32, 32)
        else:
            raise ValueError(f"Unknown layer type: {layer}")

        self.lin = torch.nn.Linear(32, out_features=out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_add_pool(x, batch=batch)

        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


def model_train(model: GCN, loader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.CrossEntropyLoss()(pred, data.y)
        loss.backward()
        optimizer.step()


def model_test(model: GCN, loader: DataLoader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs

    return correct / total if total > 0 else 0.0


def main(layer: str = "GCN"):
    dataset, train_loader, test_loader = preprocess()

    model = GCN(dataset.num_node_features, dataset.num_classes, layer=layer)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):
        model_train(model, train_loader, optimizer)
        train_acc = model_test(model, train_loader)
        test_acc = model_test(model, test_loader)
        print(f"Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()  # Epoch: 100, Train Acc: 0.8000, Test Acc: 0.7895
    main(layer="Graph")  # Epoch: 100, Train Acc: 0.9333, Test Acc: 0.8421
