# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : proteins.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/26 13:40
# @Description:

import torch

from sklearn.model_selection import train_test_split
from gnn.enzymes_ import GINClassification, model_train, model_test

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils.metrics import predict_labels, collect_labels, extract_embeddings
from utils.visualization import draw_random_graph_samples, visualize_embeddings


def preprocess():
    dataset = TUDataset("./datasets/tudatasets", name="PROTEINS")

    train, test = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)
    return dataset, train_loader, test_loader


def main():
    dataset, train_loader, test_loader = preprocess()
    draw_random_graph_samples(dataset, num_classes=dataset.num_classes, name="PROTEINS")

    model = GINClassification(dataset.num_node_features, dataset.num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

    best_acc = 0.0
    best_f1 = 0.0
    # model_ = model
    for epoch in range(1, 101):
        optimizer.zero_grad()
        model_train(model, train_loader, optimizer)

        train_acc, train_f1 = model_test(model, train_loader)
        test_acc, test_f1 = model_test(model, test_loader)

        print(
            f"Epoch: {epoch}, Train Acc: {train_acc: .4f}, Train F1: {train_f1: .4f}, Test Acc: {test_acc: .4f}, Test F1: {test_f1: .4f}"
        )

        if test_acc > best_acc or (test_acc == best_acc and test_f1 > best_f1):
            best_acc = test_acc
            best_f1 = test_f1

            # torch.save(model.state_dict(), "best_proteins_model.pth")
        print(f"Best Test Acc: {best_acc: .4f}, Best Test F1: {best_f1: .4f}")

    embeddings = extract_embeddings(model, dataset)
    labels = collect_labels(test_loader)
    pred = predict_labels(model, test_loader)

    visualize_embeddings(embeddings, labels, pred)


if __name__ == "__main__":
    main()
