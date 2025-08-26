# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : metrics.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/26 13:33
# @Description:

import torch
import numpy as np


def extract_embeddings(model, dataset):
    model.eval()
    embeddings_list = []
    for data in dataset:
        x, edge_index = data.x, data.edge_index
        with torch.no_grad():
            _, embedding = model(x, edge_index, data.batch)
        embeddings_list.append(embedding.numpy())
    embeddings_all = np.concatenate(embeddings_list)
    return embeddings_all


def predict_labels(model, test_loader):
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, edge_index, _ = data.x, data.edge_index, data.batch
            out, _ = model(x, edge_index, data.batch)
            _, predicted = torch.max(out, 1)
            predicted_labels.extend(predicted.numpy())
    return predicted_labels


def collect_labels(loader):
    # Get true labels for test set
    true_labels = []
    for data in loader:
        true_labels.append(data.y.numpy())
    true_labels = np.concatenate(true_labels)
    return true_labels
