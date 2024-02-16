import numpy as np
import torch as th
import random
import networkx as nx
from dgl import AddSelfLoop
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, AmazonCoBuyComputerDataset, \
    AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset, CoraFullDataset


def load_data(dataset_name, train_percentage=None, val_percentage=None):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    transform = (
        AddSelfLoop()
    )
    if dataset_name == 'cora':
        data = CoraGraphDataset(transform=transform)
    elif dataset_name == 'citeseer':
        data = CiteseerGraphDataset(transform=transform)
    elif dataset_name == 'pubmed':
        data = PubmedGraphDataset(transform=transform)
    elif dataset_name == 'AmazonCoBuyComputer':
        data = AmazonCoBuyComputerDataset(transform=transform)
    elif dataset_name == 'AmazonCoBuyPhoto':
        data = AmazonCoBuyPhotoDataset(transform=transform)
    elif dataset_name == 'CoauthorCS':
        data = CoauthorCSDataset(transform=transform)
    elif dataset_name == 'CoauthorPhysics':
        data = CoauthorPhysicsDataset(transform=transform)
    elif dataset_name == 'CoraFull':
        data = CoraFullDataset(transform=transform)
    g = data[0].to(device)
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    num_features = features.shape[1]
    num_nodes = features.shape[0]
    num_labels = th.unique(labels).shape[0]
    train_mask, val_mask, test_mask = th.zeros(num_nodes, dtype=bool), \
                                      th.zeros(num_nodes, dtype=bool), th.zeros(num_nodes, dtype=bool)
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        train_mask = g.ndata['train_mask'].bool().to(device)
        val_mask = g.ndata['val_mask'].bool().to(device)
        test_mask = g.ndata['test_mask'].bool().to(device)
    elif dataset_name == 'CoraFull':
        train_mask, val_mask, test_mask = generate_mask(labels, train_percentage, val_percentage)
    else:
        for i in range(num_labels):
            class_index = th.where(labels == i)[0].tolist()
            class_index = random.sample(class_index, 40)
            train_index = class_index[:20]
            val_index = class_index[20:]
            train_mask[train_index] = True
            val_mask[val_index] = True
        left_index = th.where(~th.logical_or(train_mask, val_mask))[0]
        test_mask[left_index[:1000]] = True

    nxg = g.cpu().to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)

    if dataset_name != 'CoraFull':
        if train_percentage and val_percentage:
            train_index = th.where(train_mask)[0]
            train_mask = train_mask.clone()
            train_mask[:] = False
            label = labels[train_index]
            for i in range(num_labels):
                class_index = th.where(label == i)[0].tolist()
                class_index = random.sample(class_index, train_percentage)
                train_mask[train_index[class_index]] = True

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, adj


def generate_mask(labels, train_size, val_size):
    if not train_size and not val_size:
        train_size, val_size = 20, 20
    num_nodes = labels.shape[0]
    num_labels = th.unique(labels).shape[0]
    train_mask, val_mask, test_mask = th.zeros(num_nodes, dtype=bool), \
                                      th.zeros(num_nodes, dtype=bool), th.zeros(num_nodes, dtype=bool)
    for i in range(num_labels):
        class_index = th.where(labels == i)[0].tolist()
        if len(class_index) < train_size+val_size:
            train_size = min(len(class_index), train_size)
            train_index = random.sample(class_index, train_size)
            train_mask[train_index] = True
            left_index = th.where(~th.logical_or(train_mask, val_mask))[0].tolist()
            val_index = random.sample(left_index, val_size)
            val_mask[val_index] = True
        else:
            class_index = random.sample(class_index, train_size+val_size)
            train_index = class_index[:train_size]
            val_index = class_index[train_size:]
            train_mask[train_index] = True
            val_mask[val_index] = True
    left_index = th.where(~th.logical_or(train_mask, val_mask))[0].tolist()
    test_index = random.sample(left_index, 1000)
    test_mask[test_index] = True
    return train_mask, val_mask, test_mask
