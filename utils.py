# The MIT License
#
# Copyright (c) 2016 Thomas Kipf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import random
import pickle as pkl
import sys
import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import math
import torch as th
import json
from utils_layers import *
from numpy import ufunc


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(r"/home/qianyangchao/GCN_NPL/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(r"/home/qianyangchao/GCN_NPL/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum+1e-5, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state.
    """

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    return seed


def get_models(args, nfeat, nclass):
    model_name = args.model
    if model_name == 'GCN':
        model = GCN(in_feats=nfeat,
                    n_hidden=args.num_hidden,
                    n_classes=nclass,
                    n_layers=args.num_layers,
                    activation=F.relu,
                    dropout=args.dropout_rate)
    if model_name == 'GAT':
        model = GAT(num_layers=args.num_layers,
                    in_dim=nfeat,
                    num_hidden=args.num_hidden,
                    num_classes=nclass,
                    heads=([8] * (args.num_layers - 1)) + [1],
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,
                    residual=False)
    elif model_name == 'APPNP':
        model = APPNP(num_layers=args.num_layers,
                      in_feats=nfeat,
                      hiddens=args.num_hidden,
                      n_classes=nclass,
                      activation=F.relu,
                      feat_drop=args.dropout_rate,
                      edge_drop=args.dropout_rate,
                      alpha=0.1,
                      k=10)
    elif model_name == 'GCNII':
        model = GCNII(nfeat=nfeat,
                      nlayers=args.num_layers,
                      nhidden=args.num_hidden,
                      nclass=nclass,
                      dropout=args.dropout_rate,
                      lamda=args.lamda,
                      alpha=0.1,
                      variant=args.variant)

    return model


def load_index(dataset_name, label_shape):
    boundary_nodes_index_path = r"./index/boundary_nodes_index.json"
    with open(boundary_nodes_index_path, 'r') as file_obj:
        boundary_nodes_index = json.load(file_obj)

    boundary_nodes_mask = np.zeros(label_shape)
    boundary_nodes_mask[boundary_nodes_index[dataset_name]] = 1
    boundary_nodes_mask = th.BoolTensor(boundary_nodes_mask)

    return boundary_nodes_mask


def calgain(model, graph, features, train_labels, pseudo_index):
    model.train()
    train_logits = model(graph, features)
    gain = 0
    for i in pseudo_index:
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[i], train_labels[i])
        model.zero_grad()
        train_loss.backward(retain_graph=True)
        norm1 = th.norm(model.layers[0].weight.grad)
        norm2 = th.norm(model.layers[1].weight.grad)
        gain = gain + norm1 + norm2
    print("information gain: ", gain)


def calculate_error_rate(dataset, labels, entropy):
    boundary_nodes_mask = load_index(dataset, labels.shape)
    top_mask = th.zeros(labels.shape, dtype=bool)
    bottom_mask = th.zeros(labels.shape, dtype=bool)
    sorted_index = th.argsort(entropy, dim=0, descending=False)
    top_index = sorted_index[:100]
    top_mask[top_index] = True
    bottom_index = sorted_index[-100:]
    bottom_mask[bottom_index] = True
    top_boundary_size = th.where(th.logical_and(boundary_nodes_mask, top_mask))[0].shape[0]
    bottom_boundary_size = th.where(th.logical_and(boundary_nodes_mask, bottom_mask))[0].shape[0]
    all_boundary_size = th.where(boundary_nodes_mask)[0].shape[0]
    top_ratio = float(top_boundary_size)
    bottom_ratio = float(bottom_boundary_size)
    all_ratio = float(all_boundary_size) / boundary_nodes_mask.shape[0] * 100
    return top_ratio, bottom_ratio, all_ratio
