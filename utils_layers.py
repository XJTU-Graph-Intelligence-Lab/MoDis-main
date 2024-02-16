import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from dgl.nn.pytorch import GraphConv, GATConv, GINConv, APPNPConv


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        for i in range(n_layers):
            if i == 0:
                self.layers.append(GraphConv(in_feats, n_hidden))
            elif i == n_layers - 1:
                self.layers.append(GraphConv(n_hidden, n_classes))
            else:
                self.layers.append(GraphConv(n_hidden, n_hidden))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_parameters()

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(g, h)
            h = self.activation(h)
            h = self.dropout(h)
        h = self.layers[-1](g, h)
        return h


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_class):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_class, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        logits = self.linear(F.relu(x))
        return logits


class M3S(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.encoder = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        self.classfier = Classifier(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classfier.reset_parameters()

    def forward(self, g, x):
        h = self.encoder(g, x)
        logits = self.classfier(h)
        return h, logits


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        # self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()

    def forward(self, x):
        h = x
        h = F.relu(self.linears[0](h))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation, dropout):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activation = activation
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=True)
            )  # set to True if learning epsilon
            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.ginlayers:
            layer.apply_func.reset_parameters()
            layer.eps = th.nn.Parameter(th.FloatTensor([0]))
        self.fc.reset_parameters()

    def forward(self, g, x):
        # list of hidden representation at each layer (including the input layer)
        h = x
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            # h = self.batch_norms[i](h)
            h = self.activation(h)
            h = self.drop(h)
        h = self.fc(h)
        return h


# GAT模型实现
class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop,
                 negative_slope, residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(GATConv(
                    in_dim, num_hidden, heads[0],
                    feat_drop, attn_drop, negative_slope, False, self.activation))
            elif i == num_layers - 1:
                self.gat_layers.append(GATConv(
                    num_hidden * heads[-2], num_classes, heads[-1],
                    feat_drop, attn_drop, negative_slope, residual, None))
            else:
                self.gat_layers.append(GATConv(
                    num_hidden * heads[i - 1], num_hidden, heads[i],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

    def forward(self, g, inputs):
        h = inputs
        for i, gat_layer in enumerate(self.gat_layers[:-1]):
            h = gat_layer(g, h)
            h = h.flatten(1)
        h = self.gat_layers[-1](g, h)
        h = h.mean(1)
        return h


# APPNP模型实现
class APPNP(nn.Module):
    def __init__(self, num_layers, in_feats, hiddens, n_classes, activation, feat_drop, edge_drop, alpha, k):
        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_feats, hiddens))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hiddens, n_classes))
            else:
                self.layers.append(nn.Linear(hiddens, hiddens))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.layer = GraphConv(in_features, in_features, weight=False, bias=False)
        self.weight = nn.Parameter(th.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, g, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = self.layer(g, input)
        if self.variant:
            support = th.cat([hi, h0], 1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*th.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()

        for layer in self.fcs:
            layer.reset_parameters()

    def forward(self, g, x):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, g, _layers[0], self.lamda, self.alpha, i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner
