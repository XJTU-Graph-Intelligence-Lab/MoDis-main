import argparse
import json
import os
import torch as th
import torch.nn.functional as F
import numpy as np
import utils_data
import collections
import time
import random
from utils import get_models, seed_everything


def train(model, graph, features, train_labels, labels, train_mask, val_mask, optimizer):
    model.train()
    train_logits = model(graph, features)
    train_logp = F.log_softmax(train_logits, 1)
    train_loss = F.nll_loss(train_logp[train_mask], train_labels[train_mask])
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with th.no_grad():
        val_logits = model(graph, features)
        val_logp = F.log_softmax(val_logits, 1)
        val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
        val_pred = val_logp.argmax(dim=1)
        val_acc = th.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()
    return val_acc, val_loss


def selftraining(prediction, entropy, num_k, num_class, train_mask, device):
    pseudo_labels = th.argmax(prediction, dim=1)
    n_train_mask = train_mask.clone().to(device)
    for i in range(num_class):
        class_index = th.where(th.logical_and(pseudo_labels == i, ~train_mask))[0]
        sorted_index = th.argsort(entropy[class_index], dim=0, descending=False)
        if sorted_index.shape[0] >= num_k:
            sorted_index = sorted_index[:num_k]
        alternate_index = class_index[sorted_index]
        n_train_mask[alternate_index] = True

    return n_train_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--num_hidden', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=5e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--start_epoch', type=int, default=100)
    parser.add_argument('--end_epoch', type=int, default=200)
    parser.add_argument('--wd1', type=float, default=0.01)
    parser.add_argument('--wd2', type=float, default=5e-4)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--variant', action='store_true', default=False)
    parser.add_argument('--T', type=float, default=0.1)
    parser.add_argument('--num_k', type=int, default=62)
    parser.add_argument('--stage', type=int, default=8)
    parser.add_argument('--train_size', type=int or float, default=None)
    parser.add_argument('--val_size', type=int or float, default=None)
    parser.add_argument('--num_epochs_patience', type=int, default=100)
    parser.add_argument('--num_epochs_max', type=int, default=200)
    parser.add_argument('--run_id', type=str, default='0')

    args = parser.parse_args()
    args.run_id = args.run_id.replace('\r', '')
    vars(args)['name'] = args.model + '_Modis_MS_' + str(args.num_layers) + '_Layers'

    seed_range = [random.randint(1, 65536) for _ in range(10)]
    test_acc = th.zeros(10)
    time_now = time.time()
    for i, seed in enumerate(seed_range):
        seed_everything(seed)
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, adj = \
            utils_data.load_data(args.dataset, args.train_size, args.val_size)
        net = get_models(args, num_features, num_labels)
        if args.model == 'GCNII':
            optimizer = th.optim.Adam([
                {'params': net.params1, 'weight_decay': args.wd1},
                {'params': net.params2, 'weight_decay': args.wd2},
            ], lr=args.learning_rate)
        else:
            optimizer = th.optim.Adam([{'params': net.parameters(), 'weight_decay': args.weight_decay}],
                                      lr=args.learning_rate)

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        net.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        prediction = th.zeros((labels.shape[0], num_labels)).to(device)
        count = args.end_epoch - args.start_epoch
        pre_labels = labels.clone().to(device)
        train_mask_running = train_mask.clone().to(device)

        print("------------------- PreTrain -------------------")
        for stage in range(args.stage):
            net.reset_parameters()
            net.to(device)
            optimizer.state = collections.defaultdict(dict)
            print("------------------- Stage {:02d} -------------------".format(stage))
            for epoch in range(args.end_epoch):
                val_acc, val_loss = train(net, g, features, pre_labels, labels, train_mask_running, val_mask, optimizer)
                print("Epoch {:05d} | Val Loss {:.4f} | Val Acc {:.4f}".format(epoch, val_loss, val_acc))
                if args.start_epoch <= epoch < args.end_epoch:
                    net.eval()
                    with th.no_grad():
                        logits = net(g, features)
                        pre = F.softmax(logits, 1)
                        pre = pre ** (1 / args.T)
                        pre = pre / pre.sum(dim=1, keepdim=True)
                        prediction += pre

            prediction = prediction / count
            pre_labels = prediction.argmax(dim=1)
            pre_labels[train_mask] = labels[train_mask]
            pre_log = th.log(prediction)
            pre_log = th.where(th.isinf(pre_log), th.full_like(pre_log, 0), pre_log)
            pre_entropy = th.sum(th.mul(-prediction, pre_log), dim=1)
            train_mask_running = selftraining(prediction, pre_entropy, args.num_k // args.stage,
                                              num_labels, train_mask_running, device)

        print("------------------- Last Train -------------------")
        net.reset_parameters()
        net.to(device)
        optimizer.state = collections.defaultdict(dict)
        vlss_mn = np.inf
        vacc_mx = 0.0
        state_dict_early_model = None
        for epoch in range(args.num_epochs_max):
            val_acc, val_loss = train(net, g, features, pre_labels, labels, train_mask_running, val_mask, optimizer)
            print("Epoch {:05d} | Val Loss {:.4f} | Val Acc {:.4f}".format(epoch, val_loss, val_acc))
            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    state_dict_early_model = net.state_dict()
                vacc_mx = np.max((val_acc, vacc_mx))
                vlss_mn = np.min((val_loss, vlss_mn))

        net.load_state_dict(state_dict_early_model)
        net.eval()
        with th.no_grad():
            test_logits = net(g, features)
            test_logp = F.log_softmax(test_logits, 1)
            test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
            test_pred = test_logp.argmax(dim=1)
            test_acc[i] = th.eq(test_pred[test_mask], labels[test_mask]).float().mean().item() * 100.0

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc_mean'] = test_acc.mean().item()
    results_dict['test_acc_std'] = test_acc.std(dim=0).item()
    results_dict['total_time'] = time.time() - time_now
    with open(os.path.join('runs', f'{args.name}_{args.dataset}_{args.train_size}_{args.val_size}_'
                                   f'{args.run_id}_results.txt'), 'w') as outfile:
        outfile.write(json.dumps(results_dict) + '\n')


if __name__ == '__main__':
    main()
