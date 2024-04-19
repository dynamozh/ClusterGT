import os

import torch
import torch.nn.functional as F
from collator import collator
import random
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from model import GT
from lr import PolynomialDecayLR
import argparse
import utils
from tqdm import tqdm
import scipy.sparse as sp
from torch_geometric.loader import ClusterData
from torch_geometric.datasets import Planetoid, Amazon, Actor, Coauthor
import torch_geometric.transforms as T
import time
from gtrick.pyg import LabelPropagation, CorrectAndSmooth


def train(args, model, device, loader, optimizer):
    model.train()

    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        pred = model(batch)
        y_true = batch.y.view(-1)
        loss = F.nll_loss(pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_train(args, model, device, loader):
    y_true = []
    y_pred = []
    loss_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            pred = model(batch)
            loss_list.append(F.nll_loss(pred, batch.y.view(-1)).item())
            y_true.append(batch.y)
            y_pred.append(pred.argmax(1))
            pred_list.append(pred)

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    out = torch.cat(pred_list)
    correct = (y_pred == y_true).sum()
    acc = correct.item() / len(y_true)

    return acc, np.mean(loss_list), out


def eval(args, model, device, loader):
    y_true = []
    y_pred = []
    loss_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            pred = model(batch)
            loss_list.append(F.nll_loss(pred, batch.y.view(-1)).item())
            y_true.append(batch.y)
            y_pred.append(pred.argmax(1))
            pred_list.append(pred)

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    out = torch.cat(pred_list)

    pred_list = []
    for i in torch.split(y_pred, args.num_data_augment, dim=0):
        pred_list.append(i.bincount().argmax().unsqueeze(0))
    y_pred = torch.cat(pred_list)
    y_true = y_true.view(-1, args.num_data_augment)[:, 0]
    correct = (y_pred == y_true).sum()
    acc = correct.item() / len(y_true)

    return acc, np.mean(loss_list), out


def random_split(data_list, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(len(data_list))
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * len(data_list))]
    val_idx = all_idx[int(frac_train * len(data_list)):int((frac_train + frac_valid) * len(data_list))]
    test_idx = all_idx[int((frac_train + frac_valid) * len(data_list)):]
    train_list = []
    test_list = []
    val_list = []
    for i in train_idx:
        train_list.append(data_list[i])
    for i in val_idx:
        val_list.append(data_list[i])
    for i in test_idx:
        test_list.append(data_list[i])
    return train_list, val_list, test_list


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph transformer')
    parser.add_argument('--dataset_name', type=str, default='CS')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=6)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--peak_lr', type=float, default=1e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_data_augment', type=int, default=1)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for dataset loading')
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--perturb_feature', type=bool, default=False)
    parser.add_argument('--num_parts', type=int, default=150)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--walk_length', type=int, default=10)
    # params for Label Propagation
    parser.add_argument('--lp_layers', type=int, default=50)
    parser.add_argument('--lp_alpha', type=float, default=0.9)
    # params for C & S
    parser.add_argument('--num-correction-layers', type=int, default=50)
    parser.add_argument('--correction-alpha', type=float, default=0.979)
    parser.add_argument('--num-smoothing-layers', type=int, default=50)
    parser.add_argument('--smoothing-alpha', type=float, default=0.756)
    parser.add_argument('--autoscale', action='store_true', default=True)
    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    dataset = Coauthor(root='./data/', name=args.dataset_name)
    transform1 = T.AddRandomWalkPE(args.walk_length)
    graph = transform1(dataset[0])

    # Use METIS for clustering
    cluster_data = ClusterData(graph, num_parts=args.num_parts, recursive=False)
    cluster_data_ids = np.arange(len(cluster_data.data.y))
    cluster_data_adj_t = utils.edge_index_to_adj_t(cluster_data.get_edge_index(), cluster_data.data.num_nodes)

    # preprocess
    print("Preprocessing ……")
    all_partition_mean_feature_list = []
    all_partition_power_adj_list = []
    transform2 = T.VirtualNode()
    for partition_idx in range(len(cluster_data.partition.partptr) - 1):
        # Append the mean features to the list
        # mean_feature = utils.calculate_partition_mean_feature(partition_idx, cluster_data)
        # all_partition_mean_feature_list.append(mean_feature)

        # Append the mean features to the list(SGC conv + Virtual Node)
        partition_data = cluster_data.__getitem__(partition_idx)
        # 添加Virtual Node
        partition_data = transform2(partition_data)
        adj = sp.coo_matrix(
            (np.ones(partition_data.edge_index.shape[1]), (partition_data.edge_index[0], partition_data.edge_index[1])),
            shape=(partition_data.y.shape[0], partition_data.y.shape[0]), dtype=np.float32)
        normalized_adj = utils.adj_normalize(adj)
        normalized_adj = utils.sparse_mx_to_torch_sparse_tensor(normalized_adj.tocoo()).float()
        features, precompute_time = utils.sgc_precompute(partition_data.x, normalized_adj, args.k)
        # print("{:.4f}s".format(precompute_time))
        all_partition_mean_feature_list.append(features[-1])

        # Append power_adj_list to the list
        # partition_data = cluster_data.__getitem__(partition_idx)
        # adj = sp.coo_matrix((np.ones(partition_data.edge_index.shape[1]), (partition_data.edge_index[0], partition_data.edge_index[1])),
        #                     shape=(partition_data.y.shape[0], partition_data.y.shape[0]), dtype=np.float32)
        # normalized_adj = utils.adj_normalize(adj)
        # power_adj_list = [normalized_adj]
        # for m in range(5):
        #     power_adj_list.append(power_adj_list[0] * power_adj_list[m])
        # all_partition_power_adj_list.append(power_adj_list)
    cluster_data.mean_feature_list = all_partition_mean_feature_list

    if not os.path.exists('./dataset/' + args.dataset_name):
        # 如果目录不存在，则创建目录
        os.makedirs('./dataset/' + args.dataset_name)

        # select
        print("selecting")
        select_data_list = []
        start_time = time.time()
        for id in cluster_data_ids:
            x_intra_ids, x_inter_ids = utils.optimized_concatenate_ids(id, cluster_data, all_partition_power_adj_list)
            x_ids = []
            x_ids.append(x_intra_ids)
            x_ids.append(x_inter_ids)
            # x_ids.append(attn_bias)
            select_data_list.append(x_ids)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"select took {elapsed_time:.2f} seconds to run.")
        torch.save(select_data_list, './dataset/' + args.dataset_name + '/data.pt')
    else:
        # 检查文件是否存在
        if not os.path.isfile('./dataset/' + args.dataset_name + '/data.pt'):
            # select
            print("selecting")
            select_data_list = []
            start_time = time.time()
            for id in cluster_data_ids:
                x_intra_ids, x_inter_ids = utils.optimized_concatenate_ids(id, cluster_data,
                                                                           all_partition_power_adj_list)
                x_ids = []
                x_ids.append(x_intra_ids)
                x_ids.append(x_inter_ids)
                # x_ids.append(attn_bias)
                select_data_list.append(x_ids)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"select took {elapsed_time:.2f} seconds to run.")
            torch.save(select_data_list, './dataset/' + args.dataset_name + '/data.pt')

    data_list = torch.load('./dataset/' + args.dataset_name + '/data.pt')
    print("cached select data loaded")

    train_dataset_ids, test_dataset_ids, valid_dataset_ids = random_split(data_list, frac_train=0.6, frac_valid=0.2,
                                                                          frac_test=0.2, seed=args.seed)
    print('dataset load successfully')
    train_loader = DataLoader(train_dataset_ids, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=partial(collator, data=cluster_data, shuffle=True,
                                                 perturb=args.perturb_feature))
    val_loader = DataLoader(valid_dataset_ids, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=partial(collator, data=cluster_data, shuffle=False))
    test_loader = DataLoader(test_dataset_ids, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=partial(collator, data=cluster_data, shuffle=False))
    print(args)

    model = GT(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=cluster_data.data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=cluster_data.data.y.max().item() + 1,
        attn_bias_dim=args.attn_bias_dim,
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        ffn_dim=args.ffn_dim,
        num_global_node=args.num_global_node
    )
    if not args.test and not args.validate:
        print(model)
    print('Total params:', sum(p.numel() for p in model.parameters()))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup=args.warmup_epochs,
        tot=args.epochs,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0)

    val_acc_list, test_acc_list = [], []
    best_test_acc, best_val_acc = 0, 0
    best_train_out = None
    best_val_out = None
    best_test_out = None
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train(args, model, device, train_loader, optimizer)
        lr_scheduler.step()

        print("====Evaluation")
        train_acc, train_loss, train_out = eval_train(args, model, device, train_loader)

        val_acc, val_loss, val_out = eval(args, model, device, val_loader)
        test_acc, test_loss, test_out = eval(args, model, device, test_loader)

        print("train_acc: %f val_acc: %f test_acc: %f" % (train_acc, val_acc, test_acc))
        print("train_loss: %f val_loss: %f test_loss: %f" % (train_loss, val_loss, test_loss))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_train_out = train_out
            best_val_out = val_out
            best_test_out = test_out

        all_partition_mean_feature_list = []
        all_partition_power_adj_list = []
        for partition_idx in range(len(cluster_data.partition.partptr) - 1):
            # Append the mean features to the list
            # mean_feature = utils.calculate_partition_mean_feature(partition_idx, cluster_data)
            # all_partition_mean_feature_list.append(mean_feature)

            # Append the mean features to the list(SGC conv + Virtual Node)
            partition_data = cluster_data.__getitem__(partition_idx)
            # 添加Virtual Node
            partition_data = transform2(partition_data)
            adj = sp.coo_matrix((np.ones(partition_data.edge_index.shape[1]),
                                 (partition_data.edge_index[0], partition_data.edge_index[1])),
                                shape=(partition_data.y.shape[0], partition_data.y.shape[0]), dtype=np.float32)
            normalized_adj = utils.adj_normalize(adj)
            normalized_adj = utils.sparse_mx_to_torch_sparse_tensor(normalized_adj.tocoo()).float()
            features, precompute_time = utils.sgc_precompute(partition_data.x, normalized_adj, args.k)
            all_partition_mean_feature_list.append(features[-1])

            # Append power_adj_list to the list
            # partition_data = cluster_data.__getitem__(partition_idx)
            # adj = sp.coo_matrix((np.ones(partition_data.edge_index.shape[1]), (partition_data.edge_index[0], partition_data.edge_index[1])),
            #                     shape=(partition_data.y.shape[0], partition_data.y.shape[0]), dtype=np.float32)
            # normalized_adj = utils.adj_normalize(adj)
            # power_adj_list = [normalized_adj]
            # for m in range(5):
            #     power_adj_list.append(power_adj_list[0] * power_adj_list[m])
            # all_partition_power_adj_list.append(power_adj_list)
        cluster_data.mean_feature_list = all_partition_mean_feature_list

    # use LabelPropagation to predict test labels and ensemble results
    lp = LabelPropagation(args.lp_layers, args.lp_alpha)
    train_idx = []
    test_idx = []
    for train_dataset_id in train_dataset_ids:
        train_idx.append(train_dataset_id[0][0])
    for test_dataset_id in test_dataset_ids:
        test_idx.append(test_dataset_id[0][0])
    train_idx_tensor = torch.tensor(train_idx)
    test_idx_tensor = torch.tensor(test_idx)
    yh = lp(cluster_data.data.y, cluster_data_adj_t, mask=train_idx_tensor)
    yh = yh[test_idx_tensor]

    y_pred = torch.argmax(best_test_out.to(device) + torch.log(yh).to(device), dim=1)
    y_true = cluster_data.data.y.to(device)
    y_true = y_true[test_idx_tensor]
    test_correct = (y_pred == y_true).sum()
    test_acc_lp = test_correct.item() / len(y_true)

    # define c & s
    # cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
    #                       correction_alpha=args.correction_alpha,
    #                       num_smoothing_layers=args.num_smoothing_layers,
    #                       smoothing_alpha=args.smoothing_alpha,
    #                       autoscale=args.autoscale)
    # # use labels of train and valid set to propagate
    # train_idx = []
    # valid_idx = []
    # test_idx = []
    # for train_dataset_id in train_dataset_ids:
    #     train_idx.append(train_dataset_id[0][0])
    # for valid_dataset_id in valid_dataset_ids:
    #     valid_idx.append(valid_dataset_id[0][0])
    # for test_dataset_id in test_dataset_ids:
    #     test_idx.append(test_dataset_id[0][0])
    # train_idx_tensor = torch.tensor(train_idx).to(device)
    # valid_idx_tensor = torch.tensor(valid_idx).to(device)
    # test_idx_tensor = torch.tensor(test_idx).to(device)
    # best_out = torch.zeros((len(train_idx) + len(valid_idx) + len(test_idx), best_train_out.size(1)), dtype=best_train_out.dtype).to(device)
    # best_out[train_idx_tensor] = best_train_out
    # best_out[valid_idx_tensor] = best_val_out
    # best_out[test_idx_tensor] = best_test_out
    # mask_idx = torch.cat((train_idx_tensor, valid_idx_tensor)).to(device)
    # cluster_data.data.y = cluster_data.data.y.to(device)
    # cluster_data_adj_t = cluster_data_adj_t.to(device)
    # y_soft = cs.correct(torch.exp(best_out), cluster_data.data.y[mask_idx], mask_idx, cluster_data_adj_t)
    # y_soft = cs.smooth(y_soft, cluster_data.data.y[mask_idx], mask_idx, cluster_data_adj_t)
    #
    # y_pred = y_soft.argmax(1)
    #
    # y_true = cluster_data.data.y.to(device)
    # y_pred = y_pred[test_idx_tensor]
    # y_true = y_true[test_idx_tensor]
    # test_correct = (y_pred == y_true).sum()
    # test_acc_lp = test_correct.item() / len(y_true)

    print('best validation acc: ', max(val_acc_list))
    print('best test acc: ', max(test_acc_list))
    print('best acc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
    print('best acc_lp: ', test_acc_lp)
    print('best index: ', val_acc_list.index(max(val_acc_list)))
    # test_acc_list.append(test_acc_list[val_acc_list.index(max(val_acc_list))])
    # np.save(
    #     './exps/' + args.dataset_name + '/test_acc_list_' + str(args.batch_size) + '_' + str(args.hidden_dim) + '_' + str(args.n_layers) + '_' + str(args.num_parts) + '_' + str(args.seed),
    #     np.array(test_acc_list))
    # np.save(
    #     './exps/' + args.dataset_name + '/val_acc_list_' + str(args.batch_size) + '_' + str(args.hidden_dim) + '_' + str(args.n_layers) + '_' + str(args.num_parts) + '_' + str(args.seed),
    #     np.array(val_acc_list))


if __name__ == "__main__":
    main()