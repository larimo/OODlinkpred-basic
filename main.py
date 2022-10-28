import argparse, torch, sys, os
import numpy as np
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader
from dataset import Dataset
from embedding import *
from link import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import command, get_identifier
from metrics import mcc, balanced_acc
import hashlib

def main():
    job_start_time = time.time()
    now = datetime.now()
    print("Date: ", now.strftime("%b-%d-%Y %H:%M:%S"))

    parser = argparse.ArgumentParser(description='Sample Code - Out-of-Distribution Link prediction tasks')
    parser.add_argument('--queue', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="rand_sbm_1000")
    parser.add_argument('--node_embedding', type=str, default="GCN") #SVD,GCN
    parser.add_argument('--eval_method', type=str, default="Hits@20") #Hits@K
    parser.add_argument('--device', type=str, default="cuda") #else, CPU
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=70000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--positional', action='store_true') # Force symmetric model (GNN) to be positional
    parser.add_argument('--test_distribution', type=str, default="OOD") # IN_trd, IN_ind, OOD
    parser.add_argument('--sampling_percent', type=float, default=10)
    parser.add_argument('--subsample_method', type=str, default="forestFire") # k_hop
    parser.add_argument('--feature_initialization', type=str, default='deg-inv') # ones
    parser.add_argument('--gincat', action='store_true')
    # Only for random networks
    parser.add_argument('--test_dataset', type=str, default=None) # sbm_10000, "sbm_1000ind"

    args = parser.parse_args()
    file_title, identifier = get_identifier(args)

    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('device = {}'.format(device))

    eval_methods = [args.eval_method]

    data = Dataset(name = args.dataset, device = device, eval_method = eval_methods,
                    test_distribution = args.test_distribution, test_dataset = args.test_dataset,
                    subsample_method = args.subsample_method, feature_initialization = args.feature_initialization,
                    sampling_percentage = args.sampling_percent)

    if args.node_embedding == "GCN":
        if args.test_distribution == "OOD":
            node_embedding = GCN(args.dataset[:4], data.x.size(1), args.hidden_channels, args.hidden_channels, args.num_layers,
                        args.dropout, positional=args.positional, num_nodes=data.num_nodes,
                        testOOD=True, OODnum_nodes=data.OODnum_nodes, node_subset_tensor=data.node_subset_tensor,
                        ind_node_subset_tensor=data.ind_node_subset_tensor).to(device)
        else:
            node_embedding = GCN(args.dataset[:4], data.x.size(1), args.hidden_channels, args.hidden_channels, args.num_layers,
                        args.dropout, positional=args.positional, num_nodes=data.num_nodes,node_subset_tensor=data.node_subset_tensor).to(device)
    else:
        raise Exception("node_embedding not implemented")

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                  args.num_layers, args.dropout).to(device)

    def train(node_embedding, predictor, optimizer):
        node_embedding.train()
        predictor.train()
        total_loss = total_examples = 0

        for perm in DataLoader(range(data.split_edge["train"]["edge"].size(0)), args.batch_size, shuffle=True):
            optimizer.zero_grad()
            h = node_embedding(data.x, data.adj_t)

            edge = data.split_edge["train"]["edge"][perm].t()
            pos_out = predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                                 device=h.device)
            neg_out = predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss = pos_loss + neg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(node_embedding.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

            optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def test(node_embedding, predictor, split, test_distribution):
        node_embedding.eval()
        predictor.eval()
        if split == "test" and test_distribution == "OOD":
            "Evaluating accuracy for original graph as OOD..."
            OODx = data.OODx.to(device)
            OODadj_t = data.OODadj_t.to(device)
            h = node_embedding(OODx, OODadj_t, testOOD=True)
        elif split == "test_ind_sub" and test_distribution == "OOD":
            "Evaluating accuracy on inductive subgraph..."
            sub_INDx = data.INDx.to(device)
            sub_INDadj_t = data.INDadj_t.to(device)
            h = node_embedding(sub_INDx, sub_INDadj_t, testIND=True)
        else:
            h = node_embedding(data.x, data.adj_t)

        pos_edge = data.split_edge[split]['edge']
        neg_edge = data.split_edge[split]['edge_neg']

        pos_preds = []
        for perm in DataLoader(range(pos_edge.size(0)), args.batch_size):
            edge = pos_edge[perm].t()
            pos_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)
        pos_pred_rand = torch.rand(pos_edge.size(0))

        neg_preds = []

        for perm in DataLoader(range(neg_edge.size(0)), args.batch_size):
            edge = neg_edge[perm].t()
            neg_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0)
        neg_pred_rand = torch.rand(neg_edge.size(0))

        result = data.evaluate(pos_pred, neg_pred)
        result_rand = data.evaluate(pos_pred_rand, neg_pred_rand)

        return result, result_rand # lists with results for each metric

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    runtime_list = []
    result_types = ["val", "test_trd", "test_ind", "test"] if args.test_distribution == "OOD" \
                    else ["val", "test"]
    results = {key:{type:[] for type in result_types} for key in eval_methods}
    rand_results = {key:{type:[] for type in result_types} for key in eval_methods}
    num_eval_methods = len(eval_methods)

    for run in range(args.runs):
        start = time.time()
        node_embedding.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
                list(node_embedding.parameters()) +
                list(predictor.parameters()), lr=args.lr)
        use_scheduler = False
        if use_scheduler == True:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience,
                                      min_lr=0.00001)
        best_valid = np.zeros(num_eval_methods)
        final_result = np.zeros(num_eval_methods)
        if args.test_distribution == "OOD":
            final_result_trd = np.zeros(num_eval_methods)
            final_result_ind = np.zeros(num_eval_methods)
            rand_final_result_trd = np.zeros(num_eval_methods)
            rand_final_result_ind = np.zeros(num_eval_methods)

        for epoch in range(1, 1 + args.epochs):
            loss = train(node_embedding, predictor, optimizer)
            valid_result,_= test(node_embedding, predictor, "valid", args.test_distribution)
            test_result, rand_test_result = test(node_embedding, predictor, "test", args.test_distribution)
            if args.test_distribution == "OOD":
                test_result_trd, rand_test_result_trd = test(node_embedding, predictor, "test_trd_sub", args.test_distribution)
                test_result_ind, rand_test_result_ind = test(node_embedding, predictor, "test_ind_sub", args.test_distribution)

            for i in range(num_eval_methods):
                if valid_result[i] > best_valid[i]:
                    best_valid[i] = valid_result[i]
                    final_result[i] = test_result[i]
                    rand_final_result[i] = rand_test_result[i]
                    if args.test_distribution == "OOD":
                        final_result_trd[i] = test_result_trd[i]
                        final_result_ind[i] = test_result_ind[i]
                        # Record random results
                        rand_final_result_trd[i] = rand_test_result_trd[i]
                        rand_final_result_ind[i] = rand_test_result_ind[i]

            # Print results for epoch
            print( f"--- Epoch --- {epoch} \nLoss:\t {loss:.2f} \nValid:\t {['%.2f' % x for x in valid_result]} \nTest:\t {[ '%.2f' % x for x in test_result]} LR:\t {args.lr}")
            if args.test_distribution == "OOD":
                print(f"Test Trd Subgraph:\t {['%.2f' % x for x in test_result_trd]}")
                print(f"Test Ind Subgraph:\t {['%.2f' % x for x in test_result_ind]}")

        # Record results for the run
        for i, method in enumerate(eval_methods):
            results[method]['test'].append(final_result[i])
            results[method]['val'].append(best_valid[i])
            rand_results[method]['test'].append(rand_final_result[i])
            if args.test_distribution == "OOD":
                results[method]['test_trd'].append(final_result_trd[i])
                results[method]['test_ind'].append(final_result_ind[i])
                rand_results[method]['test_trd'].append(rand_final_result_trd[i])
                rand_results[method]['test_ind'].append(rand_final_result_ind[i])

        # Record time taken for the run
        total_time = time.time() - start
        runtime_list.append(total_time)
        print("RUN\t", run)
        print('Time taken:', time.strftime("%H:%M:%S",time.gmtime(total_time)))
        print(results.keys())

    # Print average time
    avg_time = np.array(runtime_list).mean()
    avg_time_str = time.strftime("%H:%M:%S",time.gmtime(avg_time))
    print( f"\nAverage runtime:\t {avg_time_str} with std. {np.array(runtime_list).std():.2f}s")

    print('\n Time taken for job:', time.strftime("%H:%M:%S",time.gmtime(time.time()-job_start_time)))

if __name__ == "__main__":
    #
    main()
