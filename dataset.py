import torch,sys
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree
# from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from pprint import pprint
import Graph_Sampling
from torch_geometric.utils import to_networkx, from_networkx, negative_sampling, to_undirected, coalesce
from metrics import mcc, balanced_acc

class Dataset:
    def __init__(self, name, device, eval_method, test_distribution,
    test_dataset=None, subsample_method, sampling_percentage=10, feature_initialization='deg-inv'):
        self.name = name
        self.device = device
        self.eval_method = eval_method
        self.test_distribution = test_distribution
        self.OODdataset = test_dataset
        self.subsample_method = subsample_method
        self.sampling_percentage = sampling_percentage
        self.node_subset_tensor = None
        self.ind_node_subset_tensor = None
        self.ft_init = feature_initialization

        print("\n======= Running dataset.py... =====")
        print(f"\nTrain Dataset: {self.name}")
        print(f'OOD Test Dataset: {self.OODdataset}')
        if self.name[:4] == "rand":
            print("\n == Building random dataset... ==")
            self.build_random()
        else:
            raise Exception("dataset not implemented")

        self.move_to_device()

        # Reducing number of samples in split edge for OOD test
        print('Matching number of edges in OOD test to validation...')
        len_val_edge = len(self.split_edge["valid"]["edge"])
        len_val_edge_neg = len(self.split_edge["valid"]["edge_neg"])
        len_test_edge = len(self.split_edge["test"]["edge"])
        len_test_edge_neg = len(self.split_edge["test"]["edge_neg"])
        idx = torch.randperm(len_test_edge)[:len_val_edge]
        idx_neg = torch.randperm(len_test_edge_neg)[:len_val_edge_neg]
        self.split_edge["test"] = {"edge": self.split_edge["test"]["edge"][idx],
                                "edge_neg": self.split_edge["test"]["edge_neg"][idx_neg]}

        self.print_split_edge()
        print('============================')

    def build_random(self):
        print(f'Processing training and validation datasets...')

        # Define *undirected* edge_index and num nodes
        edge_index_t, self.num_nodes = self.load_edge_index(self.name)
        print(f'Shape of edge index for training: {edge_index_t.shape}')
        print(f'Number of nodes in training graph: {self.num_nodes}')

        # Load node block_id for SBM
        block_id_df = torch.load('./dataset/{}_blockID.pt'.format(self.name)) \
                     if self.name[5:8] == "sbm" else None

        # Define splits for training and validation
        data = Data(edge_index=edge_index_t, num_nodes=self.num_nodes)
        transform = T.RandomLinkSplit(num_val = 0.1, num_test = 0.1, is_undirected = True,
                                      split_labels = True)
        train_data, val_data, test_data = transform(data)
        self.split_edge = {"train": {"edge": train_data.pos_edge_label_index.t(),
                                     "edge_neg": train_data.neg_edge_label_index.t()},
                          "valid": {"edge": val_data.pos_edge_label_index.t(),
                                    "edge_neg": val_data.neg_edge_label_index.t()}}
        # Define training params
        self.x = self.initialize_node_embedding(data, self.ft_init)
        self.adj_t = edge_index_t

        start = time.time()
        if self.test_distribution == "IN_trd":
            print(f'Processing transductive test dataset...')
            self.split_edge["test"] = {"edge": test_data.pos_edge_label_index.t(),
                                        "edge_neg": test_data.neg_edge_label_index.t()}

        elif self.test_distribution == "IN_ind":
            print(f'Processing inductive test dataset...')
            INDedge_index_t,INDnum_nodes = self.load_edge_index(self.name + "ind")

            INDblock_id_df = torch.load('./dataset/{}_blockID.pt'.format(self.name + "ind")) \
                         if self.name[5:8] == "sbm" else None
            INDdata = Data(edge_index=INDedge_index_t, num_nodes=INDnum_nodes)
            self.INDadj_t = INDedge_index_t
            self.INDx = self.initialize_node_embedding(INDdata, self.ft_init)
            _,_,INDtest_data = transform(INDdata)

            self.split_edge["test"] = {"edge": INDtest_data.pos_edge_label_index.t(),
                                     "edge_neg": INDtest_data.neg_edge_label_index.t()}

        elif self.test_distribution == "OOD":
            print(f'\nProcessing transd., ind. and OOD test datasets...')
            start = time.time()

            print('Processing transductive test dataset...')
            self.split_edge["test_trd_sub"] = {"edge": test_data.pos_edge_label_index.t(),
                                        "edge_neg": test_data.neg_edge_label_index.t()}

            print('Processing inductive test dataset...')
            INDedge_index_t,INDnum_nodes = self.load_edge_index(self.name + "ind")
            INDblock_id_df = torch.load('./dataset/{}_blockID.pt'.format(self.name + "ind")) \
                         if self.name[5:8] == "sbm" else None
            INDdata = Data(edge_index=INDedge_index_t, num_nodes=INDnum_nodes)
            self.INDadj_t = INDedge_index_t
            self.INDx = self.initialize_node_embedding(INDdata, self.ft_init)
            _,_,INDtest_data = transform(INDdata)
            self.split_edge["test_ind_sub"] = {"edge": INDtest_data.pos_edge_label_index.t(),
                                     "edge_neg": INDtest_data.neg_edge_label_index.t()}

            print('Processing OOD test dataset...')
            OODedge_index_t,self.OODnum_nodes = self.load_edge_index(self.OODdataset)
            OODblock_id_df = torch.load('./dataset/{}_blockID.pt'.format(self.name)) \
                         if self.name[5:8] == "sbm" else None
            OODdata = Data(edge_index=OODedge_index_t, num_nodes=self.OODnum_nodes)
            self.OODadj_t = OODedge_index_t
            self.OODx = self.initialize_node_embedding(OODdata, self.ft_init)
            _,_,OODtest_data = transform(OODdata)

            # Create OODtest dataset
            self.split_edge["test"] = {"edge": OODtest_data.pos_edge_label_index.t(),
                                     "edge_neg": OODtest_data.neg_edge_label_index.t()}
        else:
            raise Exception("Test method not implemented.")

        total_time = time.time() - start
        print('Time taken to create test datasets:', time.strftime("%H:%M:%S",time.gmtime(total_time)))

    def sample_subgraph(self, edge_index, nx_graph, method, remove_nodes=None):
        """Samples an induced subgraph from a given edge_index
            remove_nodes: tensor with nodes to be removed from original graph
                          before sampling
        """
        if method == "forestFire":
            print("Sampling training subgraph with Forest Fire method...")
            sampler = Graph_Sampling.ForestFire()

            if remove_nodes is None:
                sample_subgraph = sampler.induced_graph(nx_graph, sample_size) # graph, number of nodes to sample
            else:
                nx_graph.remove_nodes_from(remove_nodes.tolist())
                sample_subgraph = sampler.induced_graph(nx_graph, sample_size) # graph, number of nodes to sample

            node_subset_tensor = torch.tensor(list(sample_subgraph.nodes()),dtype=torch.long)
            reorderG = nx.convert_node_labels_to_integers(sample_subgraph)
            sub_edge_index = torch.tensor(list(reorderG.edges()),dtype=torch.long).t()
            sub_norm_deg_ft = torch.tensor(list(reorderG.degree()))[:,1]/len(list(reorderG.nodes()))
            node_subset_tensor = torch.tensor(list(sample_subgraph.nodes()),dtype=torch.long)
            reorderG=nx.convert_node_labels_to_integers(sample_subgraph)
            sub_edge_index = torch.tensor(list(reorderG.edges()),dtype=torch.long).t()
            sub_norm_deg_ft=torch.tensor(list(reorderG.degree()))[:,1]/len(list(reorderG.nodes()))
        else:
            print(f'Method required: {method}')
            raise Exception("Method for sampling subgraph not implemented")

        sub_edge_index = to_undirected(sub_edge_index)

        return node_subset_tensor, sub_edge_index, sub_norm_deg_ft

    def initialize_node_embedding(self, data, type_init='deg-inv'):
        if type_init == 'deg-inv':
            G_org = to_networkx(data, to_undirected=True)
            init_embedding = torch.tensor(list(G_org.degree()))[:,1]/len(list(G_org.nodes()))
            init_embedding =torch.unsqueeze(init_embedding,1)
        elif type_init == 'ones':
            init_embedding = torch.ones((data.num_nodes,1))
        return init_embedding


    def get_random_split(self, subgraph_data):
        # Split edges into train, val and test
        transform = T.RandomLinkSplit(num_val = 0.1, num_test = 0.1, is_undirected = True,
                                      split_labels = True)
        sub_train_data, sub_val_data, sub_test_data = transform(subgraph_data)
        return sub_train_data, sub_val_data, sub_test_data

    def load_edge_index(self, name):
        print(f"Loading file {name}...")
        start = time.time()
        _, method_name, num_nodes_str = name.split("_")
        edge_index = torch.load('./dataset/{}.pt'.format(name))
        edge_index_t = edge_index.t()
        print('edge_index_t before undirected:', edge_index_t.shape)
        num_nodes = len(torch.unique(torch.stack((edge_index[:,0], edge_index[:,1]))))
        # assert num_nodes == int(num_nodes_str), "Mismatch num_nodes in filename and edge_index."
        edge_index_undir_t = to_undirected(edge_index=edge_index_t, num_nodes=num_nodes)
        print('edge_index_t after undirected:', edge_index_t.shape)
        total_time = time.time() - start
        print('Time to load torch file:', time.strftime("%H:%M:%S",time.gmtime(total_time)))

        return edge_index_undir_t, num_nodes

    def sample_fake_edges(self, size, src_l, dst_l, block_df):
        if self.name[:3] == 'sbm':
            blocks = np.unique(block_df['block_id'])
            block_src, block_dst = np.random.choice(blocks, 2, replace = False)
            cand_src_l = block_df.node[block_df.block_id == block_src]
            cand_dst_l = block_df.node[block_df.block_id == block_dst]
        else:
            nodes_list = np.unique(np.stack([src_l, dst_l]))
            cand_src_l = nodes_list
            cand_dst_l = nodes_list
        fake_src_index = np.random.choice(cand_src_l, size)
        fake_dst_index = np.random.choice(cand_dst_l, size)
        # Make sure sampled edge is non-edges
        for i in range(size):
            fake_src_neigh = dst_l[src_l==fake_src_index[i]]
            dst_in_neigh = fake_dst_index[i] in fake_src_neigh
            while dst_in_neigh:
                fake_dst_index[i] = np.random.choice(cand_dst_l)
                dst_in_neigh = fake_dst_index[i] in fake_src_neigh

        neg_edges = np.stack((fake_src_index, fake_dst_index), axis = 1)
        return torch.tensor(neg_edges)

    def evaluate(self, pos, neg):
        results = []
        for method in self.eval_method:
            if method[:4] == "Hits":
                if self.name[:4] == "ogbl":
                    evaluator = Evaluator(name=self.name)
                else:
                    evaluator = Evaluator(name='ogbl-ppa')
                K = int(method.split("@")[1])
                evaluator.K = K
                result = evaluator.eval({
                'y_pred_pos': pos,
                'y_pred_neg': neg,
                })[f'hits@{K}']
                results.append(result)
            elif method == "mcc":
                result = mcc(pos, neg)
                results.append(np.float64(result))
            elif method == "balanced_acc":
                result = balanced_acc(pos, neg)
                results.append(np.float64(result))
            else:
                raise Exception("eval_method not implemented")

        return results

    def move_to_device(self):
        for key1 in self.split_edge:
            print(f'keys available: {self.split_edge.keys()}')
            for key2 in self.split_edge[key1]:
                print(f'key1 = {key1}')
                print(f'key2 = {key2}')
                print(f'type split edge = {type(self.split_edge[key1][key2])}')
                print(f'lenght split edge = {len(self.split_edge[key1][key2])}')
                self.split_edge[key1][key2] = self.split_edge[key1][key2].to(self.device)
        self.adj_t = self.adj_t.to(self.device)
        self.x = self.x.to(self.device)
