import torch, torch_sparse
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_sort_pool, global_add_pool, global_mean_pool
import torch.nn.functional as F
import numpy as np

from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU,
                      Sequential, BatchNorm1d as BN)

class GCN(torch.nn.Module):
    def __init__(self, dataset_src, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, cached=False, positional=False, num_nodes=None, testOOD=False,
                 OODnum_nodes=None,node_subset_tensor=None, ind_node_subset_tensor=None):
        super(GCN, self).__init__()

        self.positional = positional
        self.testOOD = testOOD
        self.node_subset_tensor = node_subset_tensor
        self.ind_node_subset_tensor = ind_node_subset_tensor
        self.dataset_src = dataset_src
        if self.positional and num_nodes is None: raise Exception("positional requires num_nodes")
        if self.positional:
            self.pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            self.train_pos_embedding=self.pos_embedding.weight.detach()
            if self.testOOD:
                if self.dataset_src == "ogbl":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding=self.OODpos_embedding.weight[self.node_subset_tensor].detach()
                    self.INDpos_embedding = self.OODpos_embedding.weight[self.ind_node_subset_tensor].detach()
                elif self.dataset_src == "rand":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
                    self.INDpos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            in_channels += hidden_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        # Reset positional embeddings
        if self.positional:
            torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
            if self.testOOD:
                if self.dataset_src == "rand":
                    torch.nn.init.xavier_uniform_(self.OODpos_embedding.weight)
                    torch.nn.init.xavier_uniform_(self.INDpos_embedding.weight)

    def forward(self, x, adj_t, testOOD=False, testIND=False):
        device = x.device
        if self.positional:
            if testOOD:
                x = torch.cat([x, self.OODpos_embedding.weight.detach().to(device)], dim=1)
            elif testIND:
                if self.dataset_src == 'rand':
                    ind_pos_embedding = self.INDpos_embedding.weight.detach().to(device)
                else:
                    ind_pos_embedding = self.INDpos_embedding.to(device)
                # print(f'type of ind pos embedding: {type(ind_pos_embedding)}')
                x = torch.cat([x, ind_pos_embedding], dim=1)
            else:
                if self.dataset_src == 'rand':
                    train_pos_embedding = self.train_pos_embedding.weight.detach().to(device)
                else:
                    train_pos_embedding = self.train_pos_embedding.to(device)
                # print(f'type of train pos embedding: {type(train_pos_embedding)}')
                x = torch.cat([x, train_pos_embedding],dim=1)

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, dataset_src, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, cached=False, positional=False, num_nodes=None, testOOD=False,
                 OODnum_nodes=None,node_subset_tensor=None,ind_node_subset_tensor=None):
        super(SAGE, self).__init__()

        self.positional = positional
        self.testOOD = testOOD
        self.node_subset_tensor = node_subset_tensor
        self.ind_node_subset_tensor = ind_node_subset_tensor
        self.dataset_src = dataset_src
        if self.positional and num_nodes is None: raise Exception("positional requires num_nodes")
        if self.positional:
            self.pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            self.train_pos_embedding=self.pos_embedding.weight.detach()
            if self.testOOD:
                if self.dataset_src == "ogbl":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding=self.OODpos_embedding.weight[self.node_subset_tensor].detach()
                    self.INDpos_embedding = self.OODpos_embedding.weight[self.ind_node_subset_tensor].detach()
                elif self.dataset_src == "rand":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
                    self.INDpos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            in_channels += hidden_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        # Reset positional embeddings
        if self.positional:
            torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
            if self.testOOD:
                if self.dataset_src == "rand":
                    torch.nn.init.xavier_uniform_(self.OODpos_embedding.weight)
                    torch.nn.init.xavier_uniform_(self.INDpos_embedding.weight)

    def forward(self, x, adj_t, testOOD=False, testIND=False):
        device = x.device
        if self.positional:
            if testOOD:
                x = torch.cat([x, self.OODpos_embedding.weight.detach().to(device)], dim=1)
            elif testIND:
                if self.dataset_src == 'rand':
                    ind_pos_embedding = self.INDpos_embedding.weight.detach().to(device)
                else:
                    ind_pos_embedding = self.INDpos_embedding.to(device)
                # print(f'type of ind pos embedding: {type(ind_pos_embedding)}')
                x = torch.cat([x, ind_pos_embedding], dim=1)
            else:
                if self.dataset_src == 'rand':
                    train_pos_embedding = self.train_pos_embedding.weight.detach().to(device)
                else:
                    train_pos_embedding = self.train_pos_embedding.to(device)
                # print(f'type of train pos embedding: {type(train_pos_embedding)}')
                x = torch.cat([x, train_pos_embedding],dim=1)

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GAT(torch.nn.Module):
    def __init__(self, dataset_src, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, cached=False, positional=False, num_nodes=None, testOOD=False,
                 OODnum_nodes=None,node_subset_tensor=None,ind_node_subset_tensor=None):
        super(GAT, self).__init__()

        self.positional = positional
        self.testOOD = testOOD
        self.node_subset_tensor = node_subset_tensor
        self.ind_node_subset_tensor = ind_node_subset_tensor
        self.dataset_src = dataset_src
        if self.positional and num_nodes is None: raise Exception("positional requires num_nodes")
        if self.positional:
            self.pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            self.train_pos_embedding=self.pos_embedding.weight.detach()
            if self.testOOD:
                if self.dataset_src == "ogbl":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding=self.OODpos_embedding.weight[self.node_subset_tensor].detach()
                    self.INDpos_embedding = self.OODpos_embedding.weight[self.ind_node_subset_tensor].detach()
                elif self.dataset_src == "rand":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
                    self.INDpos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            in_channels += hidden_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        # Reset positional embeddings
        if self.positional:
            torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
            if self.testOOD:
                if self.dataset_src == "rand":
                    torch.nn.init.xavier_uniform_(self.OODpos_embedding.weight)
                    torch.nn.init.xavier_uniform_(self.INDpos_embedding.weight)

    def forward(self, x, adj_t, testOOD=False,testIND=False):
        device = x.device
        if self.positional:
            if testOOD:
                x = torch.cat([x, self.OODpos_embedding.weight.detach().to(device)], dim=1)
            elif testIND:
                if self.dataset_src == 'rand':
                    ind_pos_embedding = self.INDpos_embedding.weight.detach().to(device)
                else:
                    ind_pos_embedding = self.INDpos_embedding.to(device)
                # print(f'type of ind pos embedding: {type(ind_pos_embedding)}')
                x = torch.cat([x, ind_pos_embedding], dim=1)
            else:
                if self.dataset_src == 'rand':
                    train_pos_embedding = self.train_pos_embedding.weight.detach().to(device)
                else:
                    train_pos_embedding = self.train_pos_embedding.to(device)
                # print(f'type of train pos embedding: {type(train_pos_embedding)}')
                x = torch.cat([x, train_pos_embedding],dim=1)


        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GIN(torch.nn.Module):
    def __init__(self, dataset_src, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, cached=False, positional=False, num_nodes=None, testOOD=False,
                 OODnum_nodes=None, train_eps=False, jk=True,node_subset_tensor=None,ind_node_subset_tensor=None):
        super(GIN, self).__init__()

        self.positional = positional
        self.testOOD = testOOD
        self.node_subset_tensor = node_subset_tensor
        self.ind_node_subset_tensor = ind_node_subset_tensor
        self.dataset_src = dataset_src
        self.testOOD = testOOD
        self.jk = jk
        if positional and num_nodes is None: raise Exception("positional requires num_nodes")
        if self.positional:
            self.pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            self.train_pos_embedding=self.pos_embedding.weight.detach()
            if self.testOOD:
                if self.dataset_src == "ogbl":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding=self.OODpos_embedding.weight[self.node_subset_tensor].detach()
                    self.INDpos_embedding = self.OODpos_embedding.weight[self.ind_node_subset_tensor].detach()
                elif self.dataset_src == "rand":
                    self.OODpos_embedding = torch.nn.Embedding(OODnum_nodes, hidden_channels).requires_grad_(False)
                    self.train_pos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
                    self.INDpos_embedding = torch.nn.Embedding(num_nodes, hidden_channels).requires_grad_(False)
            in_channels += hidden_channels

        self.convs = torch.nn.ModuleList()
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)

        for _ in range(num_layers - 1):
            self.convs.append( GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))
        if self.jk:
            self.lin1 = Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin1 = Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        # Reset positional embeddings
        if self.positional:
            torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
            if self.testOOD:
                if self.dataset_src == "rand":
                    torch.nn.init.xavier_uniform_(self.OODpos_embedding.weight)
                    torch.nn.init.xavier_uniform_(self.INDpos_embedding.weight)

    def forward(self, x, adj_t, testOOD=False, testIND=False):
        device = x.device
        if self.positional:
            if testOOD:
                x = torch.cat([x, self.OODpos_embedding.weight.detach().to(device)], dim=1)
            elif testIND:
                if self.dataset_src == 'rand':
                    ind_pos_embedding = self.INDpos_embedding.weight.detach().to(device)
                else:
                    ind_pos_embedding = self.INDpos_embedding.to(device)
                # print(f'type of ind pos embedding: {type(ind_pos_embedding)}')
                x = torch.cat([x, ind_pos_embedding], dim=1)
            else:
                if self.dataset_src == 'rand':
                    train_pos_embedding = self.train_pos_embedding.weight.detach().to(device)
                else:
                    train_pos_embedding = self.train_pos_embedding.to(device)
                # print(f'type of train pos embedding: {type(train_pos_embedding)}')
                x = torch.cat([x, train_pos_embedding],dim=1)

        x = self.conv1(x, adj_t)
        xs = [x]
        for conv in self.convs:
            x = conv(x, adj_t)
            xs += [x]
        if self.jk:
            x=torch.cat(xs, dim=1)
        #else:
            #x = global_mean_pool(xs[-1],batch=None)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SVD(torch.nn.Module):
    def __init__(self, adj_t, out_channels, dataset, device):
        super(SVD, self).__init__()
        self.out_channels = out_channels
        self.dataset = dataset
        if self.dataset[:4] == "ogbl":
            self.adj_t = adj_t
            self.embedding,_,_ = torch.svd_lowrank(self.adj_t.to_torch_sparse_coo_tensor(), q=self.out_channels)
        else:
            print("SVD: COO tensor for non-ogbl dataset")
            v = torch.ones(adj_t.shape[1])
            print(f'adj_t.device = {adj_t.device}')
            print(f'v.device = {v.device}')
            self.adj_t = torch.sparse_coo_tensor(adj_t, v, device=device)
            self.embedding,_,_ = torch.svd_lowrank(self.adj_t, q=self.out_channels)
            print(f'self.adj_t = {self.adj_t}')
            # print(f'type coo tensor adj = {type(self.adj_t)}')
            # print(f'shape adj_t = {self.adj_t.shape}')

    def reset_parameters(self):
        if self.dataset[:4] == "ogbl":
            self.embedding,_,_ = torch.svd_lowrank(self.adj_t.to_torch_sparse_coo_tensor(), q=self.out_channels)
        else:
            self.embedding,_,_ = torch.svd_lowrank(self.adj_t, q=self.out_channels)

    def forward(self, x, adj_t):
        return self.embedding

class MCSVD(torch.nn.Module):
    def __init__(self, adj_t, out_channels, num_nodes, nsamples=1):
        super(MCSVD, self).__init__()
        self.adj_t = adj_t
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.nsamples = nsamples
        self.lin1 = torch.nn.Linear(out_channels, out_channels)
        self.lin2 = torch.nn.Linear(out_channels, out_channels)
    def reset_parameters(self):
        pass
    def forward(self, x, adj_t):
        x = 0
        for _ in range(self.nsamples):
            perm = torch.randperm(self.num_nodes)
            adj_t = torch_sparse.permute(adj_t, perm)
            if self.dataset[:4] == "ogbl":
                self.embedding,_,_ = torch.svd_lowrank(self.adj_t.to_torch_sparse_coo_tensor(), q=self.out_channels)
            else:
                self.embedding,_,_ = torch.svd_lowrank(self.adj_t, q=self.out_channels)
            embedding,_,_ = torch.svd_lowrank(adj_t.to_torch_sparse_coo_tensor(), q=self.out_channels, niter=1)
            inv_perm = [None]*self.num_nodes
            for i,j in enumerate(perm):
                inv_perm[j.item()] = i
            embedding = embedding[inv_perm]
            embedding = F.relu(self.lin1(embedding))
            x += embedding
        x = x/self.nsamples
        x = F.relu(self.lin2(x))
        return x
