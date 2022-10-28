import time, argparse
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
# import itertools
# from networkx.utils import py_random_state

# parser = argparse.ArgumentParser(description='Generate random edge dataset with N nodes')
# parser.add_argument('--gen_method', type=str, default='sbm') #ba #ws
# parser.add_argument('--num_nodes', type=int, default=1000)
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--test_type', type=str, default='initial') #initial, INind, OOD
# parser.add_argument('--seed_graph', type=str, default=None)


###########################
# Set parameters
## Command line
# args = parser.parse_args()
# GEN_METHOD = args.gen_method
# N = args.num_nodes
# SEED = args.seed
# SEED_GRAPH = args.seed_graph
# FILENAME = "{}_{}".format(GEN_METHOD, N)
# M = int(N * 0.05)
# if args.test_type == "INind":
#     FILENAME = FILENAME + "ind"

# Update FILENAME if initial_graph

# if GEN_METHOD == 'ba' and args.test_type == "OOD":
#     if SEED_GRAPH is not None :
#         FILENAME = FILENAME + "fromseed"
#     else:
#         FILENAME = FILENAME + "new"

## Inline
# GEN_METHOD = 'sbm'
# N = 20
# SEED = 0

def save_random_network(GEN_METHOD, N, N_OOD, SEED_l, SEED_GRAPH=True):
    assert len(SEED_l)==3, "SEED_l must have lenght 3."

    # Initialize graph
    g = nx.Graph()
    g_ind = nx.Graph()
    g_OOD = nx.Graph()

    FILENAME = "rand_{}_{}".format(GEN_METHOD, N)
    FILENAME_OOD = "rand_{}_{}".format(GEN_METHOD, N_OOD)
    start = time.time()
    if GEN_METHOD == 'sbm':
        print("=== Generating SBM networks ===")
        # Initialize params for SBM
        sizes = [int(x) for x in [0.45*N, 0.1*N, 0.45*N]]
        sizesOOD = [int(x) for x in [0.45*N_OOD, 0.1*N_OOD, 0.45*N_OOD]]
        probs = [[0.55, 0.05, 0.02], [0.05, 0.55, 0.00], [0.02, 0.00, 0.55]]

        # Generate networks
        g = nx.stochastic_block_model(sizes, probs, seed=SEED_l[0])
        g_ind = nx.stochastic_block_model(sizes, probs, seed=SEED_l[1])
        g_OOD = nx.stochastic_block_model(sizesOOD, probs, seed=SEED_l[2])

        # Save block ids
        save_block_ids(g, FILENAME)
        save_block_ids(g_ind, "{}ind".format(FILENAME))
        save_block_ids(g_OOD, "{}".format(FILENAME_OOD))

    elif GEN_METHOD == "ba":
        print("=== Generating Barabasi-Albert networks ===")
        # Define number of edges to attach from a new node to existing nodes
        M = int(N * 0.1)

        # Generate networks
        g = nx.barabasi_albert_graph(n=N, m=M, seed=SEED_l[0])
        g_ind = nx.barabasi_albert_graph(n=N, m=M, seed=SEED_l[1])

        if not SEED_GRAPH:
            g_OOD = nx.barabasi_albert_graph(n=N_OOD, m=M, seed=SEED_l[2])
        else:
            print("Creating larger BA graph from seed graph...")
            # Creating larger graph from seed
            g_OOD = nx.barabasi_albert_graph(n=N_OOD, m=M, seed=SEED_l[2], initial_graph = g)



    # elif GEN_METHOD == 'ws':
    #     K = int(0.05 * N)
    #     g = nx.watts_strogatz_graph(n=N, k=K, p=0.5, seed=SEED)

    # Graph description
    print(f'Generated graph: {g}')
    print(f'Generated graph: {g_ind}')
    print(f'Generated graph: {g_OOD}')
    # nx.draw(g)

    # Save edge_index to tensor
    print("Saving graphs...")
    save_edge_index(g, FILENAME)
    save_edge_index(g_ind, "{}ind".format(FILENAME))
    if (GEN_METHOD == "ba") and (SEED_GRAPH is True):
        save_edge_index(g_OOD, "{}seed".format(FILENAME_OOD))
    else:
        save_edge_index(g_OOD, "{}".format(FILENAME_OOD))

    total_time = time.time() - start
    print('Time taken:', time.strftime("%H:%M:%S",time.gmtime(total_time)))

    return None

def save_block_ids(g, FILENAME):
    "g: networkX graph"
    block_id_l = []
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            block_id_l.append([node, block_id])
    block_id_tensor = torch.tensor(block_id_l)
    torch.save(block_id_tensor, 'dataset/{}_blockID.pt'.format(FILENAME))
    return None

def save_edge_index(g, FILENAME):
    "g: networkX graph"
    edge_index = torch.tensor([e for e in g.edges])
    torch.save(edge_index, 'dataset/{}.pt'.format(FILENAME))

if __name__ == "__main__":
    seed_l = [0,1,2]
    methods_l = ['ba', 'sbm']
    network_size_l = [100]
    OODnetwork_size_l = [1000]
    for gen_method in methods_l:
        for num_nodes in network_size_l:
            for num_nodesOOD in OODnetwork_size_l:
                output = save_random_network(gen_method, num_nodes, num_nodesOOD, seed_l)
