# Graph generation utilities for random, grid, bipartite, cycle, complete, and near-clique graphs.

import networkx as nx
import random

def generate_random_graph(n, p, seed=None): #Used to generate both dense and non-dense graphs

    return nx.erdos_renyi_graph(n, p, seed = seed)

def generate_grid_graph(rows, cols):

    G = nx.grid_2d_graph(rows, cols)
    return nx.convert_node_labels_to_integers(G)

def generate_bipartite_graph(n, m, p, seed=None): ## There is no bipartite graph generator using Newtworkx, so we will manually create it

    if seed is not None:
        random.seed(seed)
        
    G = nx.Graph()  ## Null Graph

    leftpart = range(n)
    rightpart = range(n, n + m)
    G.add_nodes_from(leftpart, bipartite=0)
    G.add_nodes_from(rightpart, bipartite=1)

    for u in leftpart:
        for v in rightpart:
            if random.random() < p:
                G.add_edge(u, v)

    return G

def generate_cycle_graph(n):
    return nx.cycle_graph(n)

def generate_complete_graph(n):
    return nx.complete_graph(n)

def generate_near_clique(n, remove_fraction=0.05, seed=None):
    if seed is not None:
        random.seed(seed)

    G = generate_complete_graph(n)
    edges = list(G.edges())
    num_to_remove = int(len(edges) * remove_fraction)
    edges_to_remove = random.sample(edges, num_to_remove)
    G.remove_edges_from(edges_to_remove)

    return G