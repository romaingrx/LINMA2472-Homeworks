#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2020 Oct 31, 10:56:47
@last modified : 2020 Oct 31, 19:07:19
"""

import random
import numpy as np
import networkx as nx


# ----------------------------------------
#         Independent cascade model
# ----------------------------------------


def independent_cascade_model(graph:nx.Graph, active_nodes:list, p:float) -> dict:
    """
    :param graph: the graph
    :param active_nodes: starting nodes from which we spread
    :param p: the probability threshold to activate a node
    :return: dict(spread = number of activated nodes, list_active= the list of active ndoes)
    """
    g = graph.copy()

    already_active = set(active_nodes[:])
    to_propagate = active_nodes[:]

    while to_propagate:
        active = to_propagate.pop()
        neighbors = list(g.neighbors(active))
        tried = []
        for nei in neighbors:
            if nei in already_active: continue
            activation = random.random() < p
            if activation: 
                g[active][nei]['activated'] = True
                to_propagate.append(nei)
                already_active.add(nei)
            else:
                g[active][nei]['tried'] = True

    #edges_color = ["blue" if g[u][v].get('activated', False) else "black" if g[u][v].get('tried', False) else "gray" for u, v in g.edges]
    #nodes_color = ["green" if no in active_nodes else "blue" if no in already_active else "gray" for no in g.nodes]

    #nx.draw_networkx(g, edge_color=edges_color, node_color=nodes_color, node_size=35, width=0.25, with_labels=False)
    #plt.show()

    spread = len(already_active)
    return {'spread': spread, 'list_active': already_active}


# ----------------------------------------
#         Influence maximisation
# ----------------------------------------

def generate_active_edges(graph:nx.Graph, p:float) -> nx.Graph:
    """
    :param graph: the initial graph
    :param p: probability threshold for edges activation
    :return: the graph with some activated edges
    """

    g = graph.copy()

    edges = list(g.edges)
    edges_weight = np.array([graph[u][v].get('weight', 1) for u, v in graph.edges])

    # As the instruction does not specify what percentage of activated edges we have to take, we have chosen to use multinomial experiments.
    # But we left the possibility to enforce a certain percentage of activated edges as well.
    if p is not None and p > 0:
        edges_mapped = {i:n for i,n in enumerate(g.edges.keys())}
        activated_edges = random.choices(range(len(edges)), edges_weight, k = round(len(edges)*p))
        for activated in activated_edges: 
            u,v = edges_mapped.get(activated) 
            g[u][v]['activated'] = True
    else:
        edges_proba = edges_weight/sum(edges_weight)
        activated_edges = np.random.multinomial(len(edges), edges_proba) > 0
        for activated, (u, v, info) in zip(activated_edges, g.edges.data()): g[u][v]['activated'] = activated

    return g


def get_accessible_neighbors(g:nx.Graph, node:nx.Node) -> list:
    """
    :param g: the graph
    :param node: the root node
    :return: the list of accessible node's neighbors
    """
    accessible_neighbors = []
    for neighbor in g.neighbors(node):
        if g[node][neighbor].get('activated', False):
            accessible_neighbors.append(neighbor)

    return accessible_neighbors

def get_number_accessible_neighbors(g:nx.Graph, A0:list) -> dict:
    """
    :param g: the graph
    :param A0: starting nodes from which we have to retrieve accessible neighbors
    :return: dict(n= number of accessible neighbors, accessible_neighbors= the list of accessible neighbors)
    """
    neighbors = A0[:]
    new_neighbors = A0[:]
    while new_neighbors:
        node = new_neighbors.pop()
        new_neighbors += get_accessible_neighbors(g, node)
        new_neighbors = list(set(new_neighbors) - set(neighbors))
        neighbors += new_neighbors

    n = len(neighbors)
    return dict(
        n=n,
        accessible_neighbors=neighbors
    )

def greedy_influence_maximisation_problem(graph:nx.Graph, k:int, n:int, p:float) -> list:
    """
    :param graph: the graph
    :param k: the number of best nodes we would like to return
    :param n: the number of simulations to retrieve best nodes
    :param p: the probability threshold to active an edge
    :return: the list of most influential nodes
    """
    A0 = []
    graphs = [generate_active_edges(graph, p) for _ in range(n)]

    for _ in range(k):
        nodes_to_test = list(set(graph.nodes) - set(A0))
        nodes_score = dict(zip(nodes_to_test, [0]*len(nodes_to_test)))
        for node in nodes_to_test:
            for g in graphs:
                accessible_dict = get_number_accessible_neighbors(g, A0+[node])
                nodes_score[node] += accessible_dict.get('n')

        # Retrieve the max
        best_node = max(nodes_score, key=nodes_score.get)
        A0.append(best_node)

    return A0
