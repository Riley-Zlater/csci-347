# Written by Alexander Alvarez and Riley Slater

import numpy as np
import pandas as pd
import networkx as nx
import graphLib as gl
import matplotlib.pyplot as plt
import csv
import random as rd
import test_case_graphs as tg


rd.seed(42069)

# test graphs
test1 = tg.graph_0
test3 = tg.graph_3
test4 = tg.graph_4 #numVert is 49 should be 50
test7 = tg.graph_7
test8 = tg.graph_8

def randomNodeSampling(G, n):
    toRemove = len(G)-n
    for i in range(toRemove):
        G.remove_node(list(G)[rd.randrange(0, len(list(G)))])
    return np.array([[e[0], e[1]] for e in G.edges]), nx.parse_edgelist([str(e[0]) + ' ' + str(e[1]) for e in G.edges])

#import csv
with open('twitch_eng.csv', newline='') as csvfile:
    edges = np.array(list(csv.reader(csvfile))).astype(int)
    formattedEdgelist = [str(e[0]) + ' ' + str(e[1]) for e in edges]
    G = nx.parse_edgelist(formattedEdgelist, nodetype=int)
    sampledEdges, G = randomNodeSampling(G, 2000)

#tests

print("# of sampled vertices: ", gl.numVert([[e[0], e[1]] for e in G.edges]))
print("# of sampled vertices: ", len(G))
node = list(G)[rd.randrange(0, len(G))]
print("Degree of vertex " + str(node) + ": ", gl.degVert(sampledEdges, node))
print("Clustering coefficient of vertex " + str(node) + ": ", gl.clustCoeff(sampledEdges, node))
print("Avg shortest path length: ", gl.avgShortPathLength(sampledEdges))
#print("Betweeness centrality of vertex " + str(node) + " : ", gl.betweenCent(sampledEdges, node))

plt.figure(figsize=(7, 7))
deg = dict(nx.degree(G))
betw = nx.betweenness_centrality(G)
nx.draw_spring(G, nodelist=deg.keys(), node_color=[v for v in deg.values()], node_size=[v * 2000 for v in betw.values()])
plt.show()

#adjMatrix = pd.DataFrame(gl.adjMatrix(sampledEdges))
#print(adjMatrix)
# print("Betweeness centrality of vertex " + str(node) + " : ", gl.betweenCent(sampledEdges, node))

adjMatrix = pd.DataFrame(gl.adjMatrix(sampledEdges))
print(adjMatrix)
print("# of sampled vertices: ", len(G))
