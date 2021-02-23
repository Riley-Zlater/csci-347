import numpy as np
import pandas as pd
import networkx as nx
import graphLib as gl
import matplotlib.pyplot as plt
import csv
import random as rd
import test_case_graphs as tg

# Written by Alexander Alvarez and Riley Slater

# rd.seed(420696)
GRAPHDISPLAY = False

# test graphs
test1 = tg.graph_1
test2 = tg.graph_2
test3 = tg.graph_3
test4 = tg.graph_4  # numVert is 49 should be 50, ASPL is 17.708 should 17
test5 = tg.graph_5
test6 = tg.graph_6
test7 = tg.graph_7
test8 = tg.graph_8


def randomNodeSampling(G, n):
    toRemove = len(G) - n
    for i in range(toRemove):
        G.remove_node(list(G)[rd.randrange(0, len(list(G)))])
    return np.array([[e[0], e[1]] for e in G.edges]).astype(int), \
           nx.parse_edgelist([str(e[0]) + ' ' + str(e[1]) for e in G.edges])


# import csv
with open('twitch_eng.csv', newline='') as csvfile:
    edges = np.array(list(csv.reader(csvfile))).astype(int)
    formattedEdgelist = [str(e[0]) + ' ' + str(e[1]) for e in edges]
    G = nx.parse_edgelist(formattedEdgelist, nodetype=int)
    sampledEdges, G = randomNodeSampling(G, 2000)

if GRAPHDISPLAY:
    plt.figure(figsize=(7, 7))
    deg = dict(nx.degree(G))
    betw = nx.betweenness_centrality(G)
    nx.draw_spring(G, nodelist=deg.keys(), node_color=[v for v in deg.values()],
                   node_size=[v * 2000 for v in betw.values()])
    plt.show()

topNodesDeg = ""
byDeg = sorted(G.degree, key=lambda x: x[1], reverse=True)
for n in byDeg[:10]:
    if n[0] == byDeg[9][0]:
        topNodesDeg += n[0] + "."
    else:
        topNodesDeg += n[0] + ", "
print("The 10 nodes with the highest degrees are", topNodesDeg)

topNodesBet = ""
byBet = sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)
for n in byBet[:10]:
    if n[0] == byBet[9][0]:
        topNodesBet += n[0] + "."
    else:
        topNodesBet += n[0] + ", "
print("The 10 nodes with the highest betweenness centrality are", topNodesBet)

# all have a clust coeff of 1.0
topNodesClust = ""
byClust = sorted(nx.clustering(G).items(), key=lambda x: x[1], reverse=True)
print(byClust[:10])
for n in byClust[:10]:
    if n[0] == byClust[9][0]:
        topNodesClust += n[0] + "."
    else:
        topNodesClust += n[0] + ", "
print("The 10 nodes with the highest clustering coefficiency are", topNodesClust)

# tests
print("# of sampled vertices: ", gl.numVert(sampledEdges))
print("# of sampled vertices: ", len(G))
node = list(G)[rd.randrange(0, len(G))]
print("Degree of vertex " + str(node) + ": ", gl.degVert(sampledEdges, node))
print("Clustering coefficient of vertex " + str(node) + ": ", gl.clustCoeff(sampledEdges, node))
print("Avg shortest path length: ", gl.avgShortPathLength(sampledEdges))
# print("Betweeness centrality of vertex " + str(node) + " : ", gl.betweenCent(sampledEdges, node))


# adjMatrix = pd.DataFrame(gl.adjMatrix(sampledEdges))
# print(adjMatrix)
# print("Betweeness centrality of vertex " + str(node) + " : ", gl.betweenCent(sampledEdges, node))

adjMatrix = pd.DataFrame(gl.adjMatrix(sampledEdges))
print(adjMatrix)
print("# of sampled vertices: ", len(G))
