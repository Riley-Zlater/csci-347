# Written by Alexander Alvarez and Riley Slater

import numpy as np
import pandas as pd
import networkx as nx
import graphLib as gl
import matplotlib.pyplot as plt
import csv
import random as rd

rd.seed(42069)

def randomNodeSampling(G, n):
    toRemove = len(G)-n
    for n in range(toRemove):
        G.remove_node(list(G)[rd.randrange(0, len(list(G)))])
    return np.array([[e[0], e[1]] for e in nx.edges(G)])

#import csv
with open('twitch_eng.csv', newline='') as csvfile:
    edges = np.array(list(csv.reader(csvfile))).astype(int)
    formattedEdgelist = [str(e[0]) + ' ' + str(e[1]) for e in edges]
    G = nx.parse_edgelist(formattedEdgelist, nodetype=int)
    sampledEdges = randomNodeSampling(G, 2000)

#tests
print("# of sampled vertices: ", gl.numVert(sampledEdges))
node = list(G)[rd.randrange(0, len(G))]
print("Degree of vertex " + str(node) + ": ", gl.degVert(sampledEdges, node))
print("Clustering coefficient of vertex " + str(node) + ": ", gl.clustCoeff(sampledEdges, node))
print("Avg shortest path length: ", gl.avgShortPathLength(sampledEdges))
#print("Betweeness centrality of vertex " + str(node) + " : ", gl.betweenCent(sampledEdges, node))

plt.figure(figsize=(20, 20))
nx.draw(G, node_size=5)
plt.show()

#adjMatrix = pd.DataFrame(gl.adjMatrix(sampledEdges))
#print(adjMatrix)
