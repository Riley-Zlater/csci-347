import numpy as np
import pandas as pd
import networkx as nx
import graphLib as gl
import matplotlib.pyplot as plt
import csv
import random as rd
import test_case_graphs as tg

# Written by Alexander Alvarez and Riley Slater

# Set True for graphs
GRAPHDISPLAY = True


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
print("The 10 nodes with the highest degrees are", topNodesDeg + "\n")


topNodesBet = ""
byBet = sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)
for n in byBet[:10]:
    if n[0] == byBet[9][0]:
        topNodesBet += "and " + n[0] + "."
    else:
        topNodesBet += n[0] + ", "
print("The 10 nodes with the highest betweenness centrality are", topNodesBet + "\n")


topNodesClust = ""

dupe = {v : k for k, v in nx.clustering(G).items()}
fixDict = {v : k for k, v in dupe.items()}
    
byClust = sorted(fixDict.items(), key=lambda x: x[1], reverse=True)
for n in byClust[:10]:
    if n[0] == byClust[9][0]:
        topNodesClust += "and " + n[0] + "."
    else:
        topNodesClust += n[0] + ", "
print("The 10 nodes with the highest clustering coefficiency are", topNodesClust + "\n")


topNodesEig = ""
byEig = sorted(nx.eigenvector_centrality(G).items(), key=lambda x: x[1], reverse=True)
for n in byEig[:10]:
    if n[0] == byEig[9][0]:
        topNodesEig += "and " + n[0] + "."
    else:
        topNodesEig += n[0] + ", "
print("The 10 nodes with the highest eigenvector centrality are", topNodesEig + "\n")


topNodesPage = ""
byPage = sorted(nx.pagerank(G).items(), key=lambda x: x[1], reverse=True)
for n in byPage[:10]:
    if n[0] == byPage[9][0]:
        topNodesPage += "and " + n[0] + "."
    else:
        topNodesPage += n[0] + ", "
print("The 10 nodes with the highest pagerank are", topNodesPage + "\n")


print("Average shortest path length of", gl.avgShortPathLength(sampledEdges), "\n")


if GRAPHDISPLAY:
    degDict = dict(nx.degree(G))
    degVals = degDict.values()
    uni, c = np.unique(list(degVals), return_counts=True)
    plt.scatter(uni, c)
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("log(degree)")
    plt.ylabel("log(frequency)")
    plt.show()


