import networkx as nx
import scipy as sp
import numpy as np

# Written by Alexander Alvarez and Riley Slater
# Write functions to perform graph analysis


def numVert(edgeList):



    verts = []
    for i in range(len(edgeList)):
        if edgeList[i][0] not in verts:
            verts.append(edgeList[i][0])
        elif edgeList[i][1] not in verts:
            verts.append(edgeList[i][1])

    return len(verts)


def degVert(edgeList, vertex):
    deg = 0
    for i in range(len(edgeList)):
        for j in range(2):
            e = edgeList[i][j]
            if e == vertex:
                deg += 1
    return deg


def clustCoeff(edgeList, vertex):
    k = degVert(edgeList, vertex)
    edges = [(e[0], e[1]) for e in edgeList]
    nbrs = []
    for e in edges:
        if e[0] == vertex:
            nbrs.append(e[1])
        elif e[1] == vertex:
            nbrs.append(e[0])

    numEdges = 0
    for v1 in nbrs:
        for v2 in nbrs:
            if (v1, v2) in edges or (v2, v1) in edges:
                numEdges += 1
    if k <= 1:
        return 0
    return (2 * numEdges) / (k * (k - 1))


def betweenCent(edgeList, vertex):
    formattedEdgelist = [str(e[0]) + ' ' + str(e[1]) for e in edgeList]
    G = nx.parse_edgelist(formattedEdgelist, nodetype=int)
    betweenness = dict.fromkeys(G, 0.0)
    nodes = G.nodes()

    for i in nodes:
        X, Y, sig = helperSearch(G, i)
        betweenness = helperAccumulate(betweenness, X, Y, sig, i)
    betweenness = helperScale(betweenness, len(G))

    return betweenness[str(vertex)]


def avgShortPathLength(edgeList):
    formattedEdgelist = [str(e[0]) + ' ' + str(e[1]) for e in edgeList]
    G = nx.parse_edgelist(formattedEdgelist, nodetype=int)

    average = 0.0
    for n in G:
        pathLen = nx.single_source_shortest_path_length(G, n)
        average += sum(pathLen.values())

    nodes = numVert(edgeList)
    return average / (nodes * (nodes - 1))


def adjMatrix(edgeList):
    formattedEdgelist = [str(e[0]) + ' ' + str(e[1]) for e in edgeList]
    G = nx.parse_edgelist(formattedEdgelist, nodetype=int)
    print("# of sampled vertices: ", len(G))
    matrix = [[0 for i in range(len(G))] for j in range(len(G))]
    for i in G:
        nbrs = nx.neighbors(G, i)
        for j in nbrs:
            matrix[list(G).index(i)][list(G).index(j)] = 1
            matrix[list(G).index(j)][list(G).index(i)] = 1

    return matrix


# See also: https://networkx.org/documentation/networkx-1.10/_modules/networkx/algorithms/centrality/betweenness.html#betweenness_centrality
def helperSearch(G, node):
    X = []
    Y = {}
    for n in G:
        Y[n] = []
    sig = dict.fromkeys(G, 0.0)
    D = {}
    sig[node] = 1.0
    D[node] = 0
    Q = [node]
    while Q:
        n = Q.pop(0)
        X.append(n)
        Dn = D[n]
        sign = sig[n]
        for i in G[n]:
            if i not in D:
                Q.append(i)
                D[i] = Dn + 1
            if D[i] == Dn + 1:
                sig[i] += sign
                Y[i].append(n)
    return X, Y, sig


def helperAccumulate(betweenness, X, Y, sig, i):
    D = dict.fromkeys(X, 0)
    while X:
        j = X.pop()
        coef = (1.0 + D[j]) / sig[j]
        for n in Y[j]:
            D[n] += sig[n] * coef
        if j != i:
            betweenness[j] += D[j]
    return betweenness


def helperScale(betweenness, graphSize):
    s = 1.0 / ((graphSize - 1) * (graphSize - 2))
    for n in betweenness:
        betweenness[n] *= s
    return betweenness
