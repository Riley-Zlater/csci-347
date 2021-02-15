import networkx as nx

# Written by Alexander Alvarez and Riley Slater
# Write functions to perform graph analysis


# edge list used for testing the functions
#+-------------+
testEdgeList = 'C:\\Users\\riley\\Desktop\\CSCI 347\\bn-cat-mixed-species_brain_1.edges'
#+-------------+


def numVert(edgeList):
    G = nx.read_edgelist(edgeList)
    return len(G)


def degVert(edgeList, vertex):
    G = nx.read_edgelist(edgeList)
    
    if str(vertex) not in G:
        print("Vertex", vertex, "is not in the graph.")
    else:
        return len([n for n in G[str(vertex)]])


def clustCoeff(edgeList, vertex):
    G = nx.read_edgelist(edgeList)
    triDeg = nx.triangles(G, str(vertex))

    return 2.0*triDeg / (degVert(edgeList, vertex)*(degVert(edgeList, vertex)-1))


def betweenCent(edgeList, vertex):
    G = nx.read_edgelist(edgeList)
    betweenness = dict.fromkeys(G, 0.0)
    nodes = G.nodes()

    for i in nodes:
        X, Y, sig = helperSearch(G, i)
        betweenness = helperAccumulate(betweenness, X, Y, sig, i)
    betweenness = helperScale(betweenness, len(G))
    
    return betweenness[str(vertex)]


#TODO: write a function that returns the average shortest paths length of the graph
def avgShortPathLength(edgeList):
    G = nx.read_edgelist(edgeList)
    pass


#TODO: write a function that returns the adjacency matrix of the graph
def adjMatrix(edgeList):
    G = nx.read_edgelist(edgeList)
    pass





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
    s = 1.0/((graphSize-1)*(graphSize-2))
    for n in betweenness:
        betweenness[n] *= s
    return betweenness
            
