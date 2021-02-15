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


#TODO: write a function that returns the betweenness centrality of a given vertex
def betweenCent(edgeList, vertex):
    G = nx.read_edgelist(edgeList)
    pass


#TODO: write a function that returns the average shortest paths length of the graph
def avgShortPathLength(edgeList):
    G = nx.read_edgelist(edgeList)
    pass


#TODO: write a function that returns the adjacency matrix of the graph
def adjMatrix(edgeList):
    G = nx.read_edgelist(edgeList)
    pass
