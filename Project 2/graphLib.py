import networkx as nx

# Written by Alexander Alvarez and Riley Slater
# Write functions to perform graph analysis

testEdgeList = 'C:\\Users\\riley\\Desktop\\CSCI 347\\bn-cat-mixed-species_brain_1.edges'

def numVert(edgeList):
    G = nx.read_edgelist(edgeList)
    return len(G)

#TODO: write a function the returns the degree of a given vertex
def degVert(edgeList, vertex):
    G = read_edgelist(edgeList)
    pass


#TODO: write a function that returns the clustering coefficient of a given vertex
def clustCoeff(edgeList, vertex):
    G = read_edgelist(edgeList)
    pass


#TODO: write a function that returns the betweenness centrality of a given vertex
def betweenCent(edgeList, vertex):
    G = read_edgelist(edgeList)
    pass


#TODO: write a function that returns the average shortest paths length of the graph
def avgShortPathLength(edgeList):
    G = read_edgelist(edgeList)
    pass


#TODO: write a function that returns the adjacency matrix of the graph
def adjMatrix(edgeList):
    G = read_edgelist(edgeList)
    pass

print(numVert(testEdgeList))
