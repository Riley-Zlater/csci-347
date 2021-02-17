# Written by Alexander Alvarez and Riley Slater

import numpy as np
import pandas as pd
import networkx as nx
import graphLib as gl
import csv

#import csv
with open('twitch_eng.csv', newline='') as csvfile:
    edges = np.array(list(csv.reader(csvfile))).astype(int)
    print(edges)

#tests
print("# of vertices: ", gl.numVert(edges))
print("Degree of vertex 7069: ", gl.degVert(edges, 7069))
print("Avg shortest path length: ", gl.avgShortPathLength(edges))
print("Betweeness centrality of vertex 7069: ", gl.betweenCent(edges, 7069))
#gl.clustCoeff(edges, '')


adjMatrix = pd.DataFrame(gl.adjMatrix(edges))
print(adjMatrix)
