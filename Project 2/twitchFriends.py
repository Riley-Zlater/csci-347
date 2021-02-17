# Written by Alexander Alvarez and Riley Slater

import numpy as np
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
gl.clustCoeff(edges, '')