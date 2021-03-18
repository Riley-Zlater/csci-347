import numpy as np
import pandas as pd
import clusterLib as cl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA


# plot the first 2 attributes of the data
def drawDBSCAN_2D(labeledData):
    plt.scatter(np.array(labeledData[0])[:, 0], np.array(labeledData[0])[:, 1], c='r', marker='.')
    for i in range(1, len(labeledData.keys())):
        plt.scatter(np.array(labeledData[i]['C'])[:, 0], np.array(labeledData[i]['C'])[:, 1], c='g', marker='X')
        if len(labeledData[i]['B']) == 0:
            continue
        plt.scatter(np.array(labeledData[i]['B'])[:, 0], np.array(labeledData[i]['B'])[:, 1], c='b', marker='p')
    plt.show()

def drawKMEANS_2D(means, labeledData):
    plt.scatter(np.array(means)[:, 0], np.array(means)[:, 1], color='g', marker='X')
    for i in range(len(labeledData.keys())):
        plt.scatter(np.array(labeledData[i])[:, 0], np.array(labeledData[i])[:, 1], color=cm.hot(i/len(labeledData.keys())), marker='p')
    plt.show()

# import data
datafile = open('seeds_dataset.txt', 'r')
lines = datafile.readlines()
data = np.array([line.split() for line in lines]).astype(float)
print(pd.DataFrame(data))

#PCA the data to two attributes
pca = PCA(n_components=2)
pca.fit(data)
pcaData = pca.transform(data)
print(pcaData)

#plot dbscan
#drawDBSCAN_2D(cl.dbscan(pcaData, .7, 9))

#plot kmeans
means, labeledData = cl.kmeans(pcaData, 3, .01, 0)
drawKMEANS_2D(means, labeledData)

