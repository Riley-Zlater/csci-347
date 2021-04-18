import clusterLib as cl
import numpy as np
import csv
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# import csv
with open('exasens.csv', newline='') as csvfile:
    D = np.array(list(csv.reader(csvfile)))


pd.set_option('display.max_columns', None)
# format and print
def formattedData(data, attributeNames=[], classes=[]):
    if len(classes) > 0:
        data = np.append(np.transpose([classes]), data, axis=1)
    if len(attributeNames) > 0:
        data = np.append([attributeNames], data, axis=0)
    return pd.DataFrame(data)


print("Raw data:\n", formattedData(D))

# separate attribute names
attNames = D[0, :]
D = D[1:, :]


# one-hot encoding
def oneHotEncodeBinary(data, columnIndex):
    newColumns = [[], []]
    for instance in data[:, columnIndex]:
        newColumns[0].append(1 if instance == '0' else 0)
        newColumns[1].append(1 if instance == '1' else 0)
    return newColumns


encodedColumns = oneHotEncodeBinary(D, 5)
D = np.insert(D, 5, encodedColumns, axis=1)
attNames = np.insert(attNames, 5, ['Female', 'Male'])
D = np.delete(D, 7, axis=1)
attNames = np.delete(attNames, 7)
print("Encoded Data:\n", formattedData(D, attNames))

# separate classifications
classes = D[:, 0]
D = D[:, 1:]

# range normalization
def range_normalization(data):
    num_rows, num_cols = data.shape
    norm_d = np.empty(data.shape)
    for j in range(num_cols):
        min_val = data[0, j];
        max_val = data[0, j];
        for i in range(num_rows):
            if (data[i, j] < min_val):
                min_val = data[i, j]
            elif (data[i, j] > max_val):
                max_val = data[i, j]

        for i in range(num_rows):
            norm_d[i, j] = (data[i, j] - min_val) / (max_val - min_val)
    return norm_d

D = range_normalization(D.astype(float))
print("Range normalized data:\n", formattedData(D, attNames[1:]))

# PCA
pca2 = PCA(n_components=2)
D_PCA2 = pca2.fit_transform(D)
explainedVarianceRatio = 0
for ratio in pca2.explained_variance_ratio_:
    explainedVarianceRatio += ratio
print("PCA to 2 components:\n", formattedData(D_PCA2))
print("Explained variance ratio with 2 components:", explainedVarianceRatio)

pca4 = PCA(n_components=4)
D_PCA4 = pca4.fit_transform(D)
explainedVarianceRatio = 0
for ratio in pca4.explained_variance_ratio_:
    explainedVarianceRatio += ratio
print("PCA to 4 components:\n", formattedData(D_PCA4))
print("Explained variance ratio with 4 components:", explainedVarianceRatio)

# Find and plot 2D clusters
def drawDBSCAN_2D(labeledData):
    if (len(labeledData[0]) > 0):
        plt.scatter(np.array(labeledData[0])[:, 0], np.array(labeledData[0])[:, 1], c='b', marker='.')
    for i in range(1, len(labeledData.keys())):
        plt.scatter(np.array(labeledData[i]['C'])[:, 0], np.array(labeledData[i]['C'])[:, 1], color=cm.hot((i-1)/len(labeledData.keys())), marker='p')
        if len(labeledData[i]['B']) == 0:
            continue
        plt.scatter(np.array(labeledData[i]['B'])[:, 0], np.array(labeledData[i]['B'])[:, 1], color=cm.hot((i-1)/len(labeledData.keys())), marker='X')
    plt.show()


def drawKMEANS_2D(means, labeledData):
    plt.scatter(np.array(means)[:, 0], np.array(means)[:, 1], color='g', marker='X')
    for i in range(len(labeledData.keys())):
        plt.scatter(np.array(labeledData[i])[:, 0], np.array(labeledData[i])[:, 1], color=cm.hot(i/len(labeledData.keys())), marker='p')
    plt.show()


# plot dbscan
drawDBSCAN_2D(cl.dbscan(D_PCA2, .7, 9))

# plot kmeans
means, labeledData = cl.kmeans(D_PCA2, 4, .01, 0)
drawKMEANS_2D(means, labeledData)
