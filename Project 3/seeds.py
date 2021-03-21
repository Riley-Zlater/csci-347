import numpy as np
import clusterLib as cl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Written by Alexander Alvarez and Riley Slater


# set True to show all
GRAPH = True
KMEANSTESTS = True
DBSCANTESTS = True


# plot the first 2 attributes of the data
def drawDBSCAN_2D(labeledData):
    plt.scatter(np.array(labeledData[0])[:, 0], np.array(labeledData[0])[:, 1], c='b', marker='.')
    for i in range(1, len(labeledData.keys())):
        plt.scatter(np.array(labeledData[i]['C'])[:, 0], np.array(labeledData[i]['C'])[:, 1],
                    color=cm.hot((i - 1) / len(labeledData.keys())), marker='p')
        if len(labeledData[i]['B']) == 0:
            continue
        plt.scatter(np.array(labeledData[i]['B'])[:, 0], np.array(labeledData[i]['B'])[:, 1],
                    color=cm.hot((i - 1) / len(labeledData.keys())), marker='X')
    plt.title('DBSCAN')
    plt.show()


def drawKMEANS_2D(means, labeledData):
    plt.scatter(np.array(means)[:, 0], np.array(means)[:, 1], color='g', marker='X')
    for i in range(len(labeledData.keys())):
        plt.scatter(np.array(labeledData[i])[:, 0], np.array(labeledData[i])[:, 1],
                    color=cm.hot(i / len(labeledData.keys())), marker='p')
    plt.title('kmeans')
    plt.show()


# import data
datafile = open('seeds_dataset.txt', 'r')
lines = datafile.readlines()
data = np.array([line.split() for line in lines]).astype(float)

# PCA the data to two attributes
pca = PCA(n_components=2)
pca.fit(data)
pcaData = pca.transform(data)

# plot our dbscan
if GRAPH:
    drawDBSCAN_2D(cl.dbscan(pcaData, .7, 9))

# plot our kmeans
if GRAPH:
    means, labeledData = cl.kmeans(pcaData, 3, .01, 0)
    drawKMEANS_2D(means, labeledData)

# plot data reduced to two dimensions with pca
if GRAPH:
    plt.scatter(pcaData[:, 0], pcaData[:, 1])
    plt.title('pc1 vs pc2')
    plt.show()

# PCA the data with no specified components
pca2 = PCA()
pca2.fit(data)
pcaData2 = pca2.transform(data)

# elbow plot of the fraction of total variance preserved by principal components
if GRAPH:
    row, col = pcaData2.shape
    plt.plot(range(1, col + 1), np.cumsum(pca2.explained_variance_ratio_), marker='*')
    plt.title('Seed data: fraction of total variance preserved by principal components')
    plt.xlabel('r : the number of principal components')
    plt.ylabel('f(r) : fraction of total variance preserved')
    plt.show()

    # Do we need to print this information if we say it in the report, Alex? Riley
    print("\nWe will use three components and the fraction of total variance\n"
          "captured by three components is {:0.3f}"
          .format(np.cumsum(pca2.explained_variance_ratio_)[2]), '\n')

# test k means with original and pca data
if KMEANSTESTS:
    print('Inertia values for kmeans for clusters 1 - 5'
          ' with the original data')
    for i in range(1, 6):
        kmeansOrgData = KMeans(n_clusters=i).fit(data)
        print('{:0.2f}'.format(kmeansOrgData.inertia_))

    print('\nInertia values for kmeans for clusters 1 - 5'
          ' with the pca data')
    for i in range(1, 6):
        pcaKmeans = KMeans(n_clusters=i).fit(pcaData2)
        print('{:0.2f}'.format(pcaKmeans.inertia_))

# test DBSCAN with original and pca data
if DBSCANTESTS:
    print('\nThe number of clusters found for eps 0.4 - 0.8 with original data')
    for i in np.arange(0.4, 0.9, 0.1):
        originalDBSCAN = DBSCAN(eps=i, min_samples=9).fit(data)
        print(len(set(originalDBSCAN.labels_)) - (1 if -1 in originalDBSCAN.labels_ else 0))

    print('\nThe number of clusters found for mnts 6 - 10 with original data')
    for i in range(6, 11):
        originalDBSCAN2 = DBSCAN(eps=0.7, min_samples=i).fit(data)
        print(len(set(originalDBSCAN2.labels_)) - (1 if -1 in originalDBSCAN2.labels_ else 0))

    print('\nThe number of clusters found for eps 0.4 - 0.8 with pca data')
    for i in np.arange(0.4, 0.9, 0.1):
        pcaDBSCAN = DBSCAN(eps=i, min_samples=9).fit(pcaData2)
        print(len(set(pcaDBSCAN.labels_)) - (1 if -1 in pcaDBSCAN.labels_ else 0))

    print('\nThe number of clusters found for mnts 6 - 10 with pca data')
    for i in range(6, 11):
        pcaDBSCAN2 = DBSCAN(eps=0.7, min_samples=i).fit(pcaData2)
        print(len(set(pcaDBSCAN2.labels_)) - (1 if -1 in pcaDBSCAN2.labels_ else 0))
