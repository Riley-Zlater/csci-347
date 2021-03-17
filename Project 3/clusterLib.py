import numpy as np
from sklearn.metrics import pairwise_distances_argmin

# k - means written by Riley Slater
def kmeans(data, k, epsilon=None, seed=0):

    # randomly select clusters
    randomNum = np.random.RandomState(seed)
    randomData = randomNum.permutation(data.shape[0])[:k]
    center = data[randomData]

    while True:

        # labels based on closest center
        label = pairwise_distances_argmin(data, center)

        # centers found from the means
        newCenter = np.array([data[label == i].mean(0)
                               for i in range(k)])

        # convergence check
        # If the convergence parameter
        # is epsilon then should we be checking if
        # center == epsilon?
        if np.all(center == newCenter):
            break
        center = newCenter

    return center, label

# DBSCAN written by Alexander Alvarez
def dbscan(data, eps, minpts):
    C = 0

    label = {}
    for p in data:
        i = data.index(p)
        if i in label.keys():
            continue
        nbrs = findInRange(p, data, eps)
        if len(nbrs)+1 < minpts:
            label[i] = (0, 'N')
            continue
        C += 1
        label[i] = (C, 'C')
        labelNbrs(C, nbrs, minpts, eps, data, label)



def labelNbrs(C, nbrs, minpts, eps, data, label):
    for q in nbrs:
        n = data.index(q)
        if n in label.keys():
            if label[n] == (0, 'N'):
                label[n] = (C, 'B')
            continue
        label[n] = (C, 'B')
        nbrs2 = findInRange(q, data, eps)
        if nbrs2 + 1 >= minpts:
            label[n] =(C, 'C')
            labelNbrs(C, nbrs2, minpts, eps, data, label)






def findInRange(p, data, dist):
    pts = []
    for v in data:
        if np.linalg.norm(p - v) <= dist:
            pts.append(v)
    return pts;

###TEST###
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

skmeans = KMeans(n_clusters=2, random_state=0).fit(X)
centers, labels = kmeans(X, 2)

print("(seed = 0) sklearn alg labels:\n",skmeans.labels_)
print("(seed = 0) our alg labels\n",labels,'\n')
print("(seed = 0) sklearn alg cluster centers:\n",skmeans.cluster_centers_)
print("(seed = 0) our alg cluster centers:\n",centers)
