import numpy as np


# k - means written by Alexander Alvarez and Riley Slater
def kmeans(data, k, eps, seed=0):
    # randomly select clusters
    randomNum = np.random.RandomState(seed)
    randomData = randomNum.permutation(data.shape[0])[:k]
    means = data[randomData]

    while True:
        # label points based on closest center
        labeledData = {}
        for c in range(len(means)):
            labeledData[c] = []
        for p in data:
            labeledData[findClosestCenter(p, means)].append(p)

        # find new means based on labels
        newMeans = []
        for i in range(k):
            newMeans.append(np.array(labeledData[i]).mean(0).tolist())
        newMeans = np.array(newMeans)

        # convergence check
        converged = True
        for i in range(len(means)):
            delta = np.linalg.norm(means[i] - newMeans[i])
            # print(delta) Do we still need this print?
            if delta > eps:
                converged = False
                break
        if converged:
            break

        # assignment
        means = newMeans

    return means, labeledData


# DBSCAN written by Alexander Alvarez
def dbscan(data, eps, minpts):
    # initialization
    C = 0
    labels = {
        0: []  # <- noise
        # 1: {  <- cluster #1
        # C: [] <- core points
        # B: [] <- boundary points
    }

    # loop over each point
    for p in data:

        # if point is labeled, skip
        if isLabeled(labels, p):
            continue

        # find neighbors
        nbrs = findInRange(p, data, eps)

        # if there aren't enough neighbors, label point as noise
        if len(nbrs) + 1 < minpts:
            labelPoint(labels, 0, p)
            continue

        # create cluster and label point as core
        C += 1
        labels[C] = {
            'C': [],
            'B': []
        }
        labelPoint(labels, [C, 'C'], p)

        # label neighbors with cluster
        labelNbrs(C, nbrs, minpts, eps, data, labels)

    return labels


def labelNbrs(C, nbrs, minpts, eps, data, labels):
    # loop over each neighbor
    for q in nbrs:
        n = data.tolist().index(q.tolist())

        # if neighbor is labeled, continue
        if isLabeled(labels, q):
            # if neighbor is labeled as noise, relabel
            if isLabeledNoise(labels, q):
                labelPoint(labels, [C, 'B'], q)
            continue

        # find secondary neighbors
        nbrs2 = findInRange(q, data, eps)

        # if there are enough neighbors, label point as core, and label neighbors
        if len(nbrs2) + 1 >= minpts:
            labelPoint(labels, [C, 'C'], q)
            labelNbrs(C, nbrs2, minpts, eps, data, labels)

        # otherwise label neighbor as boundary
        else:
            labelPoint(labels, [C, 'B'], q)


def findInRange(p, data, dist):
    pts = []
    for v in data:
        if np.linalg.norm(p - v) <= dist:
            pts.append(v)
    return pts


def findClosestCenter(p, centers):
    closestCenter = 0
    dist = np.linalg.norm(p - centers[0])
    for c in range(len(centers)):
        d = np.linalg.norm(p - centers[c]);
        if d < dist:
            closestCenter = c
            dist = d
    return closestCenter


def isLabeled(labels, point):
    p = point.tolist()
    for cluster, label in labels.items():
        if cluster == 0:
            if p in label:
                return True
        else:
            if p in label['C']:
                return True
            elif p in label['B']:
                return True


def isLabeledNoise(labels, point):
    if point.tolist() in labels[0]:
        return True


# label is 0 for Noise or [i, 'C'|'B'], where i = cluster #, 'C' = core point, 'B' = boundary point
def labelPoint(labels, label, point):
    if label == 0:
        labels[0].append(point.tolist())
    else:
        labels[label[0]][label[1]].append(point.tolist())
