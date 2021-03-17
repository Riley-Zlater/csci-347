import numpy as np

# k - means written by Riley Slater
def kmeans():
    pass

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