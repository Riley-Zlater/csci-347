#Programmed by Riley Slater and Alexander Alvarez

import numpy as np
import math

### TEMP ###
num = np.array([[1,2,3,4],[2,3,4,2], [3, 4, 5, 1]])
cat = np.array([["A","mon"],["B","tues"], ["C", "tues"]])
############

def find_mean(d):
    rows, cols = d.shape
    output = np.empty(cols)
    means = 0
    
    for i in range(cols):
        for j in range(rows):
            mean += D[i, j]
        means = means / rows
        output[i] = means
        means = 0
        
    return output

def find_covariance(array1, array2):
    D = np.stack((array1, array2), axis=1)
    m_array = find_mean(D)

    num = 0
    den = 0
    
    for row in D:
        den += 1
        num += (row[0] - m_array[0])*(row[1] - m_array[1])

    cov = num / den

    return cov

def find_correlation(array1, array2):
    D = np.stack((array1, array2) , axis=1)
    m_array = find_mean(D)

    std1 = 0
    std2 = 0

    for row in D:
        std1 += (row[0] - m_array[0]) ** 2


def range_normalization(d):
    num_rows, num_cols = d.shape
    norm_d = np.empty(d.shape)
    for j in range(num_cols):
        min_val = d[0,j];
        max_val = d[0,j];
        for i in range(num_rows):
            if (d[i,j] < min_val):
                min_val = d[i,j]
            elif (d[i,j] > max_val):
                max_val = d[i,j]
        
        for i in range(num_rows):
            norm_d[i,j] = (d[i,j]-min_val)/(max_val-min_val)
    return norm_d

def standard_normalization(d):
    num_rows, num_cols = d.shape
    norm_d = np.empty(d.shape)
    for j in range(num_cols):
        std = np.std(d[:,j])
        mean = np.mean(d[:,j])
        for i in range(num_rows):
            norm_d[i,j] = (d[i,j]-mean)/std
    return norm_d

def label_encode(d):
    print(d.shape)
    num_rows, num_cols = d.shape
    norm_d = np.empty(d.shape).astype(int)
    for j in range(num_cols):
        n = 0;
        encoder = {}
        for i in range(num_rows):
            if d[i,j] in encoder:
                norm_d[i,j] = encoder[d[i,j]]
            else:
                encoder[d[i,j]] = n
                norm_d[i,j] = n
                n += 1
    return norm_d
    

