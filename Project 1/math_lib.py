#Programmed by Riley Slater and Alexander Alvarez

import numpy as np
import math

### TEMP ###
nums = np.array([[1,2,3,4],[2,3,4,2], [3, 4, 5, 1]])
cats = np.array([["A","mon"],["B","tues"], ["C", "tues"]])
############

def mean(d):
    rows, cols = d.shape
    output = np.empty(cols)
    
    for j in range(cols):
        mean = 0
        for i in range(rows):
            mean += d[i, j]
        output[j] = mean / rows
        
    return output

def covariance(array1, array2):
    D = np.stack((array1, array2), axis=1)
    m_array = mean(D)

    num = 0
    den = 0
    
    for row in D:
        den += 1
        num += (row[0] - m_array[0])*(row[1] - m_array[1])

    cov = num / den

    return cov

def correlation(array1, array2):
    D = np.stack((array1, array2) , axis=1)
    m_array = mean(D)

    num = 0
    denx = 0
    deny = 0
    
    for row in D:
        num += (row[0] - m_array[0])*(row[1] - m_array[1])
        denx += (row[0] - m_array[0]) ** 2
        deny += (row[1] - m_array[1]) ** 2
    den = math.sqrt(denx * deny)
    cor = num / den
    
    return cor
        
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
        mean = np.mean(np.transpose(d[:,j]))
        for i in range(num_rows):
            norm_d[i,j] = (d[i,j]-mean)/std
    return norm_d

def covariance_matrix(d):
    num_rows, num_cols = d.shape
    matrix = np.empty((num_cols, num_cols))
    for j in range(num_cols):
        for i in range(num_cols):
            matrix[i,j] = covariance(d[:,i],d[:,j])

    return matrix
def label_encode(d):
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

