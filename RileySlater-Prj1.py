import numpy as np
import math

def find_mean(2D_array):

    col1 = 0
    col2 = 0
    counter = 0
    
    for row in 2D_array:
        counter +=1
        col1 += row[0]
        col2 += row[1]

    col1 = col1 / counter
    col2 = col2 / counter

    return np.array([col1, col2])

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
    

