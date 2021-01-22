import numpy as np


def find_mean(2D_array):

    file1 = 0
    file2 = 0
    counter = 0
    
    for row in 2D_array:
        counter +=1
        file1 += row[0]
        file2 += row[1]

    file1 = file1 / counter
    file2 = file2 / counter

    return np.array([file1, file2])

def find_covariance(array1, array2):

    D = np.stack((array1, array2), axis=1)

    m_array = find_mean(D)
