import numpy as np
import pandas as pd
import math_lib as ml
import csv

#import csv
with open('forestfires.csv', newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)))

#format and print
formattedData = pd.DataFrame(data);
print("Raw data: (with attribute names)\n", formattedData)

#remove attribute names
data = data[1:,:]

#label encode
encodedColumns = ml.label_encode(np.stack((data[:,2],data[:,3]), axis=1));
data[:,2] = encodedColumns[:,0]
data[:,3] = encodedColumns[:,1]
data = data.astype(float)

#format and print
formattedData = pd.DataFrame(data);
print("Label encoded data:\n", formattedData)

### NO MISSING DATA. FILL NOT REQUIRED ###

print("Multivariate mean: ", pd.DataFrame(ml.mean(data)))

print("Covariance matrix:\n", pd.DataFrame(ml.covariance_matrix(data)))

##TODO: scatter plots of 5 pairs of attributes (any 5 pairs we think will be connected)

range_norm_data = ml.range_normalization(data)
print("Range-normalized covariance matrix:\n", pd.DataFrame(ml.covariance_matrix(range_norm_data)))

##TODO: search for greatest sample covariance, make scatterplot between attributes

std_norm_data = ml.standard_normalization(data)

##TODO: find std normalized attribute pair with greatest correlation + scatter plot

##TODO: find std normalized attribute pair with smallest correlation + scatter plot

#find attribute pairs with specific correlation / covariance
num_rows, num_cols = data.shape
n = 0
greater_corr = 0
neg_cov = 0
for j in range(num_cols):
    for i in range(j+1,num_cols):
        n += 1
        corr = ml.correlation(data[:,i],data[:,j])
        cov = ml.covariance(data[:,i], data[:,j])
        if (corr > 0.5):
            greater_corr += 1
        if (cov < 0):
            neg_cov += 1

print("# of attribute pairs: ", n);
print("# of attribute pairs with correlation greater than .5: ", greater_corr)
print("# of attribute pairs with negative covariance: ", neg_cov)

##TODO: find total variance of the data

##TODO: find total variance restricted to 5 attributes with greatest sample variance
