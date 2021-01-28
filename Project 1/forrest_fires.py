import numpy as np
import pandas as pd
import math_lib as ml
import matplotlib.pyplot as plt
import csv

#import csv
with open('forestfires.csv', newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)))

#format and print
formattedData = pd.DataFrame(data);
print("Raw data: (with attribute names)\n", formattedData)

#remove attribute names
att_names = data[0,:]
data = data[1:,:]

#label encode
encodedColumns = ml.label_encode(np.stack((data[:,2],data[:,3]), axis=1));
data[:,2] = encodedColumns[:,0]
data[:,3] = encodedColumns[:,1]
data = data.astype(float)

#format and print
formattedData = pd.DataFrame(data);
print("\nLabel encoded data:\n", formattedData)

### NO MISSING DATA. FILL NOT REQUIRED ###

print("\nMultivariate mean: ", pd.DataFrame(ml.mean(data)))

print("\nCovariance matrix:\n", pd.DataFrame(ml.covariance_matrix(data)))

plt.figure(1)
plt.scatter(x=data[:,4], y=data[:,7])
plt.xlabel('FFMC (Fine Fuel Moisture Code)')
plt.ylabel('ISI (Initial Spread Index)')
plt.title('FFMC vs ISI')

plt.figure(2)
plt.scatter(x=data[:,5], y=data[:,6])
plt.xlabel('DMC (Duff Moisture Code)')
plt.ylabel('DC (Drought Code)')
plt.title('DMC vs DC')

plt.figure(3)
plt.scatter(x=data[:,10], y=data[:,12])
plt.xlabel('Wind (km/h)')
plt.ylabel('Area Burned (hectares)')
plt.title('Wind vs Area Burned')

plt.figure(4)
plt.scatter(x=data[:,9], y=data[:,8])
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Temperature (Celsius)')
plt.title('Relative Humidity vs Temperature')

<<<<<<< HEAD
# This one kinda sucks TODO
plt.figure(5)
=======
#this isnt how label encoded actually works, I have no idea what the number-to-category is
>>>>>>> bc29818f99150fa066c5d2441af1550bde3bfa43
plt.scatter(x=data[:,2], y=data[:,11])
plt.xlabel('Month')
plt.ylabel('Rain (mm/m2)')
plt.title('Month vs Rain')
plt.show()

range_norm_data = ml.range_normalization(data)
print("\nRange-normalized covariance matrix:\n", pd.DataFrame(ml.covariance_matrix(range_norm_data)))

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

print("\n# of attribute pairs: ", n);
print("# of attribute pairs with correlation greater than .5: ", greater_corr)
print("# of attribute pairs with negative covariance: ", neg_cov)

#use this because otherwise the numbers are very big
print("\nUSING RANGE-NORMALIZED DATA:")

#find variance of total data set
varByAttribute, totalVar = ml.variance(range_norm_data)

print("Variance by attribute:\n", pd.DataFrame(varByAttribute))
print("Total variance of the data: ", totalVar)

#find variance of only 5 largest variance attributes
range_norm_short = np.stack((
    range_norm_data[:,0],
    range_norm_data[:,3],
    range_norm_data[:,5],
    range_norm_data[:,6],
    range_norm_data[:,10]), axis=1)

varByAttribute, totalVar = ml.variance(range_norm_short)
print("Total variance of the 5 attributes with largest variance: ", totalVar)
