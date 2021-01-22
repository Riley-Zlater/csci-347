import numpy as np
import math_lib as ml
import csv

with open('forrestfires.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

print(data)
