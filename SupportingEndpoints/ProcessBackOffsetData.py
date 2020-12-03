import pickle
import pandas
import numpy
import os
import sys

fileToLoad = sys.argv[1]

## Try to load the dataset
CT_Stab = []
with open(fileToLoad, "rb+") as f:
    CT_Stab = pickle.load(f)

## Convert the dataset from a dict to a 3D array
arrayOfThings = []
for i in CT_Stab.keys():
    arrayOfThings.append(CT_Stab[i])

## Flip said array
arrayOfThings = numpy.array(arrayOfThings).transpose()

## Switch Y and Z
arrayOfThings = numpy.swapaxes(arrayOfThings, 1, 2)

## AVG and STDDEV on Z axis
print(arrayOfThings.shape)
means = numpy.mean(arrayOfThings, axis=2)
dev   = numpy.std(arrayOfThings, axis=2)

## Create X time graphs of error+dev at depth Y
a = numpy.concatenate([means, dev])
print(a.shape)
ctcsv = pandas.DataFrame(a)
ctcsv.to_csv("test.csv")


