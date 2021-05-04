import os
import sys
import pickle
import numpy


# Get Root Folders
rFolder = []
for r,d,p in os.walk(sys.argv[1]):
    for f in d:
        rFolder.append((r, f))
    break

# Go into those folder
errorValues = []
for a,b in rFolder:
    for r,d,p in os.walk(os.path.join(a,b)):
        for f in p:
            if f.startswith("Error_"):
                errorValues.append([a, b, f])
                #print([a, b, f])

                #g
        break


for i in errorValues:
    root, val, filename = i

    if filename == "Error_BaseCase.dat" or filename == "Error_FollowCase.dat":
        continue

    with open(
        os.path.join(
            os.path.join(root, val),
            filename
        ), "rb+") as f:
        errorData = pickle.load(f)

        arrayOfThings = []
        for i in errorData.keys():
            arrayOfThings.append(errorData[i])

        RunError = numpy.array(arrayOfThings).astype(numpy.float)
        RunError = numpy.abs(RunError).sum(axis=0)
        # We
        #rX = numpy.max(RunError, axis=1)
        print(val, filename, RunError)
        #input()

