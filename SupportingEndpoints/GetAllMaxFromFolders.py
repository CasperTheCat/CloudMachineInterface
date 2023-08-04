##### ##### LICENSE ##### #####
# Copyright (C) 2021 Chris Anderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

