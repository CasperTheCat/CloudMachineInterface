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

import numpy
import os
import matplotlib
import matplotlib.pyplot
import matplotlib.patches
import pickle
import sklearn
import sklearn.decomposition
from mpl_toolkits.mplot3d import Axes3D
import sys

# Allocate
Amats = []
Bmats = []
Cmats = []
Dmats = []


filepath = ""

if len(sys.argv) > 1:
    filepath = sys.argv[1]


# Get all files
_, _, names = next(os.walk(filepath), (None, None, []))


runningLimit = 0
for i in names:
    if i.endswith(".pickle"):
        print(i)
        _, pt1 = i.split("RT", 1)
        index, _ = pt1.split(".", 1)

        index = int(index)

        if index > runningLimit:
            runningLimit = index


print("Found {} files".format(runningLimit + 1))


#runningLimit = 42

# Pull into memory
for i in range(runningLimit + 1):
    with open(os.path.join(filepath, "DMDc_RT{}.pickle".format(i)), "rb+") as f:
        A,B,C,D = pickle.load(f)

        Amats.append(A)
        Bmats.append(B)
        Cmats.append(C)
        Dmats.append(D)

# Convert
Amats = numpy.array(Amats)
Bmats = numpy.array(Bmats)
Cmats = numpy.array(Cmats)
Dmats = numpy.array(Dmats)


# Calculate Differences
ATDiff = numpy.diff(Amats, axis=0)
BTDiff = numpy.diff(Bmats, axis=0)
CTDiff = numpy.diff(Cmats, axis=0)
DTDiff = numpy.diff(Dmats, axis=0)

# Sum to single abs
ADiff = numpy.sum(numpy.abs(ATDiff))
ADiffx = numpy.sum(numpy.abs(ATDiff), axis=0)
BDiffx = numpy.sum(numpy.abs(BTDiff), axis=0)

ADiffx = numpy.sum(ATDiff, axis=0)
BDiffx = numpy.sum(BTDiff, axis=0)

print(BDiffx.shape)


AMatrixDifferenceMags = numpy.sum(numpy.sum(numpy.abs(ATDiff), axis=2), axis=1)
BMatrixDifferenceMags = numpy.sum(numpy.sum(numpy.abs(BTDiff), axis=2), axis=1)

print(AMatrixDifferenceMags.shape)

#for i in ATDiff
#print(ATDiff.shape)
maxY = 105
maxTDPI = 300
resolution = numpy.array((2048, 2048))
TargetDPI = maxTDPI
scalar = 2

resolution2 = numpy.array((2560, 1440))

solvedSize = resolution / TargetDPI
solvedSize2 = resolution2 / TargetDPI


distOverscan = 11.5


PCAA = []
for i in ATDiff:
    pca = sklearn.decomposition.PCA(svd_solver='full')
    pca.fit(i)
    PCAA.append(pca.singular_values_)


fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)    
#fig = matplotlib.pyplot.figure()    
axe = Axes3D(fig)#fig.gca()

#axe.plot(AMatrixDifferenceMags)
print(numpy.array(PCAA).shape)
PCAAN = numpy.array(PCAA).T
#axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T)
abb = numpy.arange(PCAAN.shape[1])
abc = numpy.expand_dims(abb, 1)
print(abc.shape)
abcdefg = numpy.dot(([1],[0],[0],[1]), abc.T) / PCAAN.shape[1]
abcdefg[3] = [1] * PCAAN.shape[1]
abcdefg[1] = 1 - (abb / PCAAN.shape[1])
print(abcdefg.T)
plottedLns = axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, c=abcdefg.T, s=6)
axe.plot3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, linewidth=0.75)
axe.dist = distOverscan
#plottedLns[0].set_color("pink")

fig.savefig("AMatG2RPCA.png")





PCAA = []
for i in BTDiff:
    pca = sklearn.decomposition.PCA(svd_solver='full')
    pca.fit(i)
    PCAA.append(pca.singular_values_)


fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)    
#fig = matplotlib.pyplot.figure()    
axe = Axes3D(fig)#fig.gca()

#axe.plot(AMatrixDifferenceMags)
print(numpy.array(PCAA).shape)
PCAAN = numpy.array(PCAA).T
#axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T)
abb = numpy.arange(PCAAN.shape[1])
abc = numpy.expand_dims(abb, 1)
print(abc.shape)
abcdefg = numpy.dot(([1],[0],[0],[1]), abc.T) / PCAAN.shape[1]
abcdefg[3] = [1] * PCAAN.shape[1]
abcdefg[1] = 1 - (abb / PCAAN.shape[1])
print(abcdefg.T)
plottedLns = axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, c=abcdefg.T, s=6)
axe.plot3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, linewidth=0.75)
axe.dist = distOverscan
#plottedLns[0].set_color("pink")

fig.savefig("BMatG2RPCA.png")









PCAA = []
for i in Amats:
    pca = sklearn.decomposition.PCA(svd_solver='full')
    pca.fit(i)
    PCAA.append(pca.singular_values_)


fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)    
#fig = matplotlib.pyplot.figure()    
axe = Axes3D(fig)#fig.gca()

#axe.plot(AMatrixDifferenceMags)
print(numpy.array(PCAA).shape)
PCAAN = numpy.array(PCAA).T
#axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T)
abb = numpy.arange(PCAAN.shape[1])
abc = numpy.expand_dims(abb, 1)
print(abc.shape)
abcdefg = numpy.dot(([1],[0],[0],[1]), abc.T) / PCAAN.shape[1]
abcdefg[3] = [1] * PCAAN.shape[1]
abcdefg[1] = 1 - (abb / PCAAN.shape[1])
print(abcdefg.T)
plottedLns = axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, c=abcdefg.T, s=6)
axe.plot3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, linewidth=0.75)
axe.dist = distOverscan
#plottedLns[0].set_color("pink")

fig.savefig("NonDiffAMatrix.png")

PCAA = []
for i in Bmats:
    pca = sklearn.decomposition.PCA(svd_solver='full')
    pca.fit(i)
    PCAA.append(pca.singular_values_)


fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)    
#fig = matplotlib.pyplot.figure()    
axe = Axes3D(fig)#fig.gca()

#axe.plot(AMatrixDifferenceMags)
print(numpy.array(PCAA).shape)
PCAAN = numpy.array(PCAA).T
#axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T)
abb = numpy.arange(PCAAN.shape[1])
abc = numpy.expand_dims(abb, 1)
print(abc.shape)
abcdefg = numpy.dot(([1],[0],[0],[1]), abc.T) / PCAAN.shape[1]
abcdefg[3] = [1] * PCAAN.shape[1]
abcdefg[1] = 1 - (abb / PCAAN.shape[1])
print(abcdefg.T)
plottedLns = axe.scatter3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, c=abcdefg.T, s=6)
axe.plot3D(PCAAN[0].T, PCAAN[1].T, PCAAN[2].T, linewidth=0.75)
axe.dist = distOverscan
#plottedLns[0].set_color("pink")


fig.savefig("NonDiffBMatrix.png")





fig2 = matplotlib.pyplot.figure(figsize=solvedSize2, dpi=TargetDPI)    
axe2 = fig2.gca()

axe2.plot(AMatrixDifferenceMags)
axe2.set_xlabel("Retrain Index")
axe2.set_ylabel("Total Change")
axe2.set_title("A Matrix Change")


fig2.savefig("AMatRunning.png")



fig2 = matplotlib.pyplot.figure(figsize=solvedSize2, dpi=TargetDPI)      
axe2 = fig2.gca()

axe2.plot(BMatrixDifferenceMags)
axe2.set_xlabel("Retrain Index")
axe2.set_ylabel("Total Change")
axe2.set_title("B Matrix Change")


fig2.savefig("BMatRunning.png")


input()





# exit()

# Print
# for i in range(ATDiff.shape[0]):
#     print(ATDiff[i])

fig = matplotlib.pyplot.figure()    
axe = fig.gca()
moviePlot = axe.imshow(numpy.abs(ADiffx).transpose(), cmap='gist_heat', interpolation='nearest')

axe.set_yticklabels(["", "Temp", "", "Level", "", "Power"])
axe.set_xticklabels(["", "Temp", "", "Level", "", "Power"])

fig.show()


print(numpy.abs(ADiffx).transpose())


fig2 = matplotlib.pyplot.figure()    
  
axe2 = fig2.gca()
axe2.set_yticklabels(["", "In Rate", "", "In Temp", "", "Target", "", "Out Rate"])
axe2.set_xticklabels(["", "Temp", "", "Level", "", "Power"])
moviePlot2 = axe2.imshow( numpy.abs(BDiffx).transpose(), cmap='gist_heat', interpolation='nearest') 

print(numpy.abs(BDiffx).transpose())

fig2.show()

input()


for i in range(len(Bmats)):
    axe.collections.clear()
    axe2.collections.clear()

    moviePlot.set_data(numpy.abs(ATDiff[i]).transpose())
    moviePlot2.set_data(numpy.abs(BTDiff[i]).transpose())

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    input()



input()
