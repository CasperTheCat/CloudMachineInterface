import numpy
import os
import matplotlib
import matplotlib.pyplot
import matplotlib.patches
import pickle

# Allocate
Amats = []
Bmats = []
Cmats = []
Dmats = []

# Get all files
_, _, names = next(os.walk("./"), (None, None, []))


runningLimit = 0
for i in names:
    if i.endswith(".pickle"):
        print(i)
        _, pt1 = i.split("RT", 1)
        index, _ = pt1.split(".", 1)

        index = int(index)

        if index > runningLimit:
            runningLimit = index


print("Found {} files".format(runningLimit))


# Pull into memory
for i in range(runningLimit):
    with open("DMDc_RT{}.pickle".format(i), "rb+") as f:
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

# Print
for i in range(ATDiff.shape[0]):
    print(ATDiff[i])

fig = matplotlib.pyplot.figure()    
axe = fig.gca()
moviePlot = axe.imshow(ADiffx.transpose(), cmap='gist_heat', interpolation='nearest')

axe.set_yticklabels(["", "Temp", "", "Level", "", "Power"])
axe.set_xticklabels(["", "Temp", "", "Level", "", "Power"])

fig.show()




fig2 = matplotlib.pyplot.figure()    
  
axe2 = fig2.gca()
axe2.set_yticklabels(["", "In Rate", "", "In Temp", "", "Target", "", "Out Rate"])
axe2.set_xticklabels(["", "Temp", "", "Level", "", "Power"])
moviePlot2 = axe2.imshow( numpy.abs(BDiffx).transpose(), cmap='gist_heat', interpolation='nearest') 

fig2.show()


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
