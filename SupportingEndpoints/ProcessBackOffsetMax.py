import pickle
import pandas
import numpy
import os
import sys
import matplotlib
import matplotlib.pyplot

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

arrayOfThings = arrayOfThings[4:]
arrayOfThings[0] /= 1.1
#arrayOfThings[2] *= 100

## AVG and STDDEV on Z axis
print(arrayOfThings.shape)
#print(arrayOfThings[0][5][0])
means = numpy.max(arrayOfThings, axis=2)
dev   = numpy.std(arrayOfThings, axis=2)

## Create X time graphs of error+dev at depth Y
a = numpy.concatenate([means, dev])
print(a.shape)
ctcsv = pandas.DataFrame(a)
ctcsv.to_csv("test.csv")

try:
    maxY = 105
    maxTDPI = 96 * 2
    resolution = numpy.array((1920, 1080))
    TargetDPI = maxTDPI
    solvedSize = resolution / TargetDPI

    fig, ((ax1, ax2, ax3)) = matplotlib.pyplot.subplots(3,1,sharex=True, dpi=TargetDPI, figsize=solvedSize)
    dra1, = ax1.plot([],[], color="red")
    dra2, = ax2.plot([],[], color="red")
    dra3, = ax3.plot([],[], color="red")

    targetColour = (0.05,0.05,0.05)

    ax1.set_facecolor(targetColour)
    ax2.set_facecolor(targetColour)
    ax3.set_facecolor(targetColour)
    fig.set_facecolor(targetColour)

    ax3.set_xlabel("Prediction Window Length", color='white')
    ax1.set_ylabel("Error (°C)", color='white')
    ax2.set_ylabel("Error (L)", color='white')
    ax3.set_ylabel("Error (100 W)", color='white')

    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white') 
    ax1.spines['right'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white') 
    ax2.spines['right'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')

    ax3.spines['bottom'].set_color('white')
    ax3.spines['top'].set_color('white') 
    ax3.spines['right'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='white')

    dra1.set_label("Temperature Error (°C)")
    dra2.set_label("Water Level Error (L)")
    dra3.set_label("Power Error (100 W)")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    # ax1.set_ylim(-100, 100)
    # ax2.set_ylim(-100, 100)
    # ax3.set_ylim(-100, 100)

    _range = numpy.arange(a[1].shape[0])

    lookupInt = 0 # I'm lazy
    dra1.set_xdata(numpy.arange(0, len(_range)))
    dra1.set_ydata(a[lookupInt])
    ax1.fill_between(numpy.arange(0, len(_range)), a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)

    lookupInt = 1 # I'm lazy
    dra2.set_xdata(numpy.arange(0, len(_range)))
    dra2.set_ydata(a[lookupInt])
    ax2.fill_between(numpy.arange(0, len(_range)), a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)

    lookupInt = 2 # I'm lazy
    dra3.set_xdata(numpy.arange(0, len(_range)))
    dra3.set_ydata(a[lookupInt])
    ax3.fill_between(numpy.arange(0, len(_range)), a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)
except Exception as e:
    print(e)
    pass
finally:
    #simulator.Shutdown()
    fig.savefig("SPFSH_{}.png".format(0))
    input("Press Any Key")
    pass

