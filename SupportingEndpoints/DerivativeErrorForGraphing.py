#!/usr/bin/env python3

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

import pickle
import pandas
import numpy
import os
import sys
import control
import modred
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
matplotlib.interactive(True)
matplotlib.use("TkAgg") 
import math
import Utils




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

print(arrayOfThings.shape)
exit()

## Switch Y and Z
arrayOfThings = numpy.swapaxes(arrayOfThings, 1, 2)

## AVG and STDDEV on Z axis
print(arrayOfThings.shape)
means = numpy.mean(arrayOfThings, axis=2)
dev   = numpy.std(arrayOfThings, axis=2)

## Create X time graphs of error+dev at depth Y
a = numpy.concatenate([means, dev])
print(a.shape)

a = numpy.concatenate([a[4:7], a[11:]])
print(a.shape)

ctcsv = pandas.DataFrame(a)

print(a[1].shape[0])

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
    ax1.set_ylabel("Temperature Error (°C)", color='white')
    ax2.set_ylabel("Water Level Error (L)", color='white')
    ax3.set_ylabel("Boiler Power Error (100 W)", color='white')

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

    dra1.set_label("Boiler Temperature Error (°C)")
    dra2.set_label("Boiler Water Level Error (L)")
    dra3.set_label("Boiler Power Error (100 W)")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_ylim(-100, 100)
    ax2.set_ylim(-100, 100)
    ax3.set_ylim(-100, 100)

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
   # fig.savefig("SPSH_{}.png".format(self.seed))
    input("Press Any Key")
    pass

ctcsv.to_csv("test.csv")

## self.boiler.GetBoilerWaterTemp(),
## self.boiler.waterVolCurrent,
## self.boiler.boilerPerformance * self.boiler.boilerPercent