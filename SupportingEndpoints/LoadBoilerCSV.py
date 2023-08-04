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

from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
matplotlib.interactive(True)
matplotlib.use("TkAgg") 
import numpy
import math
import datetime
import sys
import Utils
import Graphing
import pandas

sheetName = "Boiler.csv"
if len(sys.argv) > 1:
    sheetName = sys.argv[1]

realBoiler = pandas.read_csv(sheetName, index_col=0)

## Correct for Wallclock
# Not needed for Boiler.csv since it samples every second!!!
baseWall = datetime.datetime.strptime(realBoiler.iloc[0][0], "%H:%M:%S")


def WallTimeToDeltaTime(x, base):
    valueTime = datetime.datetime.strptime(x[0], "%H:%M:%S")

    deltaT = valueTime - base

    x[0] = deltaT.seconds
    return x

realBoiler = realBoiler.apply(WallTimeToDeltaTime, args=[baseWall], axis=1)

#print(baseWall)
print(realBoiler)


maxY = 250
maxTDPI = 160
resolution = numpy.array((1920, 1080))
TargetDPI = maxTDPI

solvedSize = resolution / TargetDPI

iTime = 60
color = (0.05,0.05,0.05)



fig, ax, ax2, packedAxis1, packedAxis2 = Graphing.MakeLiveMap(maxY, solvedSize, TargetDPI, iTime, color, labelOverrides=["Static pressure below the BFW tank (kPa)", "BFW in-line T[°C]", "Flue gas in-line T [°C]", "Hand-held T[°C]"])

dra, two, three, four, warn, warnfar, warndiff = packedAxis1
dra2, two2, three2, four2, warn2 = packedAxis2

#ax.plot([-5,iTime+5], [60,60])
#ax.plot([-5,iTime+5], [30,30])
#ax.axhline(spTemp, linestyle='--', color='red')


dataP = []
dataT = []
dataS = []
dataX = []
dataTiming = []
dataClose = []
dataFar = []
dataDiff = []



for i in range(len(realBoiler)):
    #print(realBoiler.iloc[i])
    v = realBoiler.iloc[i]
    print(v[0], v[1], v[2], v[3], v[4])


    ax.collections.clear()
    ax2.collections.clear()

    dataTiming = numpy.concatenate([dataTiming, [v[0]]])
    dataP = numpy.concatenate([dataP, [v[1]]])
    dataT = numpy.concatenate([dataT, [v[2]]])
    dataX = numpy.concatenate([dataX, [v[3]]])
    dataS = numpy.concatenate([dataS, [v[4]]])

    dra.set_xdata(dataTiming)
    dra.set_ydata(dataP)
    two.set_xdata(dataTiming)
    two.set_ydata(dataT)
    three.set_xdata(dataTiming)
    three.set_ydata(dataX)
    four.set_xdata(dataTiming)
    four.set_ydata(dataS)

    ax.set_xlim(left=-5, right=dataTiming[-1] + 5)


    fig.canvas.draw()
    fig.canvas.flush_events()
