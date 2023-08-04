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
import control
import modred
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
matplotlib.interactive(True)
matplotlib.use("TkAgg") 
import numpy
import math
import sys
import Utils



simulator = CSimulator(30, 600000)
#simulator = CSimulator(1, 200000)

spTemp = 55
spTarg = 75
seed = 0
step = Utils.step

if len(sys.argv) > 1:
    seed = int(sys.argv[1])

if len(sys.argv) > 2:
    spTemp = int(sys.argv[2])

if len(sys.argv) > 3:
    spTarg = int(sys.argv[3])

print("Using Seed: {}".format(seed))

boiler = simulator.SpawnObject(ABoiler, 20000, 30, 80, 30)
boilerController = simulator.SpawnObject(ABoilerController, 5, 75, spTemp, seed) # Heating to 95
#boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
boilerController.Possess(boiler)

boiler.SetInflowWaterTemp(24)
boiler.SetInflowRate(0.0)
boiler.SetOutflowRate(0.0001)

# boiler.SetBoilerPower(100)



maxY = 105
maxTDPI = 240
resolution = numpy.array((1920, 1080))
TargetDPI = maxTDPI

solvedSize = resolution / TargetDPI

fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)#figsize=(lScalar*scaleWidth, min((lScalar * scaleWidth*scaleWidth / 16), max(16, (lScalar * 18 / 16)))))
ax = matplotlib.pyplot.axes()
#ax2 = ax.twin()
dra, = ax.plot([],[])
two, = ax.plot([],[])
three, = ax.plot([],[])
four, = ax.plot([],[], linestyle="--")
warn, = ax.plot([],[], linestyle="dotted")

iTime = 60

color = (0.05,0.05,0.05)
# ax.plot([-5,iTime+5], [60,60])
# ax.plot([-5,iTime+5], [30,30])
#ax.axhline(spTemp, linestyle='--', color='red')
ax.yaxis.grid(True, color='white')


ax.set_facecolor(color)
fig.set_facecolor(color)

ax.set_xlabel("Window Time (Seconds)", color='white')
ax.set_ylabel("Temperature (°C) / Power (%) / Water Level (L)", color='white')
#ax.set_ylim(top=maxY, bottom=-1)
ax.set_ylim(top=maxY, bottom=-100)
#ax.set_xlim(left=-5, right=iTime+5)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')

dataP = []#[0]# * iTime 
dataT = []
dataS = []
dataX = []
data5 = []


#### SPSH
model = None

with open("Pickle.era", "rb+") as f:
    model = pickle.load(f)


#input("Press Any Key")

history = []# [[],[],[],[],[],[],[],[]]


for i in range(Utils.seqLength):
    simulator.SimulateNTicks(step * 100, 1/100)

    # Add
    hist = [
        boiler.waterInRatePerSecond,
        boiler.GetInflowWaterTemp(),
        boilerController.temperatureSetPoint,
        boiler.waterOutRatePerSecond,
        boiler.GetBoilerWaterTemp(),
        boiler.waterVolCurrent,
        boiler.boilerPerformance,
        boiler.boilerPercent
    ]
    history.append(numpy.array(hist))

    # history[0].append(boiler.waterInRatePerSecond)

    # # Temperature
    # history[1].append(boiler.GetInflowWaterTemp())

    # # Setpoint
    # history[2].append(boilerController.temperatureSetPoint)

    # # Out Flow Rate
    # history[3].append(boiler.waterOutRatePerSecond)

    # # Out Flow Temperature
    # history[4].append(boiler.GetBoilerWaterTemp())



    # # State Volume
    # history[0].append(boiler.waterVolCurrent)

    # # State Power
    # history[1].append(boiler.boilerPerformance)
    
    # history[2].append(boiler.boilerPercent)

    print("step {}".format(i))

history = numpy.array(history)
warningBar = []

try:
    for i in range(1300):
        #print("Prediction {}".format(i))

        simulator.SimulateNTicks(step * 100, 1/100)


        # Predict next step
        t, yo, xo = control.forced_response(
            model,
            numpy.arange(0, len(history)) * step,
            U=history.transpose()
        )

        forecast = yo.transpose()[-1]
        print(i, forecast, boiler.GetBoilerWaterTemp())

        # If < EPS
        delta = forecast - boiler.GetBoilerWaterTemp()

        if delta < boiler.GetBoilerWaterTemp() * 0.05:
            delta = 0


        #preds.append(forecast)
        #delta = forecast - tStat
        #delta = delta * Utils.StateOnlyWeight[4]
        #warningBar.append(delta)




        if i == 150:
            # Back
            print("Setting {}".format(i))
            boilerController.SetTarget(spTarg)


        



        # Add
        hist = [
            boiler.waterInRatePerSecond,
            boiler.GetInflowWaterTemp(),
            spTemp,
            boiler.waterOutRatePerSecond,
            boiler.GetBoilerWaterTemp(),
            boiler.waterVolCurrent,
            boiler.boilerPerformance,
            boiler.boilerPercent
        ]

        history = history[1:]
        history = numpy.concatenate((history, [numpy.array(hist)]))

        ax.collections.clear()
        #ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)


        dataP = numpy.concatenate([dataP, [boiler.GetBoilerWaterTemp()]])
        dataT = numpy.concatenate([dataT, [boiler.boilerPercent * 100]])
        dataX = numpy.concatenate([dataX, [boilerController.temperatureSetPoint]])
        dataS = numpy.concatenate([dataS, [boiler.waterVolCurrent]])
        data5 = numpy.concatenate([data5, [delta]])

        removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

        #dra.set_ydata(dataP[removalCutter:])
        at = 0#max((len(dataP) - 1) - iTime, 0)
        dataP = dataP[at:]
        dataT = dataT[at:]
        dataS = dataS[at:]
        dataX = dataX[at:]
        data5 = data5[at:]
        dra.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
        dra.set_ydata(dataT)
        two.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
        two.set_ydata(dataP)
        three.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
        three.set_ydata(dataS)
        four.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
        four.set_ydata(dataX)
        warn.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
        warn.set_ydata(data5)

        ax.set_xlim(left=-5, right=len(dataP) * simulator.timeDilation +5)

        #ax = pd.plot()
        fig.canvas.draw()
        fig.canvas.flush_events()


        #mod = math.cos(i * 0.1) * 10
        #mod = math.sin(i * 0.01) ** 640 * 30
        #boilerController.SetTarget(spTemp - math.floor(mod))

        #print("Update Setpoint {} -> {}°C".format(spTemp - mod, spTemp - math.floor(mod)))
        #print("Update Boiler Perf {}w".format(boiler.boilerPerformance))
        #print("Update Boiler Hist {}s".format(boiler.CurrentControlTime))

        #simulator.SetTimeDilation(20 * (i + 1))
        #boiler.SetBoilerPower((i + 1) * 10)
        # print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
        #print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
        #print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
        # print("[TIME {:.02f}s] Power Perc: {:.02f}%".format((i + 1) * simulator.timeDilation, boiler.boilerPercent * 100))
        # print("[TIME {:.02f}s] PID: {:.02f}i".format((i + 1) * simulator.timeDilation, boilerController.PID.iVal))
        # print("[TIME {:.02f}s] PIDdbg: {}".format((i + 1) * simulator.timeDilation, boilerController.PID.dbgLastReturn))

except Exception as e:
    print(e)
    pass
finally:
    #simulator.Shutdown()
    fig.savefig("SPSH_{}.png".format(seed))
    #input("Press Any Key")
    pass

