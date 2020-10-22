#!/usr/bin/env python3

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
import Utils

dilation = Utils.dilation
seqLength = Utils.seqLength
step = Utils.step

# disabledisturb = True

# #ins, outs, tests, rawins = MakeData(30000,55,dilation, seqLength, 35, disabledisturb)
# #yins, youts, ytest, rawyins = MakeData(15000,45,dilation, seqLength, 21, disabledisturb)

# ins, outs, tests, rawins = MakeData(10000,55,dilation, seqLength, 35, disabledisturb)
# dins, douts, dtests, drawins = MakeData(10000,55,dilation / 20, seqLength, 35, disabledisturb)
# yins, youts, ytest, rawyins = MakeData(5000,85,dilation, seqLength, 21, disabledisturb and False)


# tests = numpy.concatenate((tests, dtests))
# rawins = numpy.concatenate((rawins, drawins))

# print(ins.shape)
# print(rawins.shape)


# 60k seconds, measuring every minute
disturbs, states, targetDisturbs, targetStates = Utils.MakeData(60000, 55, dilation, seqLength, 10, False, step=step, stack=False)

disturbs2, states2, targetDisturbs2, targetStates2 = Utils.MakeData(60000, 45, dilation, seqLength, 10, False, step=step, stack=False)
disturbs3, states3, targetDisturbs3, targetStates3 = Utils.MakeData(60000, 35, dilation, seqLength, 4, False, step=step, stack=False)

disturbs = numpy.concatenate((disturbs, disturbs2, disturbs3))
states = numpy.concatenate((states, states2,states3))
targetDisturbs = numpy.concatenate((targetDisturbs, targetDisturbs2, targetDisturbs3))
targetStates = numpy.concatenate((targetStates, targetStates2, targetStates3))

val_disturbs, val_states, val_targetDisturbs, val_targetStates = Utils.MakeData(60000, 75, dilation, seqLength, 2, False, step=60, stack=False)

print(disturbs.shape)

inFeed = numpy.concatenate((disturbs, states), axis=1)
inVal = numpy.concatenate((val_disturbs, val_states), axis=1)

print(inFeed.shape)
#inVal = numpy.concatenate((val_disturbs, val_states), axis=2)


##### ##### ########## ##### #####
## Build History
##

# Build Forward
# There's no benefit to building backwards since each step is discrete

# How far back?
backstep = seqLength#len(inFeed) - 1

pairwiseErrors = []
offset = Utils.offset + seqLength # Move forward, every other predictor eats the beginning of the time series
lastStep = inFeed[offset]#numpy.ones(.shape)


for i in range(offset, offset + backstep):
    itu = numpy.expand_dims(inFeed[i], 0)

    boilerTemp = lastStep[4]
    waterIn = lastStep[0] * step
    waterTemp = lastStep[1]
    waterVol = lastStep[5]

    # S1 = (waterVol * boilerTemp + waterTemp * waterIn) / (waterIn + waterVol)
    # S2 = boilerTemp + ((lastStep[6] * lastStep[7] * step) / (4200 * waterVol))

    boilerTemp = (waterVol * boilerTemp + waterTemp * waterIn) / (waterIn + waterVol)
    boilerTemp = boilerTemp + ((lastStep[6] * lastStep[7] * step) / (4200 * waterVol))

    lastStep[4] = boilerTemp

    tStat = inFeed[i]

    # print()

    delta = lastStep - tStat

    # Weight
    delta = delta * Utils.StateOnlyWeight
    print(delta)

    pairwiseErrors.append(delta)

    lastStep = tStat


pairwiseErrors = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip)


dataP = inFeed.transpose()[4]
dataT = inFeed.transpose()[5]
dataS = inFeed.transpose()[6]
dataX = pairwiseErrors.transpose()
print(dataP.flatten().squeeze().shape)
print(dataT.flatten().squeeze().shape)
#print(len(dataP))
dataT = list(dataT.flatten())
dataP = list(dataP.flatten())
dataS = list(dataS.flatten())
dataX = list(dataX.flatten())

fig = Utils.MakeScreen(dataP, dataT, dataS, dataX)


#ax = pd.plot()


fig.savefig("EXS_{}.png".format(Utils.TimeNow()))

fig.canvas.draw()
fig.canvas.flush_events()

Utils.MakeCSV(pairwiseErrors, "EXS_{}.csv".format(Utils.TimeNow()))

#simulator.SimulateNTicks(1000, 1/1000)

#simulator.SetTimeDilation(20 * (i + 1))
#boiler.SetBoilerPower((i + 1) * 10)
# print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
#print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
#print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
# print("[TIME {:.02f}s] Power Perc: {:.02f}%".format((i + 1) * simulator.timeDilation, boiler.boilerPercent * 100))
# print("[TIME {:.02f}s] PID: {:.02f}i".format((i + 1) * simulator.timeDilation, boilerController.PID.iVal))
# print("[TIME {:.02f}s] PIDdbg: {}".format((i + 1) * simulator.timeDilation, boilerController.PID.dbgLastReturn))


input("Press Any Key")
