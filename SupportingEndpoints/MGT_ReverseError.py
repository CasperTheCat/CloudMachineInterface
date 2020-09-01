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

disabledisturb = True

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
disturbs, states, targetDisturbs, targetStates = Utils.MakeData(60000, 55, dilation, seqLength, 10, disabledisturb and False, step=step, stack=False, seed=0)

disturbs2, states2, targetDisturbs2, targetStates2 = Utils.MakeData(60000, 45, dilation, seqLength, 10, disabledisturb, step=step, stack=False, seed=2)
disturbs3, states3, targetDisturbs3, targetStates3 = Utils.MakeData(60000, 35, dilation, seqLength, 4, disabledisturb, step=step, stack=False, seed=5)

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

offset = Utils.offset + seqLength # Move forward, every other predictor eats the beginning of the time series
pairwiseErrors = []
lastStep = inVal[offset]#numpy.ones(.shape)
preds = []

for i in range(offset, offset + backstep):
    itu = numpy.expand_dims(inFeed[i], 0)

    tStat = inVal[i]

    delta = lastStep - tStat
    delta = delta * Utils.StateOnlyWeight

    preds.append(lastStep[4])

    pairwiseErrors.append(delta)

    lastStep = tStat


print("Hwl",len(pairwiseErrors))

pairwiseErrorsAcc = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip, useAbs=False)
pairwiseErrors = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip)

print(len(pairwiseErrors))

dataP = inVal[offset:].transpose()[4]
#dataT = inFeed.transpose()[5]
dataT = numpy.array(preds)
dataS = pairwiseErrorsAcc.transpose()
dataX = pairwiseErrors.transpose()
print(dataP.flatten().squeeze().shape)
print(dataT.flatten().squeeze().shape)
#print(len(dataP))
dataT = list(dataT.flatten())
dataP = list(dataP.flatten())
dataS = list(dataS.flatten())
dataX = list(dataX.flatten())

fig = Utils.MakeScreen(dataP, dataT, dataS, dataX)

fig.savefig("MGT_{}.png".format(Utils.TimeNow()))

fig.canvas.draw()
fig.canvas.flush_events()

Utils.MakeCSV(pairwiseErrors, "MGT_{}.csv".format(Utils.TimeNow()))

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
