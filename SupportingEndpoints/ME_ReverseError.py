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
# targetDisturbs = numpy.concatenate((targetDisturbs, targetDisturbs2, targetDisturbs3))
# targetStates = numpy.concatenate((targetStates, targetStates2, targetStates3))

val_disturbs, val_states, val_targetDisturbs, val_targetStates = Utils.MakeData(60000, 75, dilation, seqLength, 2, False, step=60, stack=False)

print(disturbs.shape)

inFeed = numpy.concatenate((disturbs, states), axis=1)
inVal = numpy.concatenate((val_disturbs, val_states), axis=1)

print(inFeed.shape)
#inVal = numpy.concatenate((val_disturbs, val_states), axis=2)



#print(ins.shape)

l1t = inFeed[:-1]
l2t = inFeed[1:]

l1 = l1t.transpose()[0]
l2 = l2t.transpose()[0]

l1 = l1t.transpose()
l2 = l2t.transpose()


print(l1.shape, l2.shape)


kalman = modred.OKID(l1, l2, disturbs.shape[0] // 3)
era = modred.ERA()
#a,b,c = era.compute_model(kalman, 10, 10)
#b = b * (1/(step * dilation))
a,b,c = modred.era.compute_ERA_model(kalman, 1)
#b = b * (1 / (step * dilation))
b = b * 0.0

asb = control.ss(a,b,c, numpy.zeros((c.shape[0], b.shape[1])))
#print(asb)

poles = control.pole(asb)
#print(poles)





# t, yo, xo = control.forced_response(asb, numpy.arange(0, len(l1[1])), U=l1)
# #print(t)
# yo = yo[2].transpose()
# print(yo.shape)


##### ##### ########## ##### #####
## Build History
##

# Build Forward
# There's no benefit to building backwards since each step is discrete

# How far back?
backstep = seqLength#len(l1t) - 1
print(len(l1t))
preds = []
pairwiseErrors = []

for i in range(backstep):
    itu = numpy.expand_dims(inFeed[i], 0)

    print(l1t[i:(i+1) + seqLength].shape)

    t, yo, xo = control.forced_response(
        asb,
        numpy.arange(0, len(l1t[i:(i+1) + seqLength])),
        U=l1t[i:(i+1) + seqLength].transpose()
    )

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, len(l1[0])),
    #     U=l1[i:i+seqLength]
    # )

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, 10),
    #     U=l1t[i:i+10].transpose()
    # )
    
    # print(yo.shape)

    # for j in range(0,100):
    #     print(yo[j])
    #     #print(yo[0][j], yo[1][j], yo[2][j])

    # #print(yo)
    # print(1/0)
    print(i, yo.transpose()[-1], l2t[i])
    tStat = l2t[i]
    forecast = yo.transpose()[-1]

    pairwiseErrors.append(forecast - tStat)


pairwiseErrorsAcc = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip, useAbs=False)
pairwiseErrors = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip)


dataP = v2t[seqLength:].transpose()[4]
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

fig.savefig("ME_{}.png".format(Utils.TimeNow()))   

#ax = pd.plot()
fig.canvas.draw()
fig.canvas.flush_events()



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
