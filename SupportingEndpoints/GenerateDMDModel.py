#!/usr/bin/env python3

import pickle
import control
import modred
import pydmd
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
# matplotlib.interactive(True)
# matplotlib.use("TkAgg") 
import numpy
import cmath
import math
import Utils
import scipy
from past.utils import old_div

dilation = Utils.dilation
seqLength = Utils.seqLength
step = Utils.step

disabledisturb = False
allShape = 35000

# 60k seconds, measuring every minute
disturbs, states, targetDisturbs, targetStates = Utils.MakeData(allShape, 55, dilation, seqLength, 3, disabledisturb, step=step, stack=False, seed=0)
disturbs2, states2, targetDisturbs2, targetStates2 = Utils.MakeData(allShape, 45, dilation, seqLength, 10, disabledisturb, step=step, stack=False, seed=2)
disturbs3, states3, targetDisturbs3, targetStates3 = Utils.MakeData(allShape, 35, dilation, seqLength, 4, disabledisturb, step=step, stack=False, seed=5)
disturbs4, states4, targetDisturbs4, targetStates4 = Utils.MakeData(allShape, 85, dilation, seqLength, 18, disabledisturb, step=step, stack=False, seed=8)
disturbs5, states5, targetDisturbs5, targetStates5 = Utils.MakeData(allShape, 95, dilation, seqLength, 7, disabledisturb, step=step, stack=False, seed=11)
disturbs6, states6, targetDisturbs6, targetStates6 = Utils.MakeData(allShape, 15, dilation, seqLength, 16, disabledisturb, step=step, stack=False, seed=16)
disturbs7, states7, targetDisturbs7, targetStates7 = Utils.MakeData(allShape, 25, dilation, seqLength, 6, disabledisturb, step=step, stack=False, seed=19)

print(disturbs.shape)

disturbs = numpy.concatenate((disturbs, disturbs2, disturbs3, disturbs4, disturbs5, disturbs6, disturbs7))
states = numpy.concatenate((states, states2, states3, states4, states5, states6, states7))

# targetDisturbs = numpy.concatenate((targetDisturbs, targetDisturbs2, targetDisturbs3))
# targetStates = numpy.concatenate((targetStates, targetStates2, targetStates3))

#disturbs, states = Utils.ShuffleTogether(disturbs, states)


#val_disturbs, val_states, val_targetDisturbs, val_targetStates = Utils.MakeData(10, 75, dilation, seqLength, 2, False, step=step, stack=False)

print(disturbs.shape)

# Moving concate. The matrixs ranks out of OKID are incorrect
inFeed = disturbs#numpy.concatenate((disturbs, states), axis=1)
#inVal = val_disturbs#numpy.concatenate((val_disturbs, val_states), axis=1)
inStates = states
#inValState = val_states

print(inFeed.shape)
#inVal = numpy.concatenate((val_disturbs, val_states), axis=2)

offset = Utils.offset

#print(ins.shape)

l1t = inFeed
l2t = inStates

# 
# l1t = inFeed[:-1]
# l2t = inFeed[1:]
v2t = disturbs

l1 = l1t.transpose()
#l2 = l2t.transpose()
l2 = l2t.transpose()
#v2 = v2t.transpose()

#l1 = states.transpose()
#l1 = Utils.TailState(l1, offset)
#l2 = Utils.TailState(l2, offset)
#v2 = Utils.TailState(v2, offset)

# Retranspose
l1t = l1.transpose()
l2t = l2.transpose()
#v2t = v2.transpose()

#print(l1.shape, l2.shape)

asb, score = Utils.GetBestDMD(l1, l2)

##### ##### ########## ##### #####
## Sanity Check Rank
##




##### ##### ########## ##### #####
## Pickle
##

with open("Pickle.dmd", "wb+") as f:
    pickle.dump(asb, f)

mrasb, score = Utils.GetBestMrDMD(l1, l2)

with open("Pickle.mrdmd", "wb+") as f:
    pickle.dump(mrasb, f)


## Now, let's just make a bode plot? (AKA does Scipy handle MIMO)

eigs = numpy.power(asb.eigs, old_div(asb.dmd_time['dt'], asb.original_time['dt']))
cacheA = asb.modes.dot(numpy.diag(eigs)).dot(numpy.linalg.pinv(asb.modes))
cacheB = asb.B

system = control.ss(cacheA, cacheB, numpy.identity(cacheA.shape[0]), numpy.zeros(cacheB.shape))
Utils.CreateBodePlots(system, "DMDc")



# eigs = numpy.power(mrasb.eigs, old_div(mrasb.dmd_time['dt'], mrasb.original_time['dt']))
# cacheA = mrasb.modes.dot(numpy.diag(eigs)).dot(numpy.linalg.pinv(mrasb.modes))
# cacheB = mrasb.B

# system = control.ss(cacheA, cacheB, numpy.identity(cacheA.shape[0]), numpy.zeros(cacheB.shape))
# Utils.CreateBodePlots(system, "MrDMD")


exit()








##### ##### ########## ##### #####
## Build History
##

# Build Forward
# There's no benefit to building backwards since each step is discrete

# How far back?
backstep = seqLength * 6#len(l1t) - 1
print(len(l1t))

pairwiseErrors = []
preds = []
xhat = [0]

for i in range(offset, offset + backstep):
    #itu = numpy.expand_dims(inFeed[i], 0)

    #print(l1t[i:(i) + seqLength].shape)

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, len(l1t[i:(i) + seqLength])),
    #     U=l1t[i:(i) + seqLength].transpose()
    # )

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, len(l1[0])),
    #     U=l1[i:i+seqLength]
    # )

    #print(l1t[i:(i) + seqLength])
    #itu = numpy.expand_dims(v2t[i:(i) + seqLength], 0)
    #print(l1t[i:(i) + seqLength])

    #if i == offset + 2:
    #    print(1/0)

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, len(l1t[i:(i) + seqLength])),
    #     U=Utils.TailState(v2t[i:(i) + seqLength], 10)
    # )
    
    t, yo, xhat = control.forced_response(
        asb,
        numpy.arange(0, len(l1t[i:(i) + seqLength])) * step,
        U=v2t[i:(i) + seqLength].transpose(),
        X0=xhat[-1]
    )

    xhat = numpy.array(xhat).transpose()

    # print(v2t[i:(i) + seqLength].shape)
    # print(yo.shape)

    # for j in range(0,100):
    #     print(yo[j])
    #     #print(yo[0][j], yo[1][j], yo[2][j])

    # #print(yo)
    indexer = -1
    ls = v2t[i:(i) + seqLength]
    tStat = ls[indexer]
    forecast = yo.transpose()[indexer]
    #forecast = yo.transpose()[seqLength - 1]
    #print(i, forecast, tStat)

    preds.append(forecast[4])

    delta = forecast - tStat
    delta = delta * Utils.StateOnlyWeight[:-1]

    print(i, numpy.sum(delta))

    pairwiseErrors.append(delta)


pairwiseErrorsAcc = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip, useAbs=False)
pairwiseErrors = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip)



dataP = v2t[seqLength + offset:].transpose()[4] * Utils.StateOnlyWeight[4]
#dataT = inFeed.transpose()[5]
dataT = numpy.array(preds) * Utils.StateOnlyWeight[4]
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

fig.savefig("DTCE_{}.png".format(Utils.TimeNow()))   

#ax = pd.plot()
fig.canvas.draw()
fig.canvas.flush_events()

Utils.MakeCSV(pairwiseErrors, "DTCE_{}.csv".format(Utils.TimeNow()))
Utils.MakeCSV(pairwiseErrorsAcc, "DTCE_{}_SIGNED.csv".format(Utils.TimeNow()))

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