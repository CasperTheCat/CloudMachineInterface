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

dilation = 2
seqLength = 60 * 4

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

step = 15

# 60k seconds, measuring every minute
disturbs, states, targetDisturbs, targetStates = Utils.MakeData(60000, 55, dilation, seqLength, 10, False, step=step, stack=True)

disturbs2, states2, targetDisturbs2, targetStates2 = Utils.MakeData(60000, 45, dilation, seqLength, 10, False, step=step, stack=True)
disturbs3, states3, targetDisturbs3, targetStates3 = Utils.MakeData(60000, 35, dilation, seqLength, 4, False, step=step, stack=True)

disturbs = numpy.concatenate((disturbs, disturbs2, disturbs3))
states = numpy.concatenate((states, states2,states3))
targetDisturbs = numpy.concatenate((targetDisturbs, targetDisturbs2, targetDisturbs3))
targetStates = numpy.concatenate((targetStates, targetStates2, targetStates3))

val_disturbs, val_states, val_targetDisturbs, val_targetStates = Utils.MakeData(60000, 75, dilation, seqLength, 2, False, step=60, stack=True)

print(disturbs.shape)

inFeed = numpy.concatenate((disturbs, states), axis=2)
inVal = numpy.concatenate((val_disturbs, val_states), axis=2)

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
lastStep = [0,0,0]

for i in range(backstep):
    itu = numpy.expand_dims(inFeed[i], 0)

    tStat = targetStates[i]

    pairwiseErrors.append(lastStep - tStat)

    lastStep = tStat


pairwiseErrors = numpy.array(pairwiseErrors)

# Abs Errors
absPairwise = numpy.absolute(pairwiseErrors)

pairwiseErrors = absPairwise


# Sum
print(pairwiseErrors[0])
#pairwiseErrors = numpy.sum(pairwiseErrors, axis=1)
#print(pairwiseErrors[0])

pairwiseErrors = numpy.flip(pairwiseErrors)
print(pairwiseErrors[0])
pairwiseErrors = numpy.cumsum(pairwiseErrors)
pairwiseErrors = numpy.flip(pairwiseErrors)





maxY = 240
maxTDPI = 240
resolution = numpy.array((1920, 1080))
TargetDPI = maxTDPI

solvedSize = resolution / TargetDPI

fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)#figsize=(lScalar*scaleWidth, min((lScalar * scaleWidth*scaleWidth / 16), max(16, (lScalar * 18 / 16)))))
ax = matplotlib.pyplot.axes()
#ax2 = ax.twin()
dra, = ax.plot([],[])#, linestyle="--")
two, = ax.plot([],[])
three, = ax.plot([],[])
four, = ax.plot([],[])

iTime = 30

color = (0.05,0.05,0.05)
# ax.plot([-5,iTime+5], [60,60])
# ax.plot([-5,iTime+5], [30,30])
ax.axhline(65, linestyle='--', color='red')
ax.yaxis.grid(True, color='white')


ax.set_facecolor(color)
fig.set_facecolor(color)

ax.set_xlabel("Time (Seconds)", color='white')
ax.set_ylabel("Heat (°C) / Boiler Power Level (%)", color='white')
#ax.set_ylim(top=maxY, bottom=-1)
ax.set_ylim(top=maxY, bottom=-1)
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

        # user_disturbances[0].append(boiler.waterInRatePerSecond)
        # user_disturbances[1].append(boiler.GetInflowWaterTemp())
        # user_disturbances[2].append(boiler.waterOutRatePerSecond)

        # stateInformation[3].append(boilerController.temperatureSetPoint)
        # stateInformation[4].append(boiler.waterVolCurrent)
        # stateInformation[5].append(boiler.GetBoilerWaterTemp())

#input("Press Any Key")


ax.collections.clear()
#ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)
# dataP = yo#numpy.concatenate([dataP, [yo]])
# dataT = youts
# dataP = tempsPred


# dataP = numpy.array(distPreds).transpose()[1]
# dataT = targetDisturbs.transpose()[1]
# print(dataT.flatten().squeeze().shape)
# print(len(dataP))
# dataT = list(dataT.flatten())
# dataP = list(dataP.flatten())

dataP = targetStates.transpose()[0]
dataT = targetStates.transpose()[1]
dataS = targetStates.transpose()[2]
dataX = pairwiseErrors.transpose()
print(dataP.flatten().squeeze().shape)
print(dataT.flatten().squeeze().shape)
#print(len(dataP))
dataT = list(dataT.flatten())
dataP = list(dataP.flatten())
dataS = list(dataS.flatten())
dataX = list(dataX.flatten())




#dataS = distPreds[2]




# dataT = numpy.concatenate([dataT, [boiler.boilerPercent * 100]])
# dataX = numpy.concatenate([dataX, [boiler.waterOutRatePerSecond * 100]])
# dataS = numpy.concatenate([dataS, [boiler.waterVolCurrent]])

#removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

#dra.set_ydata(dataP[removalCutter:])
at = 0#max((len(dataP) - 1) - iTime, 0)
at = min(len(dataP), len(dataX))
# dataP = dataP[at:]
# dataT = dataT[at:]
# dataS = dataS[at:]
# dataX = dataX[at:]
dataP = dataP[:at]
dataT = dataT[:at]
dataS = dataS[:at]
dataX = dataX[:at]

print(len(dataP))
print(len(dataT))

dra.set_xdata(numpy.arange(0, len(dataP)) * dilation)
dra.set_ydata(dataP)
two.set_xdata(numpy.arange(0, len(dataT)) * dilation)
two.set_ydata(dataT)
three.set_xdata(numpy.arange(0, len(dataS)) * dilation)
three.set_ydata(dataS)
four.set_xdata(numpy.arange(0, len(dataX)) * dilation)
four.set_ydata(dataX)




# three.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
# three.set_ydata(dataS)
# four.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
# four.set_ydata(dataX)

ax.set_xlim(left=-5, right=len(dataP) * dilation +5)

#ax = pd.plot()
fig.canvas.draw()
fig.canvas.flush_events()

fig.savefig("MGT.png".format(Utils.TimeNow()))

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
