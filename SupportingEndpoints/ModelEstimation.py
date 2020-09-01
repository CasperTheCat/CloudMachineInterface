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


#https://stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray
def window_stack(a, stepsize=1, width=3, nummax=None):
    if not nummax:
        indexer = numpy.arange(width)[None, :] + stepsize*numpy.arange(a.shape[0] - (width))[:, None]
    else:
        indexer = numpy.arange(width)[None, :] + stepsize*numpy.arange(nummax)[:, None]

    return a[indexer]
    return numpy.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )

def MakeData(x,y, td, width, modRange, disable):
    simulator = CSimulator(td, 600000)
    #simulator = CSimulator(1, 200000)

    spTemp = y

    boiler = simulator.SpawnObject(ABoiler, 10000, 30, 80, 30)
    boilerController = simulator.SpawnObject(ABoilerController, 5, 75, spTemp) # Heating to 95
    #boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
    boilerController.Possess(boiler)

    boiler.SetInflowWaterTemp(24)
    boiler.SetInflowRate(0.0)
    boiler.SetOutflowRate(0.0001)

    if(disable):
        boilerController.SetDisableDisturbance()

    measurements = x
    ins = [[],[],[],[],[],[]]
    outs = [[]]
    for i in range(measurements):
        if i % 100 == 0:
            print("Time {}".format(i * 10))
        
        pastTickTemp = boiler.GetBoilerWaterTemp()
        simulator.SimulateNTicks(25, 1/25)

        if(not disable):
            mod = math.sin(i * 0.005) * modRange #** 640 * 30
            boilerController.SetTarget(spTemp - math.floor(mod))

        
        #ins[3].append(boiler.waterVolCurrent)
        ins[0].append(boiler.waterInRatePerSecond)
        ins[1].append(boiler.GetInflowWaterTemp())
        ins[2].append(boiler.waterOutRatePerSecond)
        ins[3].append(boilerController.temperatureSetPoint)
        ins[4].append(boiler.waterVolCurrent)
        #ins[5].append(boiler.boilerPercent)
        ins[5].append(boiler.GetBoilerWaterTemp())
        #ins[4].append((i * 10) * simulator.timeDilation)

        #ins[3].append(boiler.boilerPercent)


        #ins[3].append(pastTickTemp)
        #ins[3].append(boiler.boilerPercent)
        #ins[4].append(boiler.waterVolCurrent)

        outs[0].append(boiler.GetBoilerWaterTemp())

    nins = numpy.array([numpy.array(xi) for xi in ins]).transpose()
    nouts = numpy.array([numpy.array(xi) for xi in outs]).transpose()

    rnins = nins#window_stack(nins, stepsize=1, width=width)
    #nouts = window_stack(nouts, stepsize=1, width=width)
    nouts = nouts#nouts[width:]

    #print(nstack.shape)
    print(rnins.shape)
    print(nouts.shape)

    ninstest = numpy.expand_dims(rnins[0], 0)
    #nouts = numpy.expand_dims(nouts, 1)

    return rnins, nouts, ninstest, nins[width:]


dilation = 25
seqLength = 256

disabledisturb = True

#ins, outs, tests, rawins = MakeData(30000,55,dilation, seqLength, 35, disabledisturb)
#yins, youts, ytest, rawyins = MakeData(15000,45,dilation, seqLength, 21, disabledisturb)

ins, outs, tests, rawins = MakeData(6000,55,dilation, seqLength, 35, disabledisturb)
dins, douts, dtests, drawins = MakeData(6000,40,dilation / 20, seqLength, 35, disabledisturb)
yins, youts, ytest, rawyins = MakeData(1500,85,dilation, seqLength, 21, disabledisturb and False)

ins = numpy.concatenate((ins, dins))
outs = outs + douts
tests = tests + dtests
rawins = rawins + drawins


print(ins.shape)

l1 = ins[:-1].transpose()
l2 = ins[1:].transpose()

print(l1.shape, l2.shape)


kalman = modred.OKID(l1, l2, ins.shape[1] // 2)
era = modred.ERA()
#a,b,c = era.compute_model(kalman, 10, 10)
# b = b * (1/100)
a,b,c = modred.era.compute_ERA_model(kalman, 5000)
#b = b * (1/25)

asb = control.ss(a,b,c, numpy.zeros((c.shape[0], b.shape[1])))
#print(asb)

poles = control.pole(asb)
print(poles)

t, yo, xo = control.forced_response(asb, numpy.arange(0, len(l1[1])), U=l1)
#print(t)
yo = yo.transpose()[5]
print(yo.shape)


#print(control.lqe(ins, ))


#print(xo)
#, 50, 50, 3, 1, 5)

#t, yo, xo = control.forced_response(asb, numpy.arange(0, len(ins[0])), U=ins)
































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
four, = ax.plot([],[])

iTime = 30

color = (0.05,0.05,0.05)
# ax.plot([-5,iTime+5], [60,60])
# ax.plot([-5,iTime+5], [30,30])
ax.axhline(100, linestyle='--', color='red')
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

#input("Press Any Key")

ax.collections.clear()
#ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)


dataP = yo#numpy.concatenate([dataP, [yo]])
# dataT = numpy.concatenate([dataT, [boiler.boilerPercent * 100]])
# dataX = numpy.concatenate([dataX, [boiler.waterOutRatePerSecond * 100]])
# dataS = numpy.concatenate([dataS, [boiler.waterVolCurrent]])

#removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

#dra.set_ydata(dataP[removalCutter:])
at = 0#max((len(dataP) - 1) - iTime, 0)
dataP = dataP[at:]
dataT = dataT[at:]
dataS = dataS[at:]
dataX = dataX[at:]
dra.set_xdata(numpy.arange(0, len(dataP)) * dilation)
dra.set_ydata(dataP)
# two.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
# two.set_ydata(dataP)
# three.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
# three.set_ydata(dataS)
# four.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
# four.set_ydata(dataX)

ax.set_xlim(left=-5, right=len(dataP) * dilation +5)

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