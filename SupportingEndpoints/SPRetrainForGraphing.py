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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
import Graphing



simulator = CSimulator(1, 600000)
#simulator = CSimulator(1, 200000)

spTemp = 55
spTarg = 75
seed = 0
step = Utils.step
dlp = 150

if len(sys.argv) > 1:
    seed = int(sys.argv[1])

if len(sys.argv) > 2:
    spTemp = int(sys.argv[2])

if len(sys.argv) > 3:
    spTarg = int(sys.argv[3])

if len(sys.argv) > 4:
    dlp = int(sys.argv[4])

print("Using Seed: {}".format(seed))

# boiler = simulator.SpawnObject(ABoiler, 20000, 30, 80, 30)
# boilerController = simulator.SpawnObject(ABoilerController, 5, 75, spTemp, seed) # Heating to 95
# #boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
#boilerController.Possess(boiler)

# boiler.SetInflowWaterTemp(24)
# boiler.SetInflowRate(0.0)
# boiler.SetOutflowRate(0.0001)

# boiler.SetBoilerPower(100)



maxY = 105
maxTDPI = 120
resolution = numpy.array((1920, 1080))
TargetDPI = maxTDPI

solvedSize = resolution / TargetDPI

iTime = 60
color = (0.05,0.05,0.05)

#### SPSH
model = None

with open("Pickle.era", "rb+") as f:
    model = pickle.load(f)


predmodel = keras.models.load_model("model.tensorflow")

cost = 0
costml = 0
stepsSinceLastTrain = 0
rtTimes = 0
tholdRTTimes = 0



#####
## 

def EvalFunction(history, feedback, i):
    _, yo, xo = control.forced_response(
        model,
        numpy.arange(0, history.shape[0]) * step,
        U=history.transpose()[:4],
        X0=feedback
    )

    # Set the input to the output bar temp
    # output = history[-1].copy()
    # print(history.transpose()[2])
    # print(output)
    # output[4] = yo.transpose()[-1][4]

    # return output, xo.transpose()[0]
    return yo.transpose()[-1], xo.transpose()[1]

def RetrainFunction(history):
    global cost
    global model
    
    beginPerfTime = time.perf_counter()

    print("Retraining")
    splitPerc = history.shape[0] // 100
    split = splitPerc * 10

    ht = history.transpose()
    ht = Utils.TailState(ht, split)
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()
    model, score = Utils.GetBestOKID(l1, l2)

    timePassed = time.perf_counter() - beginPerfTime
    cost += timePassed

def BaseEvalFunction(history, feedback, i):
    return history[-1][4:], []

def BaseRetrainFunction(history):
    pass    

def ML_EvalFunction(history, feedback, i):
    ytest = numpy.expand_dims(history[:Utils.seqLength], 0)
    forecast = predmodel.predict(ytest)
    forebar = tf.squeeze(forecast, 0).numpy()

    return forebar, []

def ML_RetrainFunction(history):
    global cost
    global rtTimes
    global predmodel

    rtTimes += 1
    
    beginPerfTime = time.perf_counter()

    print("Retraining ML")
    splitPerc = history.shape[0] // 100
    split = splitPerc * 90

    ht = history[:split].transpose()
    l1 = ht[:4].transpose()
    l2 = ht[4:].transpose()

    ht2 = history[split:].transpose()
    l12 = ht[:4].transpose()
    l22 = ht[4:].transpose()

    disturbs, states, targetDisturbs, targetStates = Utils.HandleStacking(l1, l2, True, Utils.seqLength)
    inFeed = numpy.concatenate((disturbs, states), axis=2)
    inFeedStates = targetStates

    valDisturbs, valStates, targetValDisturbs, targetValStates = Utils.HandleStacking(l12, l22, True, Utils.seqLength)
    inVal = numpy.concatenate((valDisturbs, valStates), axis=2)
    inValStates = targetValStates

    predmodel.fit(inFeed, inFeedStates, validation_data=(inVal, inValStates), batch_size=8, epochs=3)
    #predmodel.save("model.tensorflow")

    timePassed = time.perf_counter() - beginPerfTime
    cost += timePassed


def ThresholdFunction(signedError, absoluteError):
    global stepsSinceLastTrain
    global tholdRTTimes

    # Fixed step. Can be rolled into the return bool
    # But it's here to be readable
    if (stepsSinceLastTrain * Utils.dilation > 100):
        stepsSinceLastTrain = 0
        tholdRTTimes += 1
        return True

    stepsSinceLastTrain += 1

    return numpy.sum(absoluteError) > 1000


graphing = Graphing.AGraphHolder(seed, spTemp, spTarg, dlp)

## Battery
graphing.TestRetrainLive(maxY, solvedSize, TargetDPI, iTime, color, BaseEvalFunction, BaseRetrainFunction, ThresholdFunction, 300, ["Temperature (C)", "Heater Power (kW)", "Water Level (L)", "Target Temperature (C)", "Cosine Sim.", "Error"])
graphing.TestRetrainLive(maxY, solvedSize, TargetDPI, iTime, color, EvalFunction, RetrainFunction, ThresholdFunction, 300, ["Temperature (C)", "Heater Power (kW)", "Water Level (L)", "Target Temperature (C)", "Cosine Sim.", "Error"])
graphing.TestRetrainLive(maxY, solvedSize, TargetDPI, iTime, color, ML_EvalFunction, ML_RetrainFunction, ThresholdFunction, 300, ["Temperature (C)", "Heater Power (kW)", "Water Level (L)", "Target Temperature (C)", "Cosine Sim.", "Error"])












































#BEEP.LiveUpdate(maxY, solvedSize, TargetDPI, iTime, color, predictionFunction)

# fig, ax, ax2, packedAxis1, packedAxis2 = BEEP.MakeLiveMap(maxY, solvedSize, TargetDPI, iTime, color)

# dra, two, three, four, warn, warnfar, warndiff = packedAxis1
# dra2, two2, three2, four2, warn2 = packedAxis2

# ax.plot([-5,iTime+5], [60,60])
# ax.plot([-5,iTime+5], [30,30])
#ax.axhline(spTemp, linestyle='--', color='red')


# dataP = []#[0]# * iTime 
# dataT = []
# dataS = []
# dataX = []
# dataClose = []
# dataFar = []
# dataDiff = []


# #input("Press Any Key")

# history = []# [[],[],[],[],[],[],[],[]]

# historyLength = 150 + Utils.seqLength

# for i in range(historyLength):
#     simulator.SimulateNTicks(step * 100, 1/100)

#     # Add
#     hist = [
#         boiler.waterInRatePerSecond,
#         boiler.GetInflowWaterTemp(),
#         boilerController.temperatureSetPoint,
#         boiler.waterOutRatePerSecond,
#         boiler.GetBoilerWaterTemp(),
#         boiler.waterVolCurrent,
#         boiler.boilerPerformance * boiler.boilerPercent
#     ]
#     history.append(numpy.array(hist))

#     # history[0].append(boiler.waterInRatePerSecond)

#     # # Temperature
#     # history[1].append(boiler.GetInflowWaterTemp())

#     # # Setpoint
#     # history[2].append(boilerController.temperatureSetPoint)

#     # # Out Flow Rate
#     # history[3].append(boiler.waterOutRatePerSecond)

#     # # Out Flow Temperature
#     # history[4].append(boiler.GetBoilerWaterTemp())



#     # # State Volume
#     # history[0].append(boiler.waterVolCurrent)

#     # # State Power
#     # history[1].append(boiler.boilerPerformance)
    
#     # history[2].append(boiler.boilerPercent)

#     print("step {}".format(i))

# history = numpy.array(history)
# warningBar = []
# xhat = numpy.zeros((history.shape[1]))
# localXhat = numpy.zeros((history.shape[1]))

# boilerController.SetDisableDisturbance()
# backOffset = 15
# arrLength = history.shape[0]

# localHistory = numpy.zeros((backOffset, history.shape[1]))

# try:
#     for i in range(1300):
#         print("Overarching {}".format(i))
#         for x in range(1):
#             simulator.SimulateNTicks(step * 100, 1/100)

#             hist = [
#                 boiler.waterInRatePerSecond,
#                 boiler.GetInflowWaterTemp(),
#                 boilerController.temperatureSetPoint,
#                 #spTemp,
#                 boiler.waterOutRatePerSecond,
#                 boiler.GetBoilerWaterTemp(),
#                 boiler.waterVolCurrent,
#                 boiler.boilerPerformance * boiler.boilerPercent
#             ]

#             if x == 0:
#                 #print(history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset].transpose().shape)

#                 # Predict next step
#                 # Grab data *backOffset* from the end

#                 t, yo, xo = control.forced_response(
#                         model,
#                         numpy.arange(0, Utils.seqLength) * step,
#                         U=history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset].transpose(),
#                         X0=xhat
#                     )
                
#                 # Save this for the next iteration
#                 xhat = xo.transpose()[-1]

#                 # Prep for the loop
#                 localHistory[0] = yo.transpose()[-1]
#                 localXhat = xhat

#                 for sample in range(1, backOffset):

#                     #print(localHistory[:sample].shape)
#                     #print(history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample].shape)

#                     #Concat
#                     if sample < Utils.seqLength:
#                         lh = numpy.concatenate(
#                             [
#                                 history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample],
#                                 localHistory[:sample]
#                             ])
#                     else:
#                         lh = localHistory[sample-Utils.seqLength:sample]
#                     #print(lh.shape)
                    
#                     t, yo, xo = control.forced_response(
#                         model,
#                         numpy.arange(0, Utils.seqLength) * step,
#                         U=lh.transpose(),
#                         X0=localXhat
#                     )

#                     localXhat = xo.transpose()[-1]
#                     localHistory[sample] = yo.transpose()[-1]


#                 #forecast = yo.transpose()[-1]
#                 forecast = localHistory[0]
#                 forecasterErrorFromSetpoint = forecast[4] - hist[4]

#                 closePoint = forecasterErrorFromSetpoint
#                 farPoint = localHistory[-1][4] - hist[4]

#                 ldiff = hist[4] - history[-len(localHistory)][4]
#                 rdiff = farPoint

#                 print(i, ldiff, rdiff)

#                 # If < EPS
#                 #delta = numpy.sum(forecast - numpy.array(hist))

#                 # if delta < boiler.GetBoilerWaterTemp() * 0.05:
#                 #     delta = 0


#                 #preds.append(forecast)
#                 #delta = forecast - tStat
#                 #delta = delta * Utils.StateOnlyWeight[4]
#                 #warningBar.append(delta)                



#             # Add
#             history = history[1:]
#             history = numpy.concatenate((history, [numpy.array(hist)]))


#             # Update Everything
#             if i == dlp:
#                 # Back
#                 print("Setting {}".format(i))
#                 boilerController.SetTarget(spTarg)

#             ax.collections.clear()
#             ax2.collections.clear()
#             #ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)


#             # Second Set
#             #print(dataP.shape)
#             #print(localHistory.shape)
#             predDataP = numpy.concatenate( [dataP, localHistory.transpose()[4]] )
#             # print("\n\n")
#             # print(history[-1])
#             # print(localHistory[-1])
#             # print(boiler.GetBoilerWaterTemp())
#             dra2.set_xdata(numpy.arange(0, len(predDataP)) * simulator.timeDilation)
#             dra2.set_ydata(predDataP)
#             # two2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             # two2.set_ydata(dataP)
#             # three2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             # three2.set_ydata(dataS)
#             # four2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             # four2.set_ydata(dataX)
#             # warn2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             # warn2.set_ydata(data5)


#             dataP = numpy.concatenate([dataP, [boiler.GetBoilerWaterTemp()]])
#             dataT = numpy.concatenate([dataT, [boiler.boilerPercent * 100]])
#             dataX = numpy.concatenate([dataX, [boilerController.temperatureSetPoint]])
#             dataS = numpy.concatenate([dataS, [boiler.waterVolCurrent]])
#             dataClose = numpy.concatenate([dataClose, [ldiff]])
#             dataFar = numpy.concatenate([dataFar, [rdiff]])
#             dataDiff = numpy.concatenate([dataDiff, [abs(ldiff - rdiff)]])
            

#             removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

#             #dra.set_ydata(dataP[removalCutter:])
#             at = 0#max((len(dataP) - 1) - iTime, 0)
#             dataP = dataP[at:]
#             dataT = dataT[at:]
#             dataS = dataS[at:]
#             dataX = dataX[at:]
#             dataClose = dataClose[at:]
#             dataFar = dataFar[at:]
#             dataDiff = dataDiff[at:]
#             dra.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             dra.set_ydata(dataP)
#             two.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             two.set_ydata(dataT)
#             three.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             three.set_ydata(dataS)
#             four.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             four.set_ydata(dataX)
#             warn.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             warn.set_ydata(dataClose)
#             warnfar.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             warnfar.set_ydata(dataFar)
#             warndiff.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
#             warndiff.set_ydata(dataDiff)



#             ax.set_xlim(left=-5, right=len(predDataP) * simulator.timeDilation +5)

#             #ax = pd.plot()
#             fig.canvas.draw()
#             fig.canvas.flush_events()


#             #mod = math.cos(i * 0.1) * 10
#             #mod = math.sin(i * 0.01) ** 640 * 30
#             #boilerController.SetTarget(spTarg - math.floor(mod))

#             #print("Update Setpoint {} -> {}°C".format(spTemp - mod, spTemp - math.floor(mod)))
#             #print("Update Boiler Perf {}w".format(boiler.boilerPerformance))
#             #print("Update Boiler Hist {}s".format(boiler.CurrentControlTime))

#             #simulator.SetTimeDilation(20 * (i + 1))
#             #boiler.SetBoilerPower((i + 1) * 10)
#             # print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
#             #print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
#             #print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
#             # print("[TIME {:.02f}s] Power Perc: {:.02f}%".format((i + 1) * simulator.timeDilation, boiler.boilerPercent * 100))
#             # print("[TIME {:.02f}s] PID: {:.02f}i".format((i + 1) * simulator.timeDilation, boilerController.PID.iVal))
#             # print("[TIME {:.02f}s] PIDdbg: {}".format((i + 1) * simulator.timeDilation, boilerController.PID.dbgLastReturn))

# except Exception as e:
#     print(e)
#     pass
# finally:
#     #simulator.Shutdown()
#     fig.savefig("SPSH_{}.png".format(seed))
#     #input("Press Any Key")
#     pass

