#!/usr/bin/env python3

#import control
#import modred
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Utils
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
#matplotlib.interactive(True)
#matplotlib.use("TkAgg") 
import numpy
import math

dilation = 2
seqLength = 60
#seqLength = 60 * 24

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

step = 60

# 60k seconds, measuring every minute
disturbs, states, targetDisturbs, targetStates = Utils.MakeData(60000, 55, dilation, seqLength, 10, disabledisturb and False, step=step)
disturbs2, states2, targetDisturbs2, targetStates2 = Utils.MakeData(60000, 45, dilation, seqLength, 20, disabledisturb, step=step)
disturbs3, states3, targetDisturbs3, targetStates3 = Utils.MakeData(60000, 35, dilation, seqLength, 4, disabledisturb, step=step)

disturbs = numpy.concatenate((disturbs, disturbs2, disturbs3))
states = numpy.concatenate((states, states2,states3))
targetDisturbs = numpy.concatenate((targetDisturbs, targetDisturbs2, targetDisturbs3))
targetStates = numpy.concatenate((targetStates, targetStates2, targetStates3))

val_disturbs, val_states, val_targetDisturbs, val_targetStates = Utils.MakeData(60000, 75, dilation, seqLength, 2, False, step=60)


#ins, outs, tests, rawins = MakeData(3000,55,dilation, seqLength, 35, disturb)
#yins, youts, ytest, rawyins = MakeData(1000,45,dilation, seqLength, 21, disturb)


forecastmodel = keras.Sequential(
    [
        #layers.Embedding(input_shape=(100, 3), output_dim=128),
        layers.Input(shape=(seqLength, disturbs.shape[2])),
        #layers.LSTM(1024, return_sequences=True),
        #layers.Dropout(0.2),
        #layers.LSTM(1024, return_sequences=True),
        #layers.GRU(64, return_sequences=True),
        #layers.LSTM(128, return_sequences=True),
        layers.LSTM(256, return_sequences=False),
        #layers.LSTM(64, return_sequences=True),
        #layers.LSTM(64, return_sequences=True),
        
        layers.Dense(256, activation='relu'),
        layers.Dense(disturbs.shape[2])
    ]
)
# model.add(layers.Embedding(input_dim=1000, output_dim=64))
# model.add(layers.LSTM(128))
# model.add(layers.LSTM(128))
# model.add(layers.Dense(10))

predmodel = keras.Sequential(
    [
        #layers.Embedding(input_shape=(100, 3), output_dim=128),
        layers.Input(shape=(seqLength, states.shape[2] + disturbs.shape[2])),
        #layers.LSTM(1024, return_sequences=True),
        #layers.Dropout(0.2),
        #layers.LSTM(1024, return_sequences=True),
#        layers.LSTM(256, return_sequences=False),
        layers.LSTM(256, return_sequences=False),
        


        layers.Dense(256, activation='relu'),
        layers.Dense(states.shape[2])
    ]
)

forecastmodel.compile(
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #metrics=["accuracy"],
)

predmodel.compile(
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #metrics=["accuracy"],
)


predmodel.summary()
forecastmodel.summary()

epochlies = 1

#predmodel.fit(ins, outs, validation_data=(yins, youts), batch_size=16, epochs=epochlies)

print(disturbs[1][-1])
print(targetDisturbs[0])
print(len(disturbs), len(targetDisturbs))

assert(disturbs[1][-1][0] == targetDisturbs[0][0])
assert(disturbs[1][-1][1] == targetDisturbs[0][1])
assert(disturbs[1][-1][2] == targetDisturbs[0][2])
assert(len(disturbs) == len(targetDisturbs))

inFeed = numpy.concatenate((disturbs, states), axis=2)
inVal = numpy.concatenate((val_disturbs, val_states), axis=2)

print(inVal.shape)
print(inFeed.shape)
print(states.shape[2] + disturbs.shape[2])

print(inFeed[0][-1])
print(targetStates[0])

forecastmodel.fit(disturbs, targetDisturbs, validation_data=(val_disturbs, val_targetDisturbs), batch_size=16, epochs=epochlies)
predmodel.fit(inFeed, targetStates, validation_data=(inVal, val_targetStates), batch_size=16, epochs=epochlies)





# ins[0].append(boiler.waterInRatePerSecond)
# ins[1].append(boiler.GetInflowWaterTemp())
# ins[2].append(boiler.waterOutRatePerSecond)
# ins[3].append(boilerController.temperatureSetPoint)
# ins[4].append((i * 10) * simulator.timeDilation)

#preds = model.predict(ytest)  
#preds = model(ytest)

yo = []

distPreds = []
statePreds = []

inputarr = disturbs[0]
internalState = inFeed[0]
print(targetDisturbs.shape)




forecasterLength = 60

limit = disturbs.shape[0] // forecasterLength

for i in range(limit):
    actual = i * forecasterLength
    inputarr = disturbs[actual]#numpy.expand_dims(disturbs[actual], 0)
    internalState = inFeed[actual]#numpy.expand_dims(states[i], 0)

    print("Working: {}/{} (N. {}, t:{})".format(i, limit, actual, actual * step * dilation))

    for j in range(forecasterLength):
        ytest = numpy.expand_dims(inputarr, 0)
        ystate= numpy.expand_dims(internalState, 0)        





        #predtime = tf.squeeze(preds, 0)[0].numpy()
        #yo.append(predtime)
        #print(tf.squeeze(preds, 0)[-1,0].numpy())

        ## Next Timestep
        forecast = forecastmodel.predict(ytest)
        forebar = tf.squeeze(forecast, 0).numpy()
        distPreds.append(forebar)

        # don't predict the state here
        preds = predmodel.predict(ystate)
        forepred = tf.squeeze(preds, 0).numpy()
        statePreds.append(forepred)

        #print(preds)

        preds = [numpy.concatenate((forecast[0], preds[0]))]

        #print(preds)

        #lElement = inputarr[-1]
        # print()
        # print(forebar)
        # print(rawyins[i+1])
        #sys.exit()
        inputarr = numpy.concatenate((inputarr[1:], forecast))
        internalState = numpy.concatenate((internalState[1:], preds))
        #print(inputarr[1:])
        #inputarr = numpy.concatenate(inputarr[1:], [1, lElement[1], lElement[2], lElement[3]])
        #inputarr = numpy.concatenate((inputarr[1:], [numpy.array([lElement[0], lElement[1], 65, i * dilation])]))



#print(preds.shape)



maxY = 105
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

dataP = numpy.array(statePreds).transpose()[2]
dataT = targetStates.transpose()[2]
print(dataT.flatten().squeeze().shape)
print(len(dataP))
dataT = list(dataT.flatten())
dataP = list(dataP.flatten())




#dataS = distPreds[2]




# dataT = numpy.concatenate([dataT, [boiler.boilerPercent * 100]])
# dataX = numpy.concatenate([dataX, [boiler.waterOutRatePerSecond * 100]])
# dataS = numpy.concatenate([dataS, [boiler.waterVolCurrent]])

#removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

#dra.set_ydata(dataP[removalCutter:])
at = 0#max((len(dataP) - 1) - iTime, 0)
at = min(len(dataP), len(dataT))
# dataP = dataP[at:]
# dataT = dataT[at:]
# dataS = dataS[at:]
# dataX = dataX[at:]
dataP = dataP[:at]
dataT = dataT[:at]
# dataS = dataS[:at]
# dataX = dataX[:at]

print(len(dataP))
print(len(dataT))

dra.set_xdata(numpy.arange(0, len(dataP)) * dilation)
dra.set_ydata(dataP)
two.set_xdata(numpy.arange(0, len(dataT)) * dilation)
two.set_ydata(dataT)



# three.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
# three.set_ydata(dataS)
# four.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
# four.set_ydata(dataX)

ax.set_xlim(left=-5, right=len(dataP) * dilation +5)

#ax = pd.plot()
fig.canvas.draw()
fig.canvas.flush_events()

fig.savefig("ML_4.png".format(Utils.TimeNow()))

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
