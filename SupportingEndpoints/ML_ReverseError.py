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
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#matplotlib.interactive(True)
#matplotlib.use("TkAgg") 
import numpy
import math

#seqLength = 60 * 24
dilation = Utils.dilation
seqLength = Utils.seqLength
step = Utils.step

disabledisturb = False

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
        layers.LSTM(1024, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(1024, return_sequences=True),
        #layers.GRU(64, return_sequences=True),
        #layers.LSTM(128, return_sequences=True),
        layers.LSTM(1024, return_sequences=False),
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
        layers.LSTM(1024, return_sequences=True),
        layers.Dropout(0.1),
        #layers.LSTM(1024, return_sequences=True),
        #layers.GRU(64, return_sequences=True),
        #layers.LSTM(128, return_sequences=True),
        layers.LSTM(1024, return_sequences=False),
        layers.Dropout(0.1),
        #layers.LSTM(64, return_sequences=True),
        #layers.LSTM(64, return_sequences=True),
        
        layers.Dense(1024, activation='relu'),
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

epochlies = 3

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

#forecastmodel.fit(disturbs, targetDisturbs, validation_data=(val_disturbs, val_targetDisturbs), batch_size=16, epochs=epochlies)
predmodel.fit(inFeed, targetStates, validation_data=(inVal, val_targetStates), batch_size=128, epochs=epochlies)





##### ##### ########## ##### #####
## Build History
##

# Build Forward
# There's no benefit to building backwards since each step is discrete

# How far back?
backstep = seqLength

pairwiseErrors = []

for i in range(backstep):
    itu = numpy.expand_dims(inFeed[i], 0) 
    preds = predmodel.predict(itu)
    
    forecast = tf.squeeze(preds, 0).numpy()
    tStat = targetStates[i]

    pairwiseErrors.append(forecast - tStat)

pairwiseErrors = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip)

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

fig = Utils.MakeScreen(dataP, dataT, dataS, dataX)

fig.savefig("ML_{}.png".format(Utils.TimeNow()))

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
