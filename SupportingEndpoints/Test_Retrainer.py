#!/usr/bin/env python3

import pickle
import control
import modred
#import slycot
from scipy import signal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
#matplotlib.interactive(True)
#matplotlib.use("TkAgg") 
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

def EvalFunction(history, feedback):
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

def ML_EvalFunction(history, feedback):
    ytest = numpy.expand_dims(history[:Utils.seqLength], 0)
    forecast = predmodel.predict(ytest)
    forebar = tf.squeeze(forecast, 0).numpy()

    return forebar, []

def ThresholdFunction(signedError, absoluteError):
    global stepsSinceLastTrain
    global tholdRTTimes

    shouldRetrainOnFixed = stepsSinceLastTrain * Utils.dilation > 100 
    shouldRetrainOnError = numpy.sum(absoluteError) > 1000

    stepsSinceLastTrain += 1

    # Fixed step. Can be rolled into the return bool
    # But it's here to be readable
    if (shouldRetrainOnFixed):
        tholdRTTimes += 1

    if (shouldRetrainOnError or shouldRetrainOnFixed):
        stepsSinceLastTrain = 0
        return True

    return False

def RetrainFunction(history):
    global cost
    global rtTimes
    global model

    rtTimes += 1
    
    beginPerfTime = time.perf_counter()

    print("Retraining")
    ht = history.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()
    model, score = Utils.GetBestOKID(l1, l2)


    # ## Bode
    # transFunc = control.ss2tf(predmodel)
    # scipy.signal.bode(transFunc)
    # #a = control.bode(transFunc)
    # #signal.TransferFunction
    # print(a)

    timePassed = time.perf_counter() - beginPerfTime
    cost += timePassed

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

graphing = Graphing.AGraphHolder(seed, spTemp, spTarg, dlp)

_, results = graphing.TestRetraining(EvalFunction, RetrainFunction, ThresholdFunction, 300)

with open("backtrackRetrain.dat", "wb+") as f:
    pickle.dump(results, f)

print("Cost {} ({}s)".format(cost, cost/rtTimes))
print("Retrains {}. (Fixed RTs {})".format(rtTimes, tholdRTTimes))

cost = 0
costml = 0
stepsSinceLastTrain = 0
rtTimes = 0
tholdRTTimes = 0

del results

graphing = Graphing.AGraphHolder(seed, spTemp, spTarg, dlp)
_, mlresults = graphing.TestRetraining(ML_EvalFunction, ML_RetrainFunction, ThresholdFunction, 300)

with open("backtrackRetrainWithML.dat", "wb+") as f:
    pickle.dump(mlresults, f)

print("Cost {} ({}s)".format(cost, cost/rtTimes))
print("Retrains {}. (Fixed RTs {})".format(rtTimes, tholdRTTimes))
