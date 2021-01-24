#!/usr/bin/env python3

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


cost = 0
costml = 0
stepsSinceLastTrain = 0
rtTimes = 0
tholdRTTimes = 0

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