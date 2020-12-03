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


graphing = Graphing.AGraphHolder(seed, spTemp, spTarg, dlp)

results = graphing.TestOffsetWidth(EvalFunction, 500)

with open("backtrackStab.dat", "wb+") as f:
    pickle.dump(results, f)

del results

mlresults = graphing.TestOffsetWidth(ML_EvalFunction, 25)

with open("backtrackStabWithML.dat", "wb+") as f:
    pickle.dump(mlresults, f)
