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
import pydmd
import pysindy
from past.utils import old_div


simulator = CSimulator(1, 600000)
#simulator = CSimulator(1, 200000)

spTemp = 55
spTarg = 75
seed = 0
step = Utils.step
dlp = 150
nme = None

if len(sys.argv) > 1:
    nme = sys.argv[1]

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

dmdModel = None
cost = []
costToRT = []
rtReason = []
stepsSinceLastTrain = 0
rtTimes = 0
tholdRTTimes = 0


# For DMD
cacheA = None
cacheB = None

with open("Pickle.dmd", "rb+") as f:
    dmdModel = pickle.load(f)

if nme is not None:
    with open(nme, "rb+") as f:
        print("LD")
        i = pickle.load(f)
        print(i)
        cacheA, cacheB, _, _ = i
        cacheA = numpy.array(cacheA)
        cacheB = numpy.array(cacheB)

def DMDc_EvalFunction(history, feedback, i):
    global cost
    global dmdModel
    global cacheA
    global cacheB

    # Check the caches are intact
    if cacheA is None or cacheB is None:
        eigs = numpy.power(dmdModel.eigs, old_div(dmdModel.dmd_time['dt'], dmdModel.original_time['dt']))
        cacheA = dmdModel.modes.dot(numpy.diag(eigs)).dot(numpy.linalg.pinv(dmdModel.modes))
        cacheB = dmdModel.B

    ht = history.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()

    nl1 = l1.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()

    U = l1.transpose()[-1]
    X = l2.transpose()[-1]

    evalBeginTime = time.perf_counter()

    out = cacheA.dot(X) + cacheB.dot(U)

    evalEndTime = time.perf_counter()
    cost.append(evalEndTime - evalBeginTime)

    #out = dmdModel.reconstructed_data(nl1).transpose()[-1]

    return out, []

##### ##### ########## ##### #####
## Sindy
##

sindyModel = None

with open("Pickle.sindy", "rb+") as f:
    sindyModel = pickle.load(f)

def Sindy_EvalFunction(history, feedback, i):
    global sindyModel
    global cost

    ht = history.transpose()
    l1 = ht[:4].transpose()[-1]
    l2 = ht[4:].transpose()[-1]

    evalBeginTime = time.perf_counter()

    out = sindyModel.simulate(l2, 1, u=l1)

    evalEndTime = time.perf_counter()
    cost.append(evalEndTime - evalBeginTime)

    return out[0], []

#### SPSH
model = None

with open("Pickle.era", "rb+") as f:
    model = pickle.load(f)

predmodel = keras.models.load_model("model.tensorflow")

def FollowingEvalFunction(history, feedback, i):
    return history[-1][4:] + (history[-1][4:] - history[-2][4:]), []

def NoFunction(history, feedback, i):
    return history[-1][4:], []

def EvalFunction(history, feedback, i):
    global model
    global cost

    #asU = history.transpose()[:4]

    ht = history.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()

    U = l1.transpose()[-1]
    X = l2.transpose()[-1]

    A,B,C,D = control.ssdata(model)

    evalBeginTime = time.perf_counter()

    out = numpy.array(A.dot(X) + B.dot(U)).squeeze()

    evalEndTime = time.perf_counter()
    cost.append(evalEndTime - evalBeginTime)

    #print((out[0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0]).shape)

    return out, []


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

def ML_EvalFunction(history, feedback, i):
    ytest = numpy.expand_dims(history[:Utils.seqLength], 0)
    forecast = predmodel.predict(ytest)
    forebar = tf.squeeze(forecast, 0).numpy()

    return forebar, []


graphing = Graphing.AGraphHolder(seed, spTemp, spTarg, dlp)

results = graphing.TestOffsetWidth(EvalFunction, 2048)

with open("backtrackStab.dat", "wb+") as f:
    pickle.dump(results, f)

del results

# mlresults = graphing.TestOffsetWidth(ML_EvalFunction, 50)

# with open("backtrackStabWithML.dat", "wb+") as f:
#     pickle.dump(mlresults, f)
