from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import datetime
import numpy
import math
import unicodedata
import re
import modred
import pickle
import control
import pysindy
import pydmd
import matplotlib
import matplotlib.pyplot
import matplotlib.patches
import pandas
import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

dilation = 1
seqLength = 300#5 * 12
step = 5
offset = 230
Weights = [1, 1, 1, 1, 1, 1, 0.01, 0.100]
ErrorWeights = [0, 0, 0, 0, 1.1, 1, 0.01]#, 0.01]
StateOnlyWeight = [0, 0, 0, 0, 1, 1, 0.001, 0.1]
bFlip = False

def GetSimulatorFrequency():
    return 100 / dilation


def GetDetectableFrequency():
    return 0.5 / (GetTimeStep())

def GetTimeStep():
    return step * dilation


def ShuffleTogether(a ,b):
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def TailState(x, minTail =100, rowsAreSamples=False):
    if not rowsAreSamples:
        nFeatures = x.shape[0]
        nSamples = x.shape[1]
        xt = numpy.copy(x).transpose()
    else:
        nFeatures = x.shape[1]
        nSamples = x.shape[0]
        xt = numpy.copy(x) # Skip Transpose

    nTailLength = minTail#max(nSamples // 100, minTail)
    

    for i in range(nTailLength):
        xt[i] = xt[i] * math.exp(-i / (nTailLength / 4) )

    for i in range(nTailLength):
        #print(xt[-nTailLength + i], math.exp(-i / (nTailLength / 4) ) )
        xt[-nTailLength + i] = xt[-nTailLength + i] * math.exp(-i / (nTailLength / 4) )

    # begArray = numpy.zeros(nFeatures * nTailLength).reshape((nFeatures, nTailLength))
    # endArray = numpy.zeros(nFeatures * nTailLength).reshape((nFeatures, nTailLength)).tranpose()
    

    # for i in range(nTailLength):
    #     endArray[i] = x[-1] * math.exp(-i / (nTailLength / 4) )

    # endArray = 
    if rowsAreSamples:
        return xt
    else:
        return xt.transpose()

def MakeAccError(inVal, flip=True, useAbs=True):
    pairwiseErrors = numpy.array(inVal)

    # Abs Errors
    if useAbs:
        pairwiseErrors = numpy.absolute(pairwiseErrors)

    #pairwiseErrors = numpy.cumsum(absPairwise)

    if len(pairwiseErrors.shape) > 1:
        # Sum
        print(pairwiseErrors[0])
        pairwiseErrors = numpy.sum(pairwiseErrors, axis=1)
        print(pairwiseErrors[0])

    if flip:
        pairwiseErrors = numpy.flip(pairwiseErrors)
    #print(pairwiseErrors[0])
    pairwiseErrors = numpy.cumsum(pairwiseErrors)
    if flip:
        pairwiseErrors = numpy.flip(pairwiseErrors)

    return pairwiseErrors

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = value.replace(b".", b"-")
    value = re.sub(b'[^\w\s-]', b'', value).strip()#.lower()
    value = re.sub(b'[-\s]+', b'-', value)
    # ...
    return value.decode()


def TimeNow():
    #print("WEEEEEE")
   
    return slugify(datetime.datetime.now().isoformat())


#https://stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray
def window_stack(a, stepsize=1, width=3, nummax=None):
    if not nummax:
        indexer = numpy.arange(width)[None, :] + stepsize * numpy.arange(a.shape[0] - (width))[:, None]
    else:
        indexer = numpy.arange(width)[None, :] + stepsize * numpy.arange(nummax)[:, None]

    return a[indexer]
    #return numpy.hstack( a[i:1 + i-width or None:stepsize] for i in range(0, width))


def MakeCacheName(x,y, td, width, modRange, disable, boilerPower=10000, tankageLimits=(5,75,80), initWaterTemp=30, initCapacity=35, step=1, stack=True, seed=0):
    p1 = "{}_{}_{}_{}".format(x,y,td, modRange)
    p2 = "T" if disable else "F"
    p3 = "{}_{}.{}.{}_{}_{}".format(boilerPower, tankageLimits[0], tankageLimits[1], tankageLimits[2], initWaterTemp, initCapacity)
    p4 = "S{}_RNG{}".format(step,seed)
    #p5 = "T" if stack else "F"

    name = "_".join([p1,p2,p3,p4])#,p5])
    return name + ".cache"

def HandleStacking(disturbs, states, stack, width, tailed=False):
    if stack:
        disturbStacked = window_stack(disturbs, stepsize=1, width=width)
        stateStacked = window_stack(states, stepsize=1, width=width)

        # with open(cachename, "wb+") as f:
        #     pickle.dump((disturbStacked, stateStacked, disturbs[width:], states[width:]), f)

        return disturbStacked, stateStacked, disturbs[width:], states[width:]
    else:
        if tailed:
            disturbs = TailState(disturbs, offset, rowsAreSamples=True)
            states = TailState(states, offset, rowsAreSamples=True)
            return disturbs, states, disturbs[width:], states[width:]
        else:
            return disturbs, states, disturbs[width:], states[width:]

def MakeData(x,y, td, width, modRange, disable, boilerPower=10000, tankageLimits=(5,75,80), initWaterTemp=30, initCapacity=35, step=1, stack=True, seed=0, tailed=False):

    cachename = MakeCacheName(x,y,td,width,modRange,disable,boilerPower,tankageLimits,initWaterTemp,initCapacity,step,stack,seed)
    print(cachename)

    #Cache
    if os.path.exists(cachename):
        print("Using Cache")
        with open(cachename, "rb+") as f:
            a,b = pickle.load(f)
            return HandleStacking(a, b, stack, width, tailed)
            #return a,b,c,d



    simulator = CSimulator(td, 600000)
    #simulator = CSimulator(1, 200000)

    spTemp = y

    lowLevelAlarm, highLevelAlarm, totalCapacity = tankageLimits

    boiler = simulator.SpawnObject(ABoiler, boilerPower, initCapacity, totalCapacity, initWaterTemp)
    boilerController = simulator.SpawnObject(ABoilerController, lowLevelAlarm, highLevelAlarm, spTemp, seed) # Heating to 95
    #boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
    boilerController.Possess(boiler)

    boiler.SetInflowWaterTemp(24)
    boiler.SetInflowRate(0.0)
    boiler.SetOutflowRate(0.0001)

    if(disable):
        boilerController.SetDisableDisturbance()
    else:
        boilerController.SetEnableDisturbance()

    measurements = x // step
    #ins = [[],[],[],[],[],[]]

    # In Temperature, Inflow Rate, Outflow Rate
    user_disturbances = [[] ,[], [], []] 

    # Water Level, Boiler Setpoint PID, Water Temperature + disturbances
    stateInformation = [[],[],[]]

    # Nothing here!
    outputs = []

    outs = [[]]

    for i in range(measurements):
        if i % 100 == 0:
            print("Time {}: {}".format(i * td * step, boiler.GetBoilerWaterTemp()))
        
        pastTickTemp = boiler.GetBoilerWaterTemp()

        # Simulate `step` seconds
        simulator.SimulateNTicks(step * 100, 1/100)

        if(not disable):
            mod = math.sin(i * 0.05) * modRange #** 640 * 30
            boilerController.SetTarget(spTemp - math.floor(mod))


        # Flow Rate
        user_disturbances[0].append(boiler.waterInRatePerSecond)

        # Temperature
        user_disturbances[1].append(boiler.GetInflowWaterTemp())

        # Setpoint
        user_disturbances[2].append(boilerController.temperatureSetPoint)

        # Out Flow Rate
        user_disturbances[3].append(boiler.waterOutRatePerSecond)



        # Out Flow Temperature
        stateInformation[0].append(boiler.GetBoilerWaterTemp())

        # State Volume
        stateInformation[1].append(boiler.waterVolCurrent)

        # State Power
        stateInformation[2].append(boiler.boilerPerformance * boiler.boilerPercent)

        #user_disturbances[4].append(boilerController.PID.iVal)
        
        #stateInformation[2].append(boiler.boilerPercent * 100)

        # State Temperature
        #stateInformation[2].append(boiler.GetBoilerWaterTemp())


        # stateInformation[0].append(boiler.waterInRatePerSecond)
        # stateInformation[1].append(boiler.GetInflowWaterTemp())
        # stateInformation[2].append(boiler.waterOutRatePerSecond)
        # stateInformation[0].append(boilerController.temperatureSetPoint)
        # stateInformation[1].append(boiler.waterVolCurrent)
        # stateInformation[2].append(boiler.GetBoilerWaterTemp())
        # #stateInformation[3].append(boiler.boilerPercent * boiler.boilerPerformance)
        # stateInformation[3].append(boiler.boilerPercent * 100)# * boiler.boilerPerformance)
        
        # print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
        # print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
        # print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
        # print("[TIME {:.02f}s] Power Perc: {:.02f}%".format((i + 1) * simulator.timeDilation, boiler.boilerPercent * 100))
        # print("[TIME {:.02f}s] PID: {:.02f}i".format((i + 1) * simulator.timeDilation, boilerController.PID.iVal))
        # print("[TIME {:.02f}s] PIDdbg: {}".format((i + 1) * simulator.timeDilation, boilerController.PID.dbgLastReturn))

        # #ins[3].append(boiler.waterVolCurrent)
        # ins[0].append(boiler.waterInRatePerSecond)
        # ins[1].append(boiler.GetInflowWaterTemp())
        # ins[2].append(boiler.waterOutRatePerSecond)
        # ins[3].append(boilerController.temperatureSetPoint)
        # ins[4].append(boiler.waterVolCurrent)
        # #ins[5].append(boiler.boilerPercent)
        # ins[5].append(boiler.GetBoilerWaterTemp())
        # #ins[4].append((i * 10) * simulator.timeDilation)

        # #ins[3].append(boiler.boilerPercent)


        #ins[3].append(pastTickTemp)
        #ins[3].append(boiler.boilerPercent)
        #ins[4].append(boiler.waterVolCurrent)

        outs[0].append(boiler.GetBoilerWaterTemp())


    disturbs = numpy.array([numpy.array(xi) for xi in user_disturbances]).transpose()
    states = numpy.array([numpy.array(xi) for xi in stateInformation]).transpose()

    with open(cachename, "wb+") as f:
            pickle.dump((disturbs, states), f)

    return HandleStacking(disturbs, states, stack, width, tailed)

    # if stack:
    #     disturbStacked = window_stack(disturbs, stepsize=1, width=width)
    #     stateStacked = window_stack(states, stepsize=1, width=width)

    #     # with open(cachename, "wb+") as f:
    #     #     pickle.dump((disturbStacked, stateStacked, disturbs[width:], states[width:]), f)

    #     return disturbStacked, stateStacked, disturbs[width:], states[width:]

    # else:
    #     # with open(cachename, "wb+") as f:
    #     #     pickle.dump((disturbs, states, disturbs[width:], states[width:]), f)
    #     #return disturbs[:-width], states[:-width], disturbs[width:], states[width:]
    #     return disturbs, states, disturbs[width:], states[width:]

    # nins = numpy.array([numpy.array(xi) for xi in ins]).transpose()
    # nouts = numpy.array([numpy.array(xi) for xi in outs]).transpose()
    # rnins = nins#window_stack(nins, stepsize=1, width=width)
    # #nouts = window_stack(nouts, stepsize=1, width=width)
    # nouts = nouts#nouts[width:]

    # #print(nstack.shape)
    # print(rnins.shape)
    # print(nouts.shape)

    # ninstest = numpy.expand_dims(rnins[0], 0)
    # #nouts = numpy.expand_dims(nouts, 1)



    # return rnins, nouts, ninstest, nins[width:]

# ran = numpy.arange(7)
# ranStack = window_stack(ran, stepsize=1, width=3)
# rO = ran[3:]

# print(ranStack)
# print(rO)

# print(TimeNow())


def MakeScreen(dataP, dataT, dataS, dataX, maxY=240, dataPLabel = "Simulated Truth", dataTLabel = "Modelled Value", dataSLabel = "Signed Error", dataXLabel = "Abs. Error"):  
    maxTDPI = 240
    resolution = numpy.array((1920, 1080))
    TargetDPI = maxTDPI

    solvedSize = resolution / TargetDPI

    fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)#figsize=(lScalar*scaleWidth, min((lScalar * scaleWidth*scaleWidth / 16), max(16, (lScalar * 18 / 16)))))
    ax = matplotlib.pyplot.axes()
    #ax2 = ax.twin()
    dra, = ax.plot([],[])#, linestyle="--")
    dra.set_label(dataPLabel)
    two, = ax.plot([],[])
    two.set_label(dataTLabel)
    three, = ax.plot([],[])
    three.set_label(dataSLabel)
    four, = ax.plot([],[])
    four.set_label(dataXLabel)

    ax.legend([dataPLabel, dataTLabel, dataSLabel, dataXLabel])

    iTime = 30

    color = (0.05,0.05,0.05)
    # ax.plot([-5,iTime+5], [60,60])
    # ax.plot([-5,iTime+5], [30,30])
#    ax.axhline(65, linestyle='--', color='red')
    ax.yaxis.grid(True, color='white')


    ax.set_facecolor(color)
    fig.set_facecolor(color)

    ax.set_xlabel("Time (Seconds)", color='white')
    ax.set_ylabel("Heat (°C) / Boiler Power Level (%)", color='white')
    #ax.set_ylim(top=maxY, bottom=-1)
    ax.set_ylim(top=maxY, bottom=-maxY)
    #ax.set_xlim(left=-5, right=iTime+5)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

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





    #dataS = distPreds[2]




    # dataT = numpy.concatenate([dataT, [boiler.boilerPercent * 100]])
    # dataX = numpy.concatenate([dataX, [boiler.waterOutRatePerSecond * 100]])
    # dataS = numpy.concatenate([dataS, [boiler.waterVolCurrent]])

    #removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

    #dra.set_ydata(dataP[removalCutter:])
    at = 0#max((len(dataP) - 1) - iTime, 0)
    at = min(min(len(dataP), len(dataX)), min(len(dataS), len(dataT)))

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

    return fig


def MakeCSV(x, outpath):
    hey = pandas.DataFrame(x)

    hey.to_csv(outpath)


def DistanceImagToPole(x):
    rScale = x.real
    iScale = x.imag

    distSq = rScale * rScale + iScale * iScale

    return numpy.sqrt(distSq)

def DistanceToZero(x, thresh=1):
    #dSq = numpy.power(x, 2)
    accumDist = 0

    for i in x:
        ds = DistanceImagToPole(i)
        accumDist += 1 if ds > thresh else 0# ds # ds
        #accumDist += ds * (1 / len(x))

    return accumDist

# Generates a low number when close to zero
def GetFitness(x):
    return DistanceToZero(x, 1.1)


def GenerateModel(disturbs, states):

    keras.backend.clear_session()

    predmodel = keras.Sequential(
    [
        # #layers.Embedding(input_shape=(100, 3), output_dim=128),
        # layers.Input(shape=(seqLength, states.shape[2] + disturbs.shape[2])),
        # # layers.LSTM(1024, return_sequences=True),
        # # layers.Dropout(0.1),
        # # layers.LSTM(1024, return_sequences=True),
        # #layers.GRU(64, return_sequences=True),
        # #layers.LSTM(128, return_sequences=True),
        # layers.GRU(256, return_sequences=True),
        # layers.GRU(256, return_sequences=True),
        # #layers.LSTM(1024, return_sequences=False),
        # layers.Dropout(0.1),
        # layers.GRU(256, return_sequences=False),
        # #layers.LSTM(64, return_sequences=True),
        # #layers.LSTM(64, return_sequences=True),
        
        # layers.Dense(256, activation='relu'),
        # layers.Dense(states.shape[2])


        # Larger network
        layers.Input(shape=(seqLength, states.shape[2] + disturbs.shape[2])),
        layers.LSTM(1024, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(1024, return_sequences=False),
        #layers.GRU(64, return_sequences=True),
        #layers.LSTM(128, return_sequences=True),
        #layers.LSTM(512, return_sequences=False),
        #layers.LSTM(64, return_sequences=True),
        #layers.LSTM(64, return_sequences=True),
        
        layers.Dense(256, activation='relu'),
        layers.Dense(states.shape[2])
    ])

    predmodel.compile(
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    return predmodel

def CreateOKIDERA(l1, l2, i, step, dilation):
    #l1 = numpy.flip(l1, 1) 
    #l2 = numpy.flip(l2, 1) 
    kalman = modred.OKID(l1, l2, i)
    era = modred.ERA(verbosity=0)

    mco = int(math.floor( (len(kalman) - 1) / 2))
    a,b,c = era.compute_model(kalman, 3*4*3)#, mco, mco)
    #b *= 1/step
    #a,b,c = modred.era.compute_ERA_model(kalman, 1500)

    # print("Mats")
    # #print(a)
    # #print(b)
    # print(c)
    # print(c == numpy.identity(4))
    # print()

    #c = numpy.identity(3) * 0.5 + c * 0.5

    # Some *real* asserts
    assert(a.shape != (0,0))
    assert(b.shape != (0,0))
    assert(c.shape != (0,0))
    #@c = numpy.identity(3)

    ### Test Asserts
    assert(a.shape == (3,3))
    assert(b.shape == (3,4))
    assert(c.shape == (3,3))

    newScore = GetFitness(numpy.linalg.eigvals(a))
    #print(newScore)

    asb = control.ss(a,b,c, numpy.zeros((c.shape[0], b.shape[1])), step)
    poles = control.pole(asb)
    score = GetFitness(poles)

    return asb, score

def GetBestOKID(l1, l2, minmcs=1, markovs=20):
    bestIndex = 0
    #bestScore = 1 # Unstable above 1
    bestScore = 5# Allow at least 1 failed pole

    for i in range(minmcs, markovs):
        #print("Attempting to get {} markovs ({}/{})".format(i,i-minmcs,markovs-minmcs))
        try:
            asb, score = CreateOKIDERA(l1,l2,i,step,dilation)

            #print("{} scored {}".format(i, score))

            if score <= bestScore:
                bestIndex = i
                bestScore = score

                # Async start the process

        
        except Exception as e:
            #print("Fail on {}. {}".format(i,e))
            pass
        except IndexError as e:
            raise e


    #print("Using {} markovs".format(bestIndex))

    return CreateOKIDERA(l1, l2 ,bestIndex, step, dilation)

def GetBestDMDnc(l1, l2):
    model = pydmd.DMD(svd_rank=-1)

    #print(l1.shape, l2.shape)

    #Resize
    nl1 = l1.transpose()[:-1].transpose()
    #nl2 = l2.transpose()[:300].transpose()

    model.fit(l2, nl1)

    #model.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8, 8))

    #input()

    return model, 1

def GetBestDMD(l1, l2):
    model = pydmd.DMDc(svd_rank=-1)

    #print(l1.shape, l2.shape)

    #Resize
    nl1 = l1.transpose()[:-1].transpose()
    #nl2 = l2.transpose()[:300].transpose()

    model.fit(l2, nl1)

    #model.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8, 8))

    #input()

    return model, 1

def GetBestMrDMD(l1, l2):
    model = pydmd.MrDMD(svd_rank=-1, max_level=7)

    #print(l1.shape, l2.shape)

    #Resize
    nl1 = l1.transpose()[:-1].transpose()
    nl2 = l2.transpose().transpose()

    model.fit(l2)#, nl1)

    #model.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8, 8))

    #input()

    return model, 1

def GetBestSindy(l1, l2):
    features = pysindy.PolynomialLibrary(degree=4)
    optimiser = pysindy.STLSQ(threshold=0.0025, max_iter=20000000)
    differential = pysindy.FiniteDifference(order=2)

    #print(l2.shape)

    model = pysindy.SINDy(
        optimiser,
        features, 
        differential,
        feature_names=["x","y"],
        discrete_time=True
        )

    model = pysindy.SINDy(discrete_time=True)

    #model.fit(l2.transpose(), u=l1.transpose(), t=step, quiet=False)
    model.fit(l2.transpose(), u=l1.transpose(), t=step, quiet=False)

    model.print()
    model.coefficients()

    return model, 1



outnames = ["Tank Temperature", "Tank Water Volume", "Boiler Wattage"]
innames = ["Inflow Rate", "In Temperature", "PID Setpoint", "Outflow Rate"]

def NameRowCol(x):
    for ax, col in zip(x[0], outnames):
        #ax.set_title(col)
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, col in zip(x[:,0], innames):
        #.set_ylabel(col, rotation=0, size='large')
        ax.annotate(col, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')


    return 
    

def CreateBodePlots(system, savename, appendName):
    omega = control.freqplot.default_frequency_range(system, Hz=True, number_of_samples=10000)

    #omega = numpy.arange(1e-6, omega[-1], 0.05)

    mag, phase, freq = system.freqresp(omega)

    maxY = 105
    maxTDPI = 300
    resolution = numpy.array((3840, 2160))
    TargetDPI = maxTDPI
    scalar = 2

    solvedSize = resolution / TargetDPI
    #TargetDPI = TargetDPI * 2
    #solvedSize = solvedSize * 2

    # Matplotlib
    #figSize = (18,9)
    fig, axes = matplotlib.pyplot.subplots(mag.shape[1], mag.shape[0], dpi=TargetDPI, figsize=solvedSize)
    fig2, axes2 = matplotlib.pyplot.subplots(mag.shape[1], mag.shape[0], dpi=TargetDPI, figsize=solvedSize)

    freq = freq / (2*math.pi)

    # print(freq)
    NameRowCol(axes)
    NameRowCol(axes2)

    fig.suptitle("Bode Plot - Magnitude")
    fig2.suptitle("Bode Plot - Phase")

    for o, outputs in enumerate(zip(mag,phase)):
        outputMag, outputPhase = outputs
        # print(freq.shape)
        # print(outputMag.shape)
        # print(outputPhase.shape)

        for i, inputs in enumerate(zip(outputMag, outputPhase)): 
            inputMag, inputPhase = inputs
            # print(freq.shape)
            # print(inputMag.shape)
            # print(inputPhase.shape)

            phaseInDegs = inputPhase * (180 / math.pi)

            axes[i, o].plot(freq, inputMag)
            axes[i, o].set_xscale("log")
            axes[i, o].grid(which="both")
            ##axes[o, i].plot(freq, inputPhase)
            axes[i, o].set_xlabel("Frequency (Hz)")
            axes[i, o].set_ylabel("Magnitude")


            axes2[i, o].plot(freq, phaseInDegs)
            ##axes[o, i].plot(freq, inputPhase)
            axes2[i, o].set_xscale("log")
            axes2[i, o].grid(which="both")
            axes2[i, o].set_xlabel("Frequency (Hz)")
            axes2[i, o].set_ylabel("Phase (Degrees)")

            axes2[i, o].set_ylim(top=180, bottom=-180)

    if appendName:
        fig.savefig("{}_{}_{}_magnitude.png".format(savename, step, dilation))
        fig2.savefig("{}_{}_{}_phase.png".format(savename, step, dilation))
    else:
        fig.savefig("{}_magnitude.png".format(savename))
        fig2.savefig("{}_phase.png".format(savename))

    fig.clf()
    fig2.clf()

    
def CreatePolePlots(system, name, appendName):
    maxY = 105
    maxTDPI = 375
    resolution = numpy.array((2048, 2048))
    TargetDPI = maxTDPI

    solvedSize = resolution / TargetDPI
    fx = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)
    ax = fx.gca()
    #ax.set_ylim(-2, 2)
    #ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    
    for i in system.pole():
        rScale = i.real
        iScale = i.imag
        ax.plot([rScale],  [iScale],  'rx--')

    ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.add_patch(matplotlib.pyplot.Circle((0,0), 1, fill=False))
    #matplotlib.pyplot.legend()
    ax.set_title("Poles")
    if appendName:
        fx.savefig("{}_{}_{}_poles.png".format(name, step, dilation))
    else:
        fx.savefig("{}_poles.png".format(name))
    
    fx.clf()

def CreateBodeAndPolePlots(system, name, appendName=False):
    CreateBodePlots(system, name, appendName)
    CreatePolePlots(system, name, appendName)






def DetectorFunction(history, feedback, timeBase, system):
    # Transform the history U-vector
    PreU = history.transpose()[:4].transpose()
    # Transform with B
    A,B,C,D = control.ssdata(system)

    ModifiedOutput = numpy.zeros((PreU.shape[0], 3))

    for signalIter in range(PreU.shape[0]):
        workingData = PreU[signalIter]
        ModU = B.dot(workingData)
        ModifiedOutput[signalIter] = ModU[0]

    for signalIter in range(3):

        print("Stepping ", signalIter)
        print(ModifiedOutput.shape)
        workingData = ModifiedOutput.transpose()[signalIter]#[signalIter]
        print(workingData.shape)
        input()



        freq, sig_f = czt.time2freq(numpy.arange(timeBase, timeBase + workingData.shape[0]) * GetTimeStep(), workingData)

        # matplotlib.pyplot.figure(1)
        # #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
        # matplotlib.pyplot.plot(freq / 1e3,  numpy.angle(sig_f),   'ro--', label='CZT')
        # matplotlib.pyplot.xlabel("Frequency (kHz)")
        # matplotlib.pyplot.ylabel("Signal phase")
        # ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
        # #matplotlib.pyplot.legend()
        # matplotlib.pyplot.title("Frequency-domain")

        # matplotlib.pyplot.figure(2)
        # #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
        # matplotlib.pyplot.plot(freq / 1e3,  numpy.abs(sig_f),   'ro--', label='CZT')
        # matplotlib.pyplot.xlabel("Frequency (kHz)")
        # matplotlib.pyplot.ylabel("Signal mag")
        # ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
        # #matplotlib.pyplot.legend()
        # matplotlib.pyplot.title("Frequency-domain")

        # Decompile to real and imag compo
        # zU = e^(z * dt)
        # ln(zU) = z * dt
        # z = ln(zU) / dt
        
        poles = 0

        for cmptePle in zip(freq, sig_f):
            # So
            # We would like to compute the real and imaginary components of the number
            # sig_F
            #tScore = Utils.GetFitness(cmptePle[1])

            rScale = cmptePle[1].real
            iScale = cmptePle[1].imag

            print(rScale, iScale)

            input()

            matplotlib.pyplot.figure(signalIter)
            #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
            matplotlib.pyplot.plot([rScale],  [iScale],   'ro--', label='CZT')
            ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
            matplotlib.pyplot.xlabel("Real")
            matplotlib.pyplot.ylabel("Imaginary")
            #matplotlib.pyplot.legend()
            matplotlib.pyplot.title("Frequency-domain")

        
        input()    


    exit()




    return False


def GetDMDcRetrainMagnitudes():
    # Allocate
    Amats = []
    Bmats = []
    Cmats = []
    Dmats = []

    # Get all files
    _, _, names = next(os.walk("./"), (None, None, []))


    runningLimit = 0
    for i in names:
        if i.endswith(".pickle"):
            print(i)
            _, pt1 = i.split("RT", 1)
            index, _ = pt1.split(".", 1)

            index = int(index)

            if index > runningLimit:
                runningLimit = index


    print("Found {} files".format(runningLimit + 1))


    # Pull into memory
    for i in range(runningLimit + 1):
        with open("DMDc_RT{}.pickle".format(i), "rb+") as f:
            A,B,C,D = pickle.load(f)

            Amats.append(A)
            Bmats.append(B)
            Cmats.append(C)
            Dmats.append(D)

    # Convert
    Amats = numpy.array(Amats)
    Bmats = numpy.array(Bmats)
    Cmats = numpy.array(Cmats)
    Dmats = numpy.array(Dmats)


    # Calculate Differences
    ATDiff = numpy.diff(Amats, axis=0)
    BTDiff = numpy.diff(Bmats, axis=0)
    CTDiff = numpy.diff(Cmats, axis=0)
    DTDiff = numpy.diff(Dmats, axis=0)

    # Sum to single abs
    ADiff = numpy.sum(numpy.abs(ATDiff))
    ADiffx = numpy.sum(numpy.abs(ATDiff), axis=0)
    BDiffx = numpy.sum(numpy.abs(BTDiff), axis=0)

    ADiffx = numpy.sum(ATDiff, axis=0)
    BDiffx = numpy.sum(BTDiff, axis=0)

    print(BDiffx.shape)

    AMatrixDifferenceMags = numpy.sum(numpy.sum(numpy.abs(ATDiff), axis=2), axis=1)
    BMatrixDifferenceMags = numpy.sum(numpy.sum(numpy.abs(ATDiff), axis=2), axis=1)

    return AMatrixDifferenceMags, BMatrixDifferenceMags
