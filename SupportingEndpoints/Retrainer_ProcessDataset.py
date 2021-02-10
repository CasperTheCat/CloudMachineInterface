#!/usr/bin/env python3

import pickle
import pandas
import numpy
import os
import sys
import matplotlib
import matplotlib.pyplot
import Utils
# matplotlib.interactive(True)
# matplotlib.use("TkAgg") 

def ShouldHighlight(x, y, z):
    indices = x[y] == z
    for i in range(1 , 4):
        working = numpy.roll(x[y] , -i)

        indices = numpy.logical_or(indices, working == z)

    print(indices.any())

    return indices

def CreateAndSaveGraphTri(a, name, labels, limit_min = None, limit_max = None, rtReason = None):
    inShape = a[0].shape[0]
    deltaShape = rtReason.shape[0] - inShape
    LineMaker = rtReason[deltaShape // 2:-(deltaShape // 2)].transpose()

    print("LN {}".format(LineMaker.shape))

    try:
        maxY = 105
        maxTDPI = 96
        resolution = numpy.array((1920, 1080))
        TargetDPI = maxTDPI
        solvedSize = resolution / TargetDPI

        fig, ((ax1, ax2, ax3)) = matplotlib.pyplot.subplots(3,1,sharex=True, dpi=TargetDPI, figsize=solvedSize)
        dra1, = ax1.plot([],[], color="red")
        dra2, = ax2.plot([],[], color="red")
        dra3, = ax3.plot([],[], color="red")

        targetColour = (0.05,0.05,0.05)

        ax1.set_facecolor(targetColour)
        ax2.set_facecolor(targetColour)
        ax3.set_facecolor(targetColour)
        fig.set_facecolor(targetColour)

        ax3.set_xlabel("Seconds", color='white')

        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white') 
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')

        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white') 
        ax2.spines['right'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')

        ax3.spines['bottom'].set_color('white')
        ax3.spines['top'].set_color('white') 
        ax3.spines['right'].set_color('white')
        ax3.spines['left'].set_color('white')
        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')

        if not labels:
            ax1.set_ylabel("Temperature Error (°C)", color='white')
            ax2.set_ylabel("Water Level Error (L)", color='white')
            ax3.set_ylabel("Boiler Power Error (100 W)", color='white')
            dra1.set_label("Boiler Temperature Error (°C)")
            dra2.set_label("Boiler Water Level Error (L)")
            dra3.set_label("Boiler Power Error (100 W)")
        else:
            dra1.set_label(labels[0])
            dra2.set_label(labels[1])
            dra3.set_label(labels[2])
            ax1.set_ylabel(labels[0])
            ax2.set_ylabel(labels[1])
            ax3.set_ylabel(labels[2])

        dra1_1, = ax1.fill(numpy.NaN, numpy.NaN, color="yellow")
        dra1_2, = ax1.fill(numpy.NaN, numpy.NaN, color="blue")
        dra1_3, = ax1.fill(numpy.NaN, numpy.NaN, color="green")
        dra1_4, = ax1.fill(numpy.NaN, numpy.NaN, color="magenta")

        dra2_1, = ax2.fill(numpy.NaN, numpy.NaN, color="yellow")
        dra2_2, = ax2.fill(numpy.NaN, numpy.NaN, color="blue")
        dra2_3, = ax2.fill(numpy.NaN, numpy.NaN, color="green")
        dra2_4, = ax2.fill(numpy.NaN, numpy.NaN, color="magenta")

        dra3_1, = ax3.fill(numpy.NaN, numpy.NaN, color="yellow")
        dra3_2, = ax3.fill(numpy.NaN, numpy.NaN, color="blue")
        dra3_3, = ax3.fill(numpy.NaN, numpy.NaN, color="green")
        dra3_4, = ax3.fill(numpy.NaN, numpy.NaN, color="magenta")

        ax1.legend([dra1, dra1_1, dra1_2, dra1_3, dra1_4], ["Error", "Time-based", "Absolute-based", "Signed-based", "Step-based"])
        ax2.legend([dra2, dra2_1, dra2_2, dra2_3, dra2_4], ["Error", "Time-based", "Absolute-based", "Signed-based", "Step-based"])
        ax3.legend([dra3, dra3_1, dra3_2, dra3_3, dra3_4], ["Error", "Time-based", "Absolute-based", "Signed-based", "Step-based"])

        if limit_min is not None or limit_max is not None:
            lowBound = limit_min if limit_min is not None else -100
            highBound = limit_max if limit_max is not None else 100

            ax1.set_ylim(lowBound, highBound)
            ax2.set_ylim(lowBound, highBound)
            ax3.set_ylim(lowBound, highBound)

        _range = numpy.arange(a[1].shape[0]) * Utils.GetTimeStep()

        lookupInt = 0 # I'm lazy
        dra1.set_xdata(_range)
        dra1.set_ydata(a[lookupInt])
        ax1.fill_between(_range, a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)
        ax1.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 1, 1), color="yellow")
        ax1.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 2, 1), color="blue")
        ax1.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 3, 1), color="green")
        ax1.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 4, 1), color="magenta")

        lookupInt = 1 # I'm lazy
        dra2.set_xdata(_range)
        dra2.set_ydata(a[lookupInt])
        ax2.fill_between(_range, a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)
        ax2.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 1, 1), color="yellow")
        ax2.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 2, 1), color="blue")
        ax2.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 3, 1), color="green")
        ax2.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 4, 1), color="magenta")

        lookupInt = 2 # I'm lazy
        dra3.set_xdata(_range)
        dra3.set_ydata(a[lookupInt])
        ax3.fill_between(_range, a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)
        ax3.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 1, 1), color="yellow")
        ax3.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 2, 1), color="blue")
        ax3.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 3, 1), color="green")
        ax3.fill_between(_range, 2**31, -2**31, where=ShouldHighlight(LineMaker, 4, 1), color="magenta")
        
    except Exception as e:
        print(e)
        pass
    finally:
        #simulator.Shutdown()
        fig.savefig("{}".format(name), facecolor=fig.get_facecolor())
        #input("Press Any Key")
        pass

def CreateAndSaveGraphSingl(a, name, label, lmin, lmax):
    try:
        maxY = 105
        maxTDPI = 96 * 2
        resolution = numpy.array((1920, 1080))
        TargetDPI = maxTDPI
        solvedSize = resolution / TargetDPI

        fig, ax1 = matplotlib.pyplot.subplots(1,1,sharex=True, dpi=TargetDPI, figsize=solvedSize)
        dra1, = ax1.plot([],[], color="red")

        targetColour = (0.05,0.05,0.05)

        ax1.set_facecolor(targetColour)
        fig.set_facecolor(targetColour)

        ax1.set_xlabel("Samples ({}s)".format(Utils.GetTimeStep()), color='white')

        if not label:
            ax1.set_ylabel("Temperature Error (°C)", color='white')
            dra1.set_label("Boiler Temperature Error (°C)")
        else:
            dra1.set_label(label)
            ax1.set_ylabel(label, color='white')

        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white') 
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')

        ax1.legend()

        if lmin is not None or lmax is not None:
            lowBound = lmin if lmin is not None else -100
            highBound = lmax if lmax is not None else 100
            ax1.set_ylim(lowBound, highBound)

        _range = numpy.arange(a[0].shape[0])

        lookupInt = 0 # I'm lazy
        dra1.set_xdata(numpy.arange(0, len(_range)))
        dra1.set_ydata(a[lookupInt])
        ax1.fill_between(numpy.arange(0, len(_range)), a[lookupInt] + a[lookupInt+1] * 2, a[lookupInt] - a[lookupInt+1] * 2)

    except Exception as e:
        print(e)
        pass
    finally:
        #simulator.Shutdown()
        fig.savefig("{}".format(name), facecolor=fig.get_facecolor())
        pass


def Compute1DMeanStd(x, windowSize = 50):
    pS = pandas.Series(x)
    means = numpy.array(pS.rolling(window=windowSize).mean())
    stds = numpy.array(pS.rolling(window=windowSize).std())

    # means = means[~numpy.isnan(means)]
    # stds = stds[~numpy.isnan(stds)]

    # print(means.shape)

    slicer = (windowSize // 1) + 1
    means = means[slicer:-slicer]
    stds = stds[slicer:-slicer]
    print(means.shape)

    return means, stds

def ComputeMinMax(x, y=None):
    try:
        if y is not None:
            lmax = numpy.nanmax(x + 2 * y)
            lmin = numpy.nanmin(x - 2 * y)
        else:
            lmax = numpy.nanmax(x)
            lmin = numpy.nanmin(x)
        ldir = lmax - lmin

        nmax = lmin + ldir * 1.25
        nmin = lmin - ldir * 0.25

        #lcen = (lmax * 0.5) + (lmin * 0.5)
        #nMax = 

        #print(lmin, lmax, ldir, nmin, nmax)
        return nmin, nmax
    except:
        return None, None

def Handler_RunError(rE, runType, name, rtReason=None, labels=None, localMin=None, localMax=None):
    meanList = []
    stdsList = []

    # Annoyingly, Series only works in 1D data
    rowsAreSignals = rE.transpose()
    for i in range(rE.shape[1]):
        mean, std = Compute1DMeanStd(rowsAreSignals[i])
        meanList.append(mean)
        stdsList.append(std)

    lmin = None
    lmax = None

    lmin, lmax = ComputeMinMax(meanList)

    if localMax is not None:
        lmax = localMax
    if localMin is not None:
        lmin = localMin

    print(lmin, lmax)

    concat = numpy.concatenate([numpy.array(meanList), numpy.array(stdsList)])

    CreateAndSaveGraphTri(concat, os.path.join("./Results/", "TriGraph_{}_{}.png".format(runType, name)), labels, lmin, lmax, rtReason)

def Handler_SingleValue(timeSeries, runType, name, label, localMin=None, localMax=None, useMeanStdRange=False):
    means, stds = Compute1DMeanStd(timeSeries)
    concat = numpy.vstack([means, stds])

    lmin = None
    lmax = None

    if not useMeanStdRange:
        lmin, lmax = ComputeMinMax(means)
    else:
        lmin, lmax = ComputeMinMax(means, stds)

    if localMax is not None:
        lmax = localMax
    if localMin is not None:
        lmin = localMin

    CreateAndSaveGraphSingl(concat, os.path.join("./Results/", "SinglGraph_{}_{}.png".format(runType, name)), label, lmin, lmax)


def Handler_ConcatTimeSeries(timeSeries1, timeSeries2, timeSeries3):
    meanList = []
    stdsList = []

    mean, std = Compute1DMeanStd(timeSeries1)
    meanList.append(mean)
    stdsList.append(std)

    mean, std = Compute1DMeanStd(timeSeries2)
    meanList.append(mean)
    stdsList.append(std)

    mean, std = Compute1DMeanStd(timeSeries3)
    meanList.append(mean)
    stdsList.append(std)

    concat = numpy.concatenate([numpy.array(meanList), numpy.array(stdsList)])

    CreateAndSaveGraphTri(concat, os.path.join("./Results/", runType + "_ConcatGraph.png"), labels=None)

comboBox = [
    "DMDc.dat",
    "Sindy.dat",
    "OKIDERA.dat",
    "BaseCase.dat",
    "FollowCase.dat",
    "OKIDERA_FilterData.dat",
    "Sindy_FilterData.dat",
    "DMDc_FilterData.dat",
    "BaseCase_FilterData.dat",
    "Sindy_FilterTrain.dat",
    "DMDc_FilterTrain.dat",
    "BaseCase_FilterTrain.dat",
    "Recurrent_FilterTrain.dat",
    "Recurrent_FilterData.dat",
    "Recurrent.dat"
]

# Create results directory
if not os.path.exists("./Results/"):
    os.makedirs("./Results/")

for runType in comboBox:
    print("Processing {}".format(runType))
    
    RunError = []
    RunInclusiveTime = []
    RunEvaluationTime = []
    RunRetrainTime = []

    with open("Error_" + runType, "rb+") as f:
        RunError = pickle.load(f)
    
    with open("IncTime_" + runType, "rb+") as f:
        RunInclusiveTime = pickle.load(f)

    with open("EvalTime_" + runType, "rb+") as f:
        RunEvaluationTime = pickle.load(f)

    with open("RetrainTime_" + runType, "rb+") as f:
        RunRetrainTime = pickle.load(f)

    with open("RetrainReason_" + runType, "rb+") as f:
        RtReason = pickle.load(f)

    print(RunError[0])
    # print(type(RunInclusiveTime))
    # print(type(RunEvaluationTime))
    # print(type(RunRetrainTime))

    # Handle Run Error
    arrayOfThings = []
    for i in RunError.keys():
        arrayOfThings.append(RunError[i])

    # Handle Inc Time
    arrayOfIncl = []
    for i in RunInclusiveTime.keys():
        arrayOfIncl.append(RunInclusiveTime[i])


    RunError = numpy.array(arrayOfThings).astype(numpy.float)
    RunInclusiveTime = numpy.array(arrayOfIncl).astype(numpy.float) * 1000
    RunEvaluationTime = numpy.array(RunEvaluationTime).astype(numpy.float) * 1000
    RunRetrainTime = numpy.array(RunRetrainTime).astype(numpy.float) * 1000
    RtReason = numpy.array(RtReason)

    # RunError = RunError.transpose()
    # RunError[-1] *= 100
    # RunError = RunError.transpose()

    print(RunError.shape)



    Handler_RunError(RunError, runType, "CumulativeSignedError", RtReason)#, lmin=-100, lmax=100)
    Handler_RunError(RunError.cumsum(axis=0), runType, "IntegralSignedError", RtReason)
    Handler_RunError(numpy.abs(RunError), runType, "CumulativeAbsError", RtReason)#, lmin=-100, lmax=100)
    Handler_RunError(numpy.abs(RunError).cumsum(axis=0), runType, "IntegralAbsError", RtReason)

    Handler_SingleValue(RunInclusiveTime, runType, "RunInc", "Inclusive Time (ms)", localMin=0, useMeanStdRange=True)
    Handler_SingleValue(RunEvaluationTime, runType, "Eval", "Evaluation Time (ms)", localMin=0, useMeanStdRange=True)
    Handler_SingleValue(RunRetrainTime, runType, "RTCost", "Retrain Cost (ms)", useMeanStdRange=True)

    Handler_SingleValue(RunInclusiveTime.cumsum(), runType, "CumulativeRunInc", "Cumulative Inclusive Time (ms)", localMin=0)
    Handler_SingleValue(RunEvaluationTime.cumsum(), runType, "CumulativeEval", "Cumulative Evaluation Time (ms)", localMin=0)
    Handler_SingleValue(RunRetrainTime.cumsum(), runType, "CumulativeRTCost", "Cumulative Retrain Cost (ms)", localMin=0)   

    # tt = numpy.array([RunInclusiveTime, RunEvaluationTime])
    # print(tt.shape)

    RunError = numpy.abs(numpy.diff(RunError, axis=0))
    ctcsv = pandas.DataFrame(RunError)
    ctcsv.to_csv("./Results/RE_{}.csv".format(runType), index=False)

    ctcsv = pandas.DataFrame(RunInclusiveTime)
    ctcsv.to_csv("./Results/RuntimeInc_{}.csv".format(runType), index=False)

    ctcsv = pandas.DataFrame(RunEvaluationTime)
    ctcsv.to_csv("./Results/RuntimeEval_{}.csv".format(runType), index=False)

    ctcsv = pandas.DataFrame(RunRetrainTime)
    ctcsv.to_csv("./Results/RetrainTime_{}.csv".format(runType), index=False)

exit()

fileToLoad = ""
## Try to load the dataset
CT_Stab = []
with open(fileToLoad, "rb+") as f:
    CT_Stab = pickle.load(f)

## Convert the dataset from a dict to a 3D array
arrayOfThings = []
for i in CT_Stab.keys():
    arrayOfThings.append(CT_Stab[i])

arrayOfThings = numpy.array(arrayOfThings).astype(numpy.float)

print(numpy.array(arrayOfThings).shape)

## Flip said array
#arrayOfThings = numpy.array(arrayOfThings).transpose()

## Switch Y and Z
#arrayOfThings = numpy.swapaxes(arrayOfThings, 0, 1)

## AVG and STDDEV on Z axis
print(arrayOfThings.shape)

a = arrayOfThings#numpy.sum(arrayOfThings, axis=2)
#print("REr")
print(a.shape)

# exit()

# means = numpy.mean(arrayOfThings, axis=2)
# dev   = numpy.std(arrayOfThings, axis=2)

## Create X time graphs of error+dev at depth Y
#a = numpy.concatenate([means, dev])
#print(a.shape)

#a = numpy.concatenate()
#a = numpy.concatenate([a[4:7], a[11:]])
#print(a.shape)

ctcsv = pandas.DataFrame(a)
ctcsv.plot()

input()
exit()

print(a[1].shape[0])

try:
    maxY = 105
    maxTDPI = 96 * 2
    resolution = numpy.array((1920, 1080))
    TargetDPI = maxTDPI
    solvedSize = resolution / TargetDPI

    fig, ((ax1, ax2, ax3)) = matplotlib.pyplot.subplots(3,1,sharex=True, dpi=TargetDPI, figsize=solvedSize)
    dra1, = ax1.plot([],[], color="red")
    dra2, = ax2.plot([],[], color="red")
    dra3, = ax3.plot([],[], color="red")

    targetColour = (0.05,0.05,0.05)

    ax1.set_facecolor(targetColour)
    ax2.set_facecolor(targetColour)
    ax3.set_facecolor(targetColour)
    fig.set_facecolor(targetColour)

    ax3.set_xlabel("Prediction Window Length", color='white')
    ax1.set_ylabel("Temperature Error (°C)", color='white')
    ax2.set_ylabel("Water Level Error (L)", color='white')
    ax3.set_ylabel("Boiler Power Error (100 W)", color='white')

    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white') 
    ax1.spines['right'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white') 
    ax2.spines['right'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')

    ax3.spines['bottom'].set_color('white')
    ax3.spines['top'].set_color('white') 
    ax3.spines['right'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='white')

    dra1.set_label("Boiler Temperature Error (°C)")
    dra2.set_label("Boiler Water Level Error (L)")
    dra3.set_label("Boiler Power Error (100 W)")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_ylim(-100, 100)
    ax2.set_ylim(-100, 100)
    ax3.set_ylim(-100, 100)

    _range = numpy.arange(a[1].shape[0])

    lookupInt = 0 # I'm lazy
    dra1.set_xdata(numpy.arange(0, len(_range)))
    dra1.set_ydata(a[lookupInt])
    ax1.fill_between(numpy.arange(0, len(_range)), a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)

    lookupInt = 1 # I'm lazy
    dra2.set_xdata(numpy.arange(0, len(_range)))
    dra2.set_ydata(a[lookupInt])
    ax2.fill_between(numpy.arange(0, len(_range)), a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)

    lookupInt = 2 # I'm lazy
    dra3.set_xdata(numpy.arange(0, len(_range)))
    dra3.set_ydata(a[lookupInt])
    ax3.fill_between(numpy.arange(0, len(_range)), a[lookupInt] + a[lookupInt+3] * 2, a[lookupInt] - a[lookupInt+3] * 2)
except Exception as e:
    print(e)
    pass
finally:
    #simulator.Shutdown()
   # fig.savefig("SPSH_{}.png".format(self.seed))
    input("Press Any Key")
    pass

ctcsv.to_csv("test.csv")

## self.boiler.GetBoilerWaterTemp(),
## self.boiler.waterVolCurrent,
## self.boiler.boilerPerformance * self.boiler.boilerPercent