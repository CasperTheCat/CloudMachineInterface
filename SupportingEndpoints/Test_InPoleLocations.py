#!/usr/bin/env python3

import pickle

import control
import modred
import pysindy
import pydmd

import os
#import slycot
import scipy
from scipy import signal, fftpack
from scipy.fftpack import fftshift
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
# matplotlib.interactive(True)
# matplotlib.use("TkAgg") 
import numpy
import math
import sys
import Utils
import Graphing
import czt
from past.utils import old_div
import inspect
import gc

def FilterFrequenciesByPower(x, PowerCutoff=0.005, timeBase=0):
    #for signalIter in range(x.shape[0]):
    #freq, sig_f = czt.time2freq(numpy.arange(timeBase, timeBase + x.shape[0]) * Utils.step, x)
    #working = x.transpose()[signalIter].transpose()

    # print(x.shape)

    # curframe = inspect.currentframe()
    # calframe = inspect.getouterframes(curframe, 4)
    # print("Callstack")
    # for i, frame in enumerate(reversed(calframe)):
    #     tabby = "\t" + "  " * i
    #     print(tabby + str(frame[2]) + ":" + str(frame[3]))

    impack = fftpack.rfft(fftpack.rfft(x, axis=0), axis=1)
    unpack = fftpack.irfft(fftpack.irfft(impack, axis=1), axis=0)
    # print(unpack[0], x[0])

    #impack = fftpack.rfft(working, axis=0)
    #unpack = fftpack.irfft(impack, axis=0)

    assert(numpy.allclose(x, unpack))

    # matplotlib.pyplot.figure(1)
    # #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
    # matplotlib.pyplot.plot(numpy.arange(0, impack.shape[0]),  numpy.angle(impack),   'ro--', label='CZT')
    # matplotlib.pyplot.xlabel("Frequency (kHz)")
    # matplotlib.pyplot.ylabel("Signal phase")
    # ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Frequency-domain")

    # matplotlib.pyplot.figure(2)
    # #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
    # matplotlib.pyplot.plot(numpy.arange(0, impack.shape[0]) / 1e3,  numpy.abs(impack),   'ro--', label='CZT')
    # matplotlib.pyplot.xlabel("Frequency (kHz)")
    # matplotlib.pyplot.ylabel("Signal mag")
    # ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Frequency-domain")


    powers = impack * numpy.conj(impack) / numpy.product(x.shape)

    autoCutoff = numpy.max(powers) * PowerCutoff

    indices = powers > autoCutoff
    impack = indices * impack

    # matplotlib.pyplot.figure(1)
    # #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
    # matplotlib.pyplot.plot(numpy.arange(0, impack.shape[0]),  powers,   'ro--', label='CZT')
    # matplotlib.pyplot.xlabel("Frequency (kHz)")
    # matplotlib.pyplot.ylabel("Signal phase")
    # ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Frequency-domain")

    # matplotlib.pyplot.figure(2)
    # #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
    # matplotlib.pyplot.plot(numpy.arange(0, impack.shape[0]) / 1e3,  numpy.abs(impack),   'ro--', label='CZT')
    # matplotlib.pyplot.xlabel("Frequency (kHz)")
    # matplotlib.pyplot.ylabel("Signal mag")
    # ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Frequency-domain")

    #input()

    #out = fftpack.irfft(impack, axis=0)
    out = fftpack.irfft(fftpack.irfft(impack, axis=1), axis=0)

    return out


maxY = 105
maxTDPI = 120
resolution = numpy.array((1920, 1080))
TargetDPI = maxTDPI

solvedSize = resolution / TargetDPI

iTime = 60
color = (0.05,0.05,0.05)



simulator = CSimulator(1, 600000)
#simulator = CSimulator(1, 200000)

spTemp = 55
spTarg = 75
seed = 0
step = Utils.GetTimeStep()
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

cost = []
costToRT = []
stepsSinceLastTrain = 0
rtTimes = 0
tholdRTTimes = 0

# For DMD
cacheA = None
cacheB = None

##### ##### ########## ##### #####
## BASE
##

def BaseEvalFunction(history, feedback, i):
    return history[-1][4:], []

def BaseRetrainFunction(history):
    global rtTimes

    rtTimes += 1

def BaseDetectorFunction(history, feedback, timeBase):
    return False

##### ##### ########## ##### #####
## Control Base
##

def EvalFunction(history, feedback, timeBase):
    global model
    global cost

    asU = history.transpose()[:4]

    evalBeginTime = time.perf_counter()

    _, yo, xo = control.forced_response(
        model,
        #(numpy.arange(time, time + history.shape[0])) * step,
        U=asU,
        X0=feedback
    )

    evalEndTime = time.perf_counter()
    cost.append(evalEndTime - evalBeginTime)

    # Set the input to the output bar temp
    # output = history[-1].copy()
    # print(history.transpose()[2])
    # print(output)
    # output[4] = yo.transpose()[-1][4]

    # return output, xo.transpose()[0]
    return yo.transpose()[-1], xo.transpose()[1]

def RetrainFunction(history):
    global costToRT
    global rtTimes
    global model

    rtTimes += 1
    
    evalBeginTime = time.perf_counter()

    print("Retraining")
    ht = Utils.TailState(history, Utils.offset, rowsAreSamples=True)
    ht = ht.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()
    newModel = None
    try:
        newModel, score = Utils.GetBestOKID(l1, l2)
        model = newModel
    except:
        # Don't update the model!
        pass


    # ## Bode
    # transFunc = control.ss2tf(predmodel)
    # scipy.signal.bode(transFunc)
    # #a = control.bode(transFunc)
    # #signal.TransferFunction
    # print(a)

    timePassed = time.perf_counter() - evalBeginTime
    costToRT.append(timePassed)

def DetectorFunction(history, feedback, timeBase):
    global model
    # yShift = fftshift(history[-1]) # shift of the step function
    # Fourier = scipy.fft(yShift) # Fourier transform of y implementing the FFT
    # Fourier = fftshift(Fourier) # inverse shift of the Fourier Transform
    # print(Fourier)
    # matplotlib.pyplot.plot(Fourier) # plot of the Fourier transform

    # input()
    # exit()

    # ### Laplace Transform
    # lU = numpy.random.laplace()

    # ### Do the maths!
    # uVector = history[-1]
    # uVt = timeBase + history.shape[0]

    return False

    # Transform the history U-vector
    PreU = history.transpose()[:4].transpose()
    # Transform with B
    A,B,C,D = control.ssdata(model)

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



        freq, sig_f = czt.time2freq(numpy.arange(timeBase, timeBase + workingData.shape[0]) * Utils.GetTimeStep(), workingData)

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
            matplotlib.pyplot.title(Utils.outnames[signalIter])

        
        input()    


    exit()




    return False

##### ##### ########## ##### #####
## Machine Learning
##

def ML_EvalFunction(history, feedback, timeBase):
    global cost

    # Timing Harness
    evalBeginTime = time.perf_counter()
    
    ytest = numpy.expand_dims(history[:Utils.seqLength], 0)
    forecast = predmodel.predict(ytest)
    forebar = tf.squeeze(forecast, 0).numpy()

    evalEndTime = time.perf_counter()
    cost.append(evalEndTime - evalBeginTime)

    return forebar, []

def ML_RetrainFunction(history):
    global costToRT
    global rtTimes
    global predmodel

    rtTimes += 1
    
    evalBeginTime = time.perf_counter()

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

    # Delete and force a GC
    del predmodel
    predmodel = Utils.GenerateModel(disturbs, states)
    gc.collect() 

    predmodel.fit(inFeed, inFeedStates, validation_data=(inVal, inValStates), batch_size=8, epochs=3)
    #predmodel.save("model.tensorflow")

    timePassed = time.perf_counter() - evalBeginTime
    costToRT.append(timePassed)

def ML_DetectorFunction(history, feedback, timeBase):
    return False


##### ##### ########## ##### #####
## DMDc
##

dmdModel = None

with open("Pickle.dmd", "rb+") as f:
    dmdModel = pickle.load(f)

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

def DMDc_RetrainFunction(history):
    global dmdModel
    global rtTimes
    global costToRT
    global cacheA
    global cacheB

    rtTimes += 1
    evalBeginTime = time.perf_counter()

    print("Retraining")
    ht = history.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()

    #nl1 = l1.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()
    #nl2 = l2.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()

    dmdModel, score = Utils.GetBestDMD(l1, l2)

    eigs = numpy.power(dmdModel.eigs, old_div(dmdModel.dmd_time['dt'], dmdModel.original_time['dt']))
    cacheA = dmdModel.modes.dot(numpy.diag(eigs)).dot(numpy.linalg.pinv(dmdModel.modes))
    cacheB = dmdModel.B

    timePassed = time.perf_counter() - evalBeginTime
    costToRT.append(timePassed)


def DMDc_DetectorFunction(history, feedback, timeBase):
    global dmdModel
    global rtTimes

    if(rtTimes == 0):
        return False

    # yShift = fftshift(history[-1]) # shift of the step function
    # Fourier = scipy.fft(yShift) # Fourier transform of y implementing the FFT
    # Fourier = fftshift(Fourier) # inverse shift of the Fourier Transform
    # print(Fourier)
    # matplotlib.pyplot.plot(Fourier) # plot of the Fourier transform

    # input()
    # exit()

    # ### Laplace Transform
    # lU = numpy.random.laplace()

    # ### Do the maths!
    # uVector = history[-1]
    # uVt = timeBase + history.shape[0]

    # Transform the history U-vector
    PreU = history.transpose()[:4].transpose()

    # Modify
    # PreU = PreU.transpose()
    # PreU = PreU + numpy.sin(numpy.arange(0, history.shape[0]) * 100000) * 10 + numpy.cos(2 + numpy.arange(0, history.shape[0]) * 5) * 10
    # PreU = PreU.transpose()

    # Transform with B
    B = dmdModel.B

    ModifiedOutput = numpy.zeros((PreU.shape[0], 3))

    print(PreU.shape)
    print("NB: Max Freq: {}".format(Utils.GetDetectableFrequency()))

    for sample in range(PreU.shape[0]):
        workingData = PreU[sample]
        ModU = B.dot(workingData)
        ModifiedOutput[sample] = ModU

    for signalIter in range(3):

        #print("Stepping ", signalIter)
        #print(ModifiedOutput.shape)
        workingData = ModifiedOutput.transpose()[signalIter]#[signalIter]
        #print(workingData.shape)
        #input()



        freq, sig_f = czt.time2freq(numpy.arange(timeBase, timeBase + workingData.shape[0]) * Utils.GetTimeStep(), workingData)

        f0 = matplotlib.pyplot.figure(8, dpi=120, figsize=(36,18))
        a0 = f0.gca()
        #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
        a0.plot(freq,  numpy.angle(sig_f),   'ro--', label='CZT')
        a0.set_xlabel("Frequency (Hz)")
        a0.set_ylabel("Signal phase")
        ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
        #matplotlib.pyplot.legend()
        a0.set_title(Utils.outnames[signalIter])
        f0.savefig("{}_WWPhase.png".format(signalIter))
        f0.clf()

        f1 = matplotlib.pyplot.figure(9, dpi=120, figsize=(36,18))
        a1 = f1.gca()
        #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
        a1.plot(freq,  numpy.abs(sig_f),   'ro--', label='CZT')
        a1.set_xlabel("Frequency (Hz)")
        a1.set_ylabel("Signal Mag. (dB)")
        a1.set_title(Utils.outnames[signalIter])
        ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
        #matplotlib.pyplot.legend()
        f1.savefig("{}_WWMag.png".format(signalIter))
        f1.clf()

        # Decompile to real and imag compo
        # zU = e^(z * dt)
        # ln(zU) = z * dt
        # z = ln(zU) / dt
        
        poles = 0

        fx = matplotlib.pyplot.figure(signalIter + 10, dpi=120, figsize=(36,18))
        ax = fx.gca()

        for cmptePle in zip(freq, sig_f):
            # So
            # We would like to compute the real and imaginary components of the number
            # sig_F
            #tScore = Utils.GetFitness(cmptePle[1])

            rScale = cmptePle[1].real
            iScale = cmptePle[1].imag

            #print(rScale, iScale)

            #input()

            
            #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
            ax.plot([rScale],  [iScale],   'ro--', label='CZT')
            ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
            #matplotlib.pyplot.legend()
            ax.set_title(Utils.outnames[signalIter])

        fx.savefig("{}_WW.png".format(signalIter))
        fx.clf()

    print("Wait")        
    input()    







##### ##### ########## ##### #####
## MrDMD
##

mrdmdModel = None

with open("Pickle.mrdmd", "rb+") as f:
    mrdmdModel = pickle.load(f)

def MrDMD_EvalFunction(history, feedback, i):
    global mrdmdModel

    ht = history.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()

    #nl1 = l1.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()

    eigs = numpy.power(mrdmdModel.eigs, old_div(mrdmdModel.dmd_time['dt'], mrdmdModel.original_time['dt']))
    A = mrdmdModel.modes.dot(numpy.diag(eigs)).dot(numpy.linalg.pinv(mrdmdModel.modes))
    X = l2.transpose()[-1]
    out = A.dot(X)

    # B = dmdModel.B
    # U = l1.transpose()[-1]
    

    # out = A.dot(X) + B.dot(U)

    #out = dmdModel.reconstructed_data(nl1).transpose()[-1]

    return out, []

def MrDMD_RetrainFunction(history):
    global mrdmdModel
    global rtTimes
    global costToRT

    rtTimes += 1
    evalBeginTime = time.perf_counter()

    print("Retraining")
    ht = history.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()

    #nl1 = l1.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()
    #nl2 = l2.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()

    mrdmdModel, score = Utils.GetBestMrDMD(l1, l2)

    timePassed = time.perf_counter() - evalBeginTime
    costToRT.append(timePassed)

def MrDMD_DetectorFunction(history, feedback, timeBase):
    return False


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

def Sindy_RetrainFunction(history):
    global sindyModel
    global rtTimes
    global costToRT

    rtTimes += 1
    evalBeginTime = time.perf_counter()

    print("Retraining")
    ht = history.transpose()
    l1 = ht[:4]#.transpose()
    l2 = ht[4:]#.transpose()

    #nl1 = l1.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()
    #nl2 = l2.transpose()[:dmdModel.dynamics.shape[1] - 1].transpose()

    try:
        sindyModel, score = Utils.GetBestSindy(l1, l2)
    except:
        rtTimes -= 1
        pass

    timePassed = time.perf_counter() - evalBeginTime
    costToRT.append(timePassed)

def Sindy_DetectorFunction(history, feedback, timeBase):
    return False


def ThresholdFunction(signedError, absoluteError):
    global stepsSinceLastTrain
    global tholdRTTimes

    shouldRetrainOnFixed = stepsSinceLastTrain * Utils.GetTimeStep() >= 500
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


def ZeroAllVars():
    global cost
    global costToRT
    global stepsSinceLastTrain
    global rtTimes
    global tholdRTTimes

    cost = []
    costToRT = []
    stepsSinceLastTrain = 0
    rtTimes = 0
    tholdRTTimes = 0


def ModulateSP_LF(base, i):
    if i > 250:
        tgFreqHz = 1.1

        # Each sample adds 1 radians per ST
        # A Hz is 2Pi rads
        SampleTime = Utils.GetTimeStep()
        radsPerSecond = 1 / SampleTime
        hertz = radsPerSecond / (math.pi * 2)

        freqMul = tgFreqHz / hertz

        #print(freqMul, hertz)

        return base + 100 * math.sin(i * freqMul)
    else:
        return base



comboBox = [
    # (Sindy_EvalFunction, Sindy_RetrainFunction, Sindy_DetectorFunction, None, None, "Sindy.dat"),
    (DMDc_EvalFunction, DMDc_RetrainFunction, DMDc_DetectorFunction, None, None, ModulateSP_LF, "DMDc.dat"),
    # (EvalFunction, RetrainFunction, DetectorFunction, None, None, "OKIDERA.dat"),
    # #(MrDMD_EvalFunction, MrDMD_RetrainFunction, MrDMD_DetectorFunction, "MrDMD.dat"),
    # (BaseEvalFunction, BaseRetrainFunction, BaseDetectorFunction, None, None, "BaseCase.dat"),


    # (EvalFunction, RetrainFunction, DetectorFunction, FilterFrequenciesByPower, None, "OKIDERA_FilterData.dat"),
    # (Sindy_EvalFunction, Sindy_RetrainFunction, Sindy_DetectorFunction, FilterFrequenciesByPower, None, "Sindy_FilterData.dat"),
    # #(MrDMD_EvalFunction, MrDMD_RetrainFunction, MrDMD_DetectorFunction, "MrDMD.dat"),
    # (DMDc_EvalFunction, DMDc_RetrainFunction, DMDc_DetectorFunction, FilterFrequenciesByPower, None, "DMDc_FilterData.dat"),
    # (BaseEvalFunction, BaseRetrainFunction, BaseDetectorFunction, FilterFrequenciesByPower, None, "BaseCase_FilterData.dat"),
    

    # (EvalFunction, RetrainFunction, DetectorFunction, None, FilterFrequenciesByPower, "OKIDERA_FilterTrain.dat"), ## EXCLUDED DUE TO CRASH
    # (Sindy_EvalFunction, Sindy_RetrainFunction, Sindy_DetectorFunction, None, FilterFrequenciesByPower, "Sindy_FilterTrain.dat"),
    # #(MrDMD_EvalFunction, MrDMD_RetrainFunction, MrDMD_DetectorFunction, "MrDMD.dat"),
    # (DMDc_EvalFunction, DMDc_RetrainFunction, DMDc_DetectorFunction, None, FilterFrequenciesByPower, "DMDc_FilterTrain.dat"),
    # (BaseEvalFunction, BaseRetrainFunction, BaseDetectorFunction, None, FilterFrequenciesByPower, "BaseCase_FilterTrain.dat"),

    # (ML_EvalFunction, ML_RetrainFunction, ML_DetectorFunction, None, None, "Recurrent.dat"),
    # (ML_EvalFunction, ML_RetrainFunction, ML_DetectorFunction, None, FilterFrequenciesByPower, "Recurrent_FilterTrain.dat"),
    # (ML_EvalFunction, ML_RetrainFunction, ML_DetectorFunction, FilterFrequenciesByPower, None, "Recurrent_FilterData.dat"),
   
]

for evf, rtf, dtf, flt, rtfflt, modu, name in comboBox:
    ZeroAllVars()

    graphing = Graphing.AGraphHolder(seed, spTemp, spTarg, dlp)
    _, results, timeResults = graphing.TestRetraining(evf, rtf, ThresholdFunction, 16384, detectorFunction=dtf, filterFunction=flt, retrainFilter=rtfflt, modulator=modu)
    #graphing.TestRetrainLive(maxY, solvedSize, TargetDPI, iTime, color, evf, rtf, ThresholdFunction, 300, ["Temperature (C)", "Heater Power (kW)", "Water Level (L)", "Target Temperature (C)", "Cosine Sim.", "Error"], filterFunction=flt, retrainFilter=rtfflt)



input()
exit()

# cost = 0
# costml = 0
# stepsSinceLastTrain = 0
# rtTimes = 0
# tholdRTTimes = 0

# graphing = Graphing.AGraphHolder(seed, spTemp, spTarg, dlp)
# _, mlresults = graphing.TestRetraining(ML_EvalFunction, ML_RetrainFunction, ThresholdFunction, 300)

# with open("backtrackRetrainWithML.dat", "wb+") as f:
#     pickle.dump(mlresults, f)

# print("Cost {} ({}s)".format(cost, cost/rtTimes))
# print("Retrains {}. (Fixed RTs {})".format(rtTimes, tholdRTTimes))
