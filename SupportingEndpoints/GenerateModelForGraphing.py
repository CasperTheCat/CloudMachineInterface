#!/usr/bin/env python3

import pickle
import control
import modred
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
# matplotlib.interactive(True)
# matplotlib.use("TkAgg") 
import numpy
import cmath
import math
import Utils
import scipy
import slycot


dilation = Utils.dilation
seqLength = Utils.seqLength
step = Utils.step

disabledisturb = False
allShape = 35000

# 60k seconds, measuring every minute
disturbs, states, targetDisturbs, targetStates = Utils.MakeData(allShape, 55, dilation, seqLength, 3, disabledisturb, step=step, stack=False, seed=0, tailed=True)
disturbs2, states2, targetDisturbs2, targetStates2 = Utils.MakeData(allShape, 45, dilation, seqLength, 10, disabledisturb, step=step, stack=False, seed=2, tailed=True)
disturbs3, states3, targetDisturbs3, targetStates3 = Utils.MakeData(allShape, 35, dilation, seqLength, 4, disabledisturb, step=step, stack=False, seed=5, tailed=True)
disturbs4, states4, targetDisturbs4, targetStates4 = Utils.MakeData(allShape, 85, dilation, seqLength, 18, disabledisturb, step=step, stack=False, seed=8, tailed=True)
disturbs5, states5, targetDisturbs5, targetStates5 = Utils.MakeData(allShape, 95, dilation, seqLength, 7, disabledisturb, step=step, stack=False, seed=11, tailed=True)
disturbs6, states6, targetDisturbs6, targetStates6 = Utils.MakeData(allShape, 15, dilation, seqLength, 16, disabledisturb, step=step, stack=False, seed=16, tailed=True)
disturbs7, states7, targetDisturbs7, targetStates7 = Utils.MakeData(allShape, 25, dilation, seqLength, 6, disabledisturb, step=step, stack=False, seed=19, tailed=True)

print(disturbs.shape)

disturbs = numpy.concatenate((disturbs, disturbs2, disturbs3, disturbs4, disturbs5, disturbs6, disturbs7))
states = numpy.concatenate((states, states2, states3, states4, states5, states6, states7))

# targetDisturbs = numpy.concatenate((targetDisturbs, targetDisturbs2, targetDisturbs3))
# targetStates = numpy.concatenate((targetStates, targetStates2, targetStates3))

#disturbs, states = Utils.ShuffleTogether(disturbs, states)


#val_disturbs, val_states, val_targetDisturbs, val_targetStates = Utils.MakeData(10, 75, dilation, seqLength, 2, False, step=step, stack=False)

print(disturbs.shape)

# Moving concate. The matrixs ranks out of OKID are incorrect
inFeed = disturbs#numpy.concatenate((disturbs, states), axis=1)
#inVal = val_disturbs#numpy.concatenate((val_disturbs, val_states), axis=1)
inStates = states
#inValState = val_states

print(inFeed.shape)
#inVal = numpy.concatenate((val_disturbs, val_states), axis=2)

offset = Utils.offset

#print(ins.shape)

l1t = inFeed
l2t = inStates

# 
# l1t = inFeed[:-1]
# l2t = inFeed[1:]
v2t = disturbs

l1 = l1t.transpose()
#l2 = l2t.transpose()
l2 = l2t.transpose()
#v2 = v2t.transpose()

#l1 = states.transpose()
#l1 = Utils.TailState(l1, offset)
#l2 = Utils.TailState(l2, offset)
#v2 = Utils.TailState(v2, offset)

# Retranspose
l1t = l1.transpose()
l2t = l2.transpose()
#v2t = v2.transpose()

#print(l1.shape, l2.shape)


markovs = disturbs.shape[0] // 120
minmcs = 0
markovs = 25#66

bestIndex = 0
#bestScore = 1 # Unstable above 1
bestScore = 3.5# Allow at least 1 failed pole

def DistanceImagToPole(x):
    rScale = x.real
    iScale = x.imag

    distSq = rScale * rScale + iScale * iScale

    return numpy.sqrt(distSq)

def DistanceToZero(x):
    #dSq = numpy.power(x, 2)
    accumDist = 0

    for i in x:
        ds = DistanceImagToPole(i)
        accumDist += 1 if ds > 1 else ds # ds

    return accumDist

# Generates a low number when close to zero
def GetFitness(x):
    return DistanceToZero(x)

def CreateOKIDERA(l1, l2, i, step, dilation):
    #l1 = numpy.flip(l1, 1) 
    #l2 = numpy.flip(l2, 1) 
    kalman = modred.OKID(l1, l2, i)
    era = modred.ERA()
    a,b,c = era.compute_model(kalman, 20000)
    #b *= 1/step
    #a,b,c = modred.era.compute_ERA_model(kalman, 1500)

    print("Mats")
    print(a.shape)
    print(b.shape)
    print(c.shape)
    # print(c == numpy.identity(4))
    print()

    #c = numpy.identity(3) * 0.5 + c * 0.5

    # Some *real* asserts
    assert(a.shape != (0,0))
    assert(b.shape != (0,0))
    assert(c.shape != (0,0))
    #@c = numpy.identity(3)

    ### Test Asserts
    # assert(a.shape == (3,3))
    # assert(b.shape == (3,4))
    # assert(c.shape == (3,3))

    newScore = GetFitness(numpy.linalg.eigvals(a))
    print(newScore)

    

    asb = control.ss(a,b,c, numpy.zeros((c.shape[0], b.shape[1])), step)
    #asb = control.balred(asb, i)
    poles = control.pole(asb)
    score = GetFitness(poles)

    


    for p in poles:
        rScale = p.real
        iScale = p.imag

        print(rScale, iScale)

        plfig = matplotlib.pyplot.figure(i)
        plplt = plfig.gca()
        #matplotlib.pyplot.plot(f_fft / 1e3, np.angle(sig_fft), 'k',    label='FFT')
        plplt.plot([rScale],  [iScale],   'ro--', label='Poles')
        ##matplotlib.pyplot.xlim([f_fft.min()/1e3, f_fft.max()/1e3])
        plplt.set_xlabel("Real")
        plplt.set_ylabel("Imaginary")
        #matplotlib.pyplot.legend()
        plplt.set_title("Poles")
    
    plfig.savefig("OKID_AllPoles.{}.png".format(i))

    return asb, score



print(l1.shape)


for i in range(minmcs, markovs):
    print("Attempting to get {} markovs ({}/{})".format(i,i-minmcs,markovs-minmcs))
    try:
        asb, score = Utils.CreateOKIDERA(l1,l2,i,step,dilation)

        print("{} scored {}".format(i, score))
        
        if score < bestScore:
            bestIndex = i
            bestScore = score

            # Async start the process

       
    except Exception as e:
        print("Fail on {}. {}".format(i,e))


print("Using {} markovs".format(bestIndex))

asb, score = CreateOKIDERA(l1, l2 ,bestIndex, step, dilation)

##### ##### ########## ##### #####
## Sanity Check Rank
##




##### ##### ########## ##### #####
## Pickle
##

with open("Pickle.era", "wb+") as f:
    pickle.dump(asb, f)

Utils.CreateBodeAndPolePlots(asb, "OKIDERA", True)
    

# A,B,C,D = control.ssdata(asb)

# scipylti = asb.returnScipySignalLTI()

# for outputs in enumerate(scipylti):
#     for inputs in enumerate(outputs):
#         print(inputs)
#         w, mag, phase = scipy.signal.bode(inputs)

#         print(w.shape)
#         print(mag.shape)
#         print(phase.shape)


# # Suppress ragged array 
# ragged = numpy.array([A,B,C,D], dtype=object)
# sys = scipy.signal.StateSpace(ragged, dt=Utils.step)




exit()
##### ##### ########## ##### #####
## Build History
##

# Build Forward
# There's no benefit to building backwards since each step is discrete

# How far back?
backstep = seqLength * 6#len(l1t) - 1
print(len(l1t))

pairwiseErrors = []
preds = []
xhat = [0]

for i in range(offset, offset + backstep):
    #itu = numpy.expand_dims(inFeed[i], 0)

    #print(l1t[i:(i) + seqLength].shape)

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, len(l1t[i:(i) + seqLength])),
    #     U=l1t[i:(i) + seqLength].transpose()
    # )

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, len(l1[0])),
    #     U=l1[i:i+seqLength]
    # )

    #print(l1t[i:(i) + seqLength])
    #itu = numpy.expand_dims(v2t[i:(i) + seqLength], 0)
    #print(l1t[i:(i) + seqLength])

    #if i == offset + 2:
    #    print(1/0)

    # t, yo, xo = control.forced_response(
    #     asb,
    #     numpy.arange(0, len(l1t[i:(i) + seqLength])),
    #     U=Utils.TailState(v2t[i:(i) + seqLength], 10)
    # )
    
    t, yo, xhat = control.forced_response(
        asb,
        numpy.arange(0, len(l1t[i:(i) + seqLength])) * step,
        U=v2t[i:(i) + seqLength].transpose(),
        X0=xhat[-1]
    )

    xhat = numpy.array(xhat).transpose()

    # print(v2t[i:(i) + seqLength].shape)
    # print(yo.shape)

    # for j in range(0,100):
    #     print(yo[j])
    #     #print(yo[0][j], yo[1][j], yo[2][j])

    # #print(yo)
    indexer = -1
    ls = v2t[i:(i) + seqLength]
    tStat = ls[indexer]
    forecast = yo.transpose()[indexer]
    #forecast = yo.transpose()[seqLength - 1]
    #print(i, forecast, tStat)

    preds.append(forecast[4])

    delta = forecast - tStat
    delta = delta * Utils.StateOnlyWeight[:-1]

    print(i, numpy.sum(delta))

    pairwiseErrors.append(delta)


pairwiseErrorsAcc = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip, useAbs=False)
pairwiseErrors = Utils.MakeAccError(pairwiseErrors, flip=Utils.bFlip)



dataP = v2t[seqLength + offset:].transpose()[4] * Utils.StateOnlyWeight[4]
#dataT = inFeed.transpose()[5]
dataT = numpy.array(preds) * Utils.StateOnlyWeight[4]
dataS = pairwiseErrorsAcc.transpose()
dataX = pairwiseErrors.transpose()
print(dataP.flatten().squeeze().shape)
print(dataT.flatten().squeeze().shape)
#print(len(dataP))
dataT = list(dataT.flatten())
dataP = list(dataP.flatten())
dataS = list(dataS.flatten())
dataX = list(dataX.flatten())

fig = Utils.MakeScreen(dataP, dataT, dataS, dataX)

fig.savefig("DTCE_{}.png".format(Utils.TimeNow()))   

#ax = pd.plot()
fig.canvas.draw()
fig.canvas.flush_events()

Utils.MakeCSV(pairwiseErrors, "DTCE_{}.csv".format(Utils.TimeNow()))
Utils.MakeCSV(pairwiseErrorsAcc, "DTCE_{}_SIGNED.csv".format(Utils.TimeNow()))

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
