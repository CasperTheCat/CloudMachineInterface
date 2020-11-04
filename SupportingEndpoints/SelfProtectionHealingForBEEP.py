#!/usr/bin/env python3

import pickle
import control
import modred
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

boiler = simulator.SpawnObject(ABoiler, 20000, 30, 80, 30)
boilerController = simulator.SpawnObject(ABoilerController, 5, 75, spTemp, seed) # Heating to 95
#boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
boilerController.Possess(boiler)

boiler.SetInflowWaterTemp(24)
boiler.SetInflowRate(0.0)
boiler.SetOutflowRate(0.0001)

# boiler.SetBoilerPower(100)



maxY = 105
maxTDPI = 240
resolution = numpy.array((1920, 1080))
TargetDPI = maxTDPI

solvedSize = resolution / TargetDPI

#fig = matplotlib.pyplot.figure(dpi=TargetDPI, figsize=solvedSize)#figsize=(lScalar*scaleWidth, min((lScalar * scaleWidth*scaleWidth / 16), max(16, (lScalar * 18 / 16)))))
fig, ((ax,ax2)) = matplotlib.pyplot.subplots(2,1,sharex=True, dpi=TargetDPI, figsize=solvedSize)
#ax = matplotlib.pyplot.axes()
#ax2 = ax.twin()
dra, = ax.plot([],[], color="red")
dra.set_label("Boiler Temperature")
two, = ax.plot([],[])
two.set_label("1")
three, = ax.plot([],[])
three.set_label("2")
four, = ax.plot([],[], linestyle="--")
four.set_label("3")
warn, = ax.plot([],[], linestyle="dotted")
warn.set_label("4")

dra2, = ax2.plot([],[])
dra2.set_label("Boiler Temperature")
two2, = ax2.plot([],[])
two2.set_label("1")
three2, = ax2.plot([],[])
three2.set_label("2")
four2, = ax2.plot([],[], linestyle="--")
four2.set_label("3")
warn2, = ax2.plot([],[], linestyle="dotted")
warn2.set_label("4")

iTime = 60

color = (0.05,0.05,0.05)
# ax.plot([-5,iTime+5], [60,60])
# ax.plot([-5,iTime+5], [30,30])
#ax.axhline(spTemp, linestyle='--', color='red')
ax.yaxis.grid(True, color='white')
ax2.yaxis.grid(True, color='white')


ax.set_facecolor(color)
ax2.set_facecolor(color)
fig.set_facecolor(color)

ax2.set_xlabel("Window Time (Seconds)", color='white')
ax.set_ylabel("Temperature (°C) / Power (%) / Water Level (L)", color='white')
ax.set_ylabel("True Values", color='white')
ax2.set_ylabel("Future Trend Values", color='white')
ax.set_ylim(top=maxY, bottom=-1)
ax2.set_ylim(top=maxY, bottom=-1)
#ax.set_ylim(top=maxY, bottom=-100)
#ax.set_xlim(left=-5, right=iTime+5)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')

ax2.spines['bottom'].set_color('white')
ax2.spines['top'].set_color('white') 
ax2.spines['right'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')

dataP = []#[0]# * iTime 
dataT = []
dataS = []
dataX = []
data5 = []


#### SPSH
model = None

with open("Pickle.era", "rb+") as f:
    model = pickle.load(f)


#input("Press Any Key")

history = []# [[],[],[],[],[],[],[],[]]

historyLength = 150 + Utils.seqLength

for i in range(historyLength):
    simulator.SimulateNTicks(step * 100, 1/100)

    # Add
    hist = [
        boiler.waterInRatePerSecond,
        boiler.GetInflowWaterTemp(),
        boilerController.temperatureSetPoint,
        boiler.waterOutRatePerSecond,
        boiler.GetBoilerWaterTemp(),
        boiler.waterVolCurrent,
        boiler.boilerPerformance * boiler.boilerPercent
    ]
    history.append(numpy.array(hist))

    # history[0].append(boiler.waterInRatePerSecond)

    # # Temperature
    # history[1].append(boiler.GetInflowWaterTemp())

    # # Setpoint
    # history[2].append(boilerController.temperatureSetPoint)

    # # Out Flow Rate
    # history[3].append(boiler.waterOutRatePerSecond)

    # # Out Flow Temperature
    # history[4].append(boiler.GetBoilerWaterTemp())



    # # State Volume
    # history[0].append(boiler.waterVolCurrent)

    # # State Power
    # history[1].append(boiler.boilerPerformance)
    
    # history[2].append(boiler.boilerPercent)

    print("step {}".format(i))

history = numpy.array(history)
warningBar = []
xhat = numpy.zeros((history.shape[1]))
localXhat = numpy.zeros((history.shape[1]))

boilerController.SetDisableDisturbance()
backOffset = 60
arrLength = history.shape[0]

localHistory = numpy.zeros((backOffset, history.shape[1]))

try:
    for i in range(1300):
        print("Overarching {}".format(i))
        for x in range(1):
            simulator.SimulateNTicks(step * 100, 1/100)

            hist = [
                boiler.waterInRatePerSecond,
                boiler.GetInflowWaterTemp(),
                boilerController.temperatureSetPoint,
                #spTemp,
                boiler.waterOutRatePerSecond,
                boiler.GetBoilerWaterTemp(),
                boiler.waterVolCurrent,
                boiler.boilerPerformance * boiler.boilerPercent
            ]

            if x == 0:
                #print(history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset].transpose().shape)

                # Predict next step
                # Grab data *backOffset* from the end

                t, yo, xo = control.forced_response(
                        model,
                        numpy.arange(0, Utils.seqLength) * step,
                        U=history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset].transpose(),
                        X0=xhat
                    )
                
                # Save this for the next iteration
                xhat = xo.transpose()[-1]

                # Prep for the loop
                localHistory[0] = yo.transpose()[-1]
                localXhat = xhat

                for sample in range(1, backOffset):

                    #print(localHistory[:sample].shape)
                    #print(history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample].shape)

                    #Concat
                    if sample < Utils.seqLength:
                        lh = numpy.concatenate(
                            [
                                history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample],
                                localHistory[:sample]
                            ])
                    else:
                        lh = localHistory[sample-Utils.seqLength:sample]
                    #print(lh.shape)
                    
                    t, yo, xo = control.forced_response(
                        model,
                        numpy.arange(0, Utils.seqLength) * step,
                        U=lh.transpose(),
                        X0=localXhat
                    )

                    localXhat = xo.transpose()[-1]
                    localHistory[sample] = yo.transpose()[-1]


                #forecast = yo.transpose()[-1]
                forecast = localHistory[-1]
                forecasterErrorFromSetpoint = hist[2] - forecast[4]

                delta = forecast[4]

                print(i, forecast[2], forecasterErrorFromSetpoint)

                # If < EPS
                #delta = numpy.sum(forecast - numpy.array(hist))

                # if delta < boiler.GetBoilerWaterTemp() * 0.05:
                #     delta = 0


                #preds.append(forecast)
                #delta = forecast - tStat
                #delta = delta * Utils.StateOnlyWeight[4]
                #warningBar.append(delta)                



            # Add
            history = history[1:]
            history = numpy.concatenate((history, [numpy.array(hist)]))


            # Update Everything
            if i == dlp:
                # Back
                print("Setting {}".format(i))
                boilerController.SetTarget(spTarg)

            ax.collections.clear()
            ax2.collections.clear()
            #ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)


            # Second Set
            #print(dataP.shape)
            #print(localHistory.shape)
            predDataP = numpy.concatenate( [dataP, localHistory.transpose()[4]] )
            # print("\n\n")
            # print(history[-1])
            # print(localHistory[-1])
            # print(boiler.GetBoilerWaterTemp())
            dra2.set_xdata(numpy.arange(0, len(predDataP)) * simulator.timeDilation)
            dra2.set_ydata(predDataP)
            # two2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            # two2.set_ydata(dataP)
            # three2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            # three2.set_ydata(dataS)
            # four2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            # four2.set_ydata(dataX)
            # warn2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            # warn2.set_ydata(data5)


            dataP = numpy.concatenate([dataP, [boiler.GetBoilerWaterTemp()]])
            dataT = numpy.concatenate([dataT, [boiler.boilerPercent * 100]])
            dataX = numpy.concatenate([dataX, [boilerController.temperatureSetPoint]])
            dataS = numpy.concatenate([dataS, [boiler.waterVolCurrent]])
            data5 = numpy.concatenate([data5, [delta]])

            removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

            #dra.set_ydata(dataP[removalCutter:])
            at = 0#max((len(dataP) - 1) - iTime, 0)
            dataP = dataP[at:]
            dataT = dataT[at:]
            dataS = dataS[at:]
            dataX = dataX[at:]
            data5 = data5[at:]
            dra.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            dra.set_ydata(dataT)
            two.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            two.set_ydata(dataP)
            three.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            three.set_ydata(dataS)
            four.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            four.set_ydata(dataX)
            warn.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
            warn.set_ydata(data5)



            ax.set_xlim(left=-5, right=len(predDataP) * simulator.timeDilation +5)

            #ax = pd.plot()
            fig.canvas.draw()
            fig.canvas.flush_events()


            #mod = math.cos(i * 0.1) * 10
            #mod = math.sin(i * 0.01) ** 640 * 30
            #boilerController.SetTarget(spTarg - math.floor(mod))

            #print("Update Setpoint {} -> {}°C".format(spTemp - mod, spTemp - math.floor(mod)))
            #print("Update Boiler Perf {}w".format(boiler.boilerPerformance))
            #print("Update Boiler Hist {}s".format(boiler.CurrentControlTime))

            #simulator.SetTimeDilation(20 * (i + 1))
            #boiler.SetBoilerPower((i + 1) * 10)
            # print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
            #print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
            #print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
            # print("[TIME {:.02f}s] Power Perc: {:.02f}%".format((i + 1) * simulator.timeDilation, boiler.boilerPercent * 100))
            # print("[TIME {:.02f}s] PID: {:.02f}i".format((i + 1) * simulator.timeDilation, boilerController.PID.iVal))
            # print("[TIME {:.02f}s] PIDdbg: {}".format((i + 1) * simulator.timeDilation, boilerController.PID.dbgLastReturn))

except Exception as e:
    print(e)
    pass
finally:
    #simulator.Shutdown()
    fig.savefig("SPSH_{}.png".format(seed))
    #input("Press Any Key")
    pass

