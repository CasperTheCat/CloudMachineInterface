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

dilation = Utils.dilation
seqLength = Utils.seqLength
step = Utils.step
allShape = 3500
#seqLength = 60 * 24

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
# 60k seconds, measuring every minute
# disturbs, states, targetDisturbs, targetStates = Utils.MakeData(allShape, 55, dilation, seqLength, 10, disabledisturb and False, step=step, stack=True, seed=0)
# disturbs2, states2, targetDisturbs2, targetStates2 = Utils.MakeData(allShape, 45, dilation, seqLength, 10, disabledisturb, step=step, stack=True, seed=2)
# disturbs3, states3, targetDisturbs3, targetStates3 = Utils.MakeData(allShape, 35, dilation, seqLength, 4, disabledisturb, step=step, stack=True, seed=5)
# disturbs4, states4, targetDisturbs4, targetStates4 = Utils.MakeData(allShape, 85, dilation, seqLength, 18, disabledisturb, step=step, stack=True, seed=8)
# disturbs5, states5, targetDisturbs5, targetStates5 = Utils.MakeData(allShape, 95, dilation, seqLength, 7, disabledisturb, step=step, stack=True, seed=11)

# disturbs = numpy.concatenate((disturbs, disturbs2, disturbs3, disturbs4, disturbs5))
# states = numpy.concatenate((states, states2, states3, states4, states5))
# targetDisturbs = numpy.concatenate((targetDisturbs, targetDisturbs2, targetDisturbs3, targetDisturbs4, targetDisturbs5))
# targetStates = numpy.concatenate((targetStates, targetStates2, targetStates3, targetStates4, targetStates5))

# val_disturbs, val_states, val_targetDisturbs, val_targetStates = Utils.MakeData(60000, 75, dilation, seqLength, 2, True, step=step)


# #ins, outs, tests, rawins = MakeData(3000,55,dilation, seqLength, 35, disturb)
# #yins, youts, ytest, rawyins = MakeData(1000,45,dilation, seqLength, 21, disturb)


# forecastmodel = keras.Sequential(
#     [
#         #layers.Embedding(input_shape=(100, 3), output_dim=128),
#         layers.Input(shape=(seqLength, disturbs.shape[2])),
#         layers.LSTM(1024, return_sequences=True),
#         layers.Dropout(0.2),
#         layers.LSTM(1024, return_sequences=True),
#         #layers.GRU(64, return_sequences=True),
#         #layers.LSTM(128, return_sequences=True),
#         layers.LSTM(1024, return_sequences=False),
#         #layers.LSTM(64, return_sequences=True),
#         #layers.LSTM(64, return_sequences=True),
        
#         layers.Dense(256, activation='relu'),
#         layers.Dense(disturbs.shape[2])
#     ]
# )
# # model.add(layers.Embedding(input_dim=1000, output_dim=64))
# # model.add(layers.LSTM(128))
# # model.add(layers.LSTM(128))
# # model.add(layers.Dense(10))

# predmodel = keras.Sequential(
#     [
#         #layers.Embedding(input_shape=(100, 3), output_dim=128),
#         layers.Input(shape=(seqLength, states.shape[2] + disturbs.shape[2])),
#         layers.LSTM(1024, return_sequences=True),
#         layers.Dropout(0.1),
#         #layers.LSTM(1024, return_sequences=True),
#         #layers.GRU(64, return_sequences=True),
#         #layers.LSTM(128, return_sequences=True),
#         layers.LSTM(1024, return_sequences=False),
#         layers.Dropout(0.1),
#         #layers.LSTM(64, return_sequences=True),
#         #layers.LSTM(64, return_sequences=True),
        
#         layers.Dense(1024, activation='relu'),
#         layers.Dense(states.shape[2]+ disturbs.shape[2])
#     ]
# )

# forecastmodel.compile(
#     #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     loss="mse",
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     #metrics=["accuracy"],
# )

# predmodel.compile(
#     #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     loss="mse",
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     #metrics=["accuracy"],
# )


# predmodel.summary()
# forecastmodel.summary()

# epochlies = 3

# #predmodel.fit(ins, outs, validation_data=(yins, youts), batch_size=16, epochs=epochlies)

# print(disturbs[1][-1])
# print(targetDisturbs[0])
# print(len(disturbs), len(targetDisturbs))

# assert(disturbs[1][-1][0] == targetDisturbs[0][0])
# assert(disturbs[1][-1][1] == targetDisturbs[0][1])
# assert(disturbs[1][-1][2] == targetDisturbs[0][2])
# assert(len(disturbs) == len(targetDisturbs))

# inFeed = numpy.concatenate((disturbs, states), axis=2)
# inFeedStates= numpy.concatenate((targetDisturbs, targetStates), axis=1)
# inVal = numpy.concatenate((val_disturbs, val_states), axis=2)
# inValStates = numpy.concatenate((val_targetDisturbs, val_targetStates), axis=1)

# print(inVal.shape)
# print(inFeed.shape)
# print(states.shape[2] + disturbs.shape[2])

# print(inFeed[0][-1])
# print(targetStates[0])

#forecastmodel.fit(disturbs, targetDisturbs, validation_data=(val_disturbs, val_targetDisturbs), batch_size=16, epochs=epochlies)
#predmodel.fit(inFeed, inFeedStates, validation_data=(inVal, inValStates), batch_size=8, epochs=epochlies)











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
dra, = ax.plot([],[])
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
        boiler.boilerPerformance*boiler.boilerPercent
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
                boiler.boilerPerformance*boiler.boilerPercent
            ]

            if x == 0:
                lastStep = history[-1]
                boilerTemp = lastStep[4]
                waterIn = lastStep[0] * step
                waterTemp = lastStep[1]
                waterVol = lastStep[5]

                boilerTemp = (waterVol * boilerTemp + waterTemp * waterIn) / (waterIn + waterVol)
                boilerTemp = boilerTemp + ((lastStep[6] * step) / (4200 * waterVol))

                compVal = numpy.copy(lastStep)
                compVal[4] = boilerTemp


                # Prep for the loop
                localHistory[0] = compVal

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
                    
                    lastStep = lh[-1]
                    boilerTemp = lastStep[4]
                    waterIn = lastStep[0] * step
                    waterTemp = lastStep[1]
                    waterVol = lastStep[5]

                    boilerTemp = (waterVol * boilerTemp + waterTemp * waterIn) / (waterIn + waterVol)
                    boilerTemp = boilerTemp + ((lastStep[6] * step) / (4200 * waterVol))

                    compVal = numpy.copy(lastStep)
                    compVal[4] = boilerTemp

                    #localXhat = xo.transpose()[-1]
                    localHistory[sample] = compVal


                #forecast = yo.transpose()[-1]
                forecast = localHistory[0]
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
            print("\n\n")
            print(history[-1])
            print(localHistory[-1])
            print(boiler.GetBoilerWaterTemp())
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

