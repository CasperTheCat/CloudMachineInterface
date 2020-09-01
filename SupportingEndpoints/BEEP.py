
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
import BEEP

# spTemp = 55
# spTarg = 75
# seed = 0
step = Utils.step
# dlp = 150

class ABEEP():
    def __init__(self, seed, spTemp, spTarg, dlp):
        self.simulator = CSimulator(1, 600000)
        self.boiler = self.simulator.SpawnObject(ABoiler, 20000, 30, 80, 30)
        self.boilerController = self.simulator.SpawnObject(ABoilerController, 5, 75, spTemp, seed)
        self.boilerController.Possess(self.boiler)
        self.history = []
        self.spTemp = spTemp
        self.spTarg = spTarg
        self.dlp = dlp
        self.seed = seed

    def LiveUpdate(self, maxY, solvedSize, TargetDPI, iTime, color, predictionFunction, loopLimit):
        dataP = []
        dataT = []
        dataS = []
        dataX = []
        dataClose = []
        dataFar = []
        dataDiff = []


        fig, ax, ax2, packedAxis1, packedAxis2 = MakeLiveMap(maxY, solvedSize, TargetDPI, iTime, color)

        dra, two, three, four, warn, warnfar, warndiff = packedAxis1
        dra2, two2, three2, four2, warn2 = packedAxis2


        ##### Actually do the loop!
        historyLength = 150 + Utils.seqLength

        for i in range(historyLength):
            self.simulator.SimulateNTicks(step * 100, 1/100)

            # Add
            hist = [
                self.boiler.waterInRatePerSecond,
                self.boiler.GetInflowWaterTemp(),
                self.boilerController.temperatureSetPoint,
                self.boiler.waterOutRatePerSecond,
                self.boiler.GetBoilerWaterTemp(),
                self.boiler.waterVolCurrent,
                self.boiler.boilerPerformance * self.boiler.boilerPercent
            ]
            self.history.append(numpy.array(hist))
        
        self.history = numpy.array(self.history)
        warningBar = []

        xhat = numpy.ones((self.history.shape[1]))
        localXhat = numpy.zeros((self.history.shape[1]))

        #!!!
        self.boilerController.SetDisableDisturbance()
        
        backOffset = 15
        arrLength = self.history.shape[0]

        localHistory = numpy.zeros((backOffset, self.history.shape[1]))

        try:
            for i in range(loopLimit):
                print("Overarching {}".format(i))
                for x in range(1):
                    self.simulator.SimulateNTicks(step * 100, 1/100)

                    hist = [
                        self.boiler.waterInRatePerSecond,
                        self.boiler.GetInflowWaterTemp(),
                        self.boilerController.temperatureSetPoint,
                        #spTemp,
                        self.boiler.waterOutRatePerSecond,
                        self.boiler.GetBoilerWaterTemp(),
                        self.boiler.waterVolCurrent,
                        self.boiler.boilerPerformance * self.boiler.boilerPercent
                    ]

                    if x == 0:
                        #print(history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset].transpose().shape)

                        # Predict next step
                        # Grab data *backOffset* from the end

                        prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat)
                        
                        # Save this for the next iteration
                        xhat = feedback

                        # Prep for the loop
                        localHistory[0] = prediction
                        localXhat = xhat

                        for sample in range(1, backOffset):

                            #print(localHistory[:sample].shape)
                            #print(history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample].shape)

                            #Concat
                            if sample < Utils.seqLength:
                                lh = numpy.concatenate(
                                    [
                                        self.history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample],
                                        localHistory[:sample]
                                    ])
                            else:
                                lh = localHistory[sample-Utils.seqLength:sample]
                            #print(lh.shape)

                            futurePrediction, futureFeedback = predictionFunction(lh, localXhat)
                            
                            # t, yo, xo = control.forced_response(
                            #     model,
                            #     numpy.arange(0, Utils.seqLength) * step,
                            #     U=lh.transpose(),
                            #     X0=localXhat
                            # )

                            localXhat = futureFeedback
                            localHistory[sample] = futurePrediction


                        #forecast = yo.transpose()[-1]
                        forecast = localHistory[0]
                        forecasterErrorFromSetpoint = forecast[4] - hist[4]

                        closePoint = forecasterErrorFromSetpoint
                        farPoint = localHistory[-1][4] - hist[4]

                        ldiff = hist[4] - self.history[-len(localHistory)][4]
                        rdiff = farPoint

                        print(i, ldiff, rdiff)

                        # If < EPS
                        #delta = numpy.sum(forecast - numpy.array(hist))

                        # if delta < boiler.GetBoilerWaterTemp() * 0.05:
                        #     delta = 0


                        #preds.append(forecast)
                        #delta = forecast - tStat
                        #delta = delta * Utils.StateOnlyWeight[4]
                        #warningBar.append(delta)                



                    # Add
                    self.history = self.history[1:]
                    self.history = numpy.concatenate((self.history, [numpy.array(hist)]))


                    # Update Everything
                    if i == self.dlp:
                        # Back
                        print("Setting {}".format(i))
                        self.boilerController.SetTarget(self.spTarg)

                    ax.collections.clear()
                    ax2.collections.clear()
                    #ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)


                    # Second Set
                    #print(dataP.shape)
                    #print(localHistory.shape)
                    predDataP = numpy.concatenate( [dataP, localHistory.transpose()[4]] )
                    predDataT = numpy.concatenate( [dataT, localHistory.transpose()[6]] )
                    predDataX = numpy.concatenate( [dataX, localHistory.transpose()[2]] )
                    predDataS = numpy.concatenate( [dataS, localHistory.transpose()[5]] )
                    # print("\n\n")
                    # print(history[-1])
                    # print(localHistory[-1])
                    # print(boiler.GetBoilerWaterTemp())
                    dra2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    dra2.set_ydata(predDataP)
                    two2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    two2.set_ydata(predDataT)
                    three2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    three2.set_ydata(predDataS)
                    four2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    four2.set_ydata(predDataX)
                    #warn2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
                    #warn2.set_ydata(data5)


                    dataP = numpy.concatenate([dataP, [self.boiler.GetBoilerWaterTemp()]])
                    dataT = numpy.concatenate([dataT, [self.boiler.boilerPercent * self.boiler.boilerPerformance * 0.1]])
                    dataX = numpy.concatenate([dataX, [self.boilerController.temperatureSetPoint]])
                    dataS = numpy.concatenate([dataS, [self.boiler.waterVolCurrent]])
                    dataClose = numpy.concatenate([dataClose, [ldiff]])
                    dataFar = numpy.concatenate([dataFar, [rdiff]])
                    dataDiff = numpy.concatenate([dataDiff, [abs(ldiff - rdiff)]])
                    

                    removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

                    #dra.set_ydata(dataP[removalCutter:])
                    at = 0#max((len(dataP) - 1) - iTime, 0)
                    dataP = dataP[at:]
                    dataT = dataT[at:]
                    dataS = dataS[at:]
                    dataX = dataX[at:]
                    dataClose = dataClose[at:]
                    dataFar = dataFar[at:]
                    dataDiff = dataDiff[at:]
                    dra.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    dra.set_ydata(dataP)
                    two.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    two.set_ydata(dataT)
                    three.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    three.set_ydata(dataS)
                    four.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    four.set_ydata(dataX)
                    warn.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    warn.set_ydata(dataClose)
                    warnfar.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    warnfar.set_ydata(dataFar)
                    warndiff.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    warndiff.set_ydata(dataDiff)



                    ax.set_xlim(left=-5, right=len(predDataP) * self.simulator.timeDilation +5)

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
            fig.savefig("SPSH_{}.png".format(self.seed))
            #input("Press Any Key")
            pass


def MakeLiveMap(maxY, solvedSize, TargetDPI, iTime, targetColour, labelOverrides = None, label2Overrides = None):
    fig, ((ax,ax2)) = matplotlib.pyplot.subplots(2,1,sharex=True, dpi=TargetDPI, figsize=solvedSize)
    #ax = matplotlib.pyplot.axes()
    #ax2 = ax.twin()
    dra, = ax.plot([],[], color="red")    
    two, = ax.plot([],[])
    three, = ax.plot([],[], color="green")
    four, = ax.plot([],[], linestyle="--")#, color="magenta")
    warn, = ax.plot([],[], linestyle="dotted")#, color="purple")
    warnfar, = ax.plot([],[], linestyle="dotted", color="purple")
    warndiff, = ax.plot([],[], linestyle="dotted", color="magenta")

    dra2, = ax2.plot([],[], color="red")
    two2, = ax2.plot([],[])
    three2, = ax2.plot([],[], color="green")
    four2, = ax2.plot([],[], linestyle="--")
    warn2, = ax2.plot([],[], linestyle="dotted")

    ax.yaxis.grid(True, color='white')
    ax2.yaxis.grid(True, color='white')

    if labelOverrides:
        if len(labelOverrides) > 0:
            dra.set_label(labelOverrides[0])
        if len(labelOverrides) > 1:
            two.set_label(labelOverrides[1])
        if len(labelOverrides) > 2:
            three.set_label(labelOverrides[2])
        if len(labelOverrides) > 3:
            four.set_label(labelOverrides[3])
        if len(labelOverrides) > 4:
            warn.set_label(labelOverrides[4])
        if len(labelOverrides) > 5:
            warnfar.set_label(labelOverrides[5])
        if len(labelOverrides) > 6:
            warndiff.set_label(labelOverrides[6])

        # If we have overrides for 2, use
        if label2Overrides:
            if len(label2Overrides) > 0:
                dra2.set_label(label2Overrides[0])
            if len(label2Overrides) > 1:
                two2.set_label(label2Overrides[1])
            if len(label2Overrides) > 2:
                three2.set_label(label2Overrides[2])
            if len(label2Overrides) > 3:
                four2.set_label(label2Overrides[3])
            if len(label2Overrides) > 4:
                warn2.set_label(label2Overrides[4])
        else:
            if len(labelOverrides) > 0:
                dra2.set_label(labelOverrides[0])
            if len(labelOverrides) > 1:
                two2.set_label(labelOverrides[1])
            if len(labelOverrides) > 2:
                three2.set_label(labelOverrides[2])
            if len(labelOverrides) > 3:
                four2.set_label(labelOverrides[3])
            if len(labelOverrides) > 4:
                warn2.set_label(labelOverrides[4])

    # Add legends
    ax.legend()
    ax2.legend()

    ax.set_facecolor(targetColour)
    ax2.set_facecolor(targetColour)
    fig.set_facecolor(targetColour)

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

    return fig, ax, ax2, (dra, two, three, four, warn, warnfar, warndiff), (dra2, two2, three2, four2, warn2)