
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
import numpy
import math
import sys
import Utils
import Graphing

# spTemp = 55
# spTarg = 75
# seed = 0
step = Utils.step
nVars = 3
# dlp = 150

def normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

class AGraphHolder():
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

        # Last Frame
        self.lastFrameTime = 0

        # Used for AVG eval cycle
        self.lastUpdateTimes = numpy.array([0])

        # Used for deleting stale info from the above metric
        self.lastUpdateTimesRt = numpy.array([0])

        self.perfWindow = 30

    def ProcessAvgFramerate(self):
        realWindow = min(30, len(self.lastUpdateTimes))

        comp = numpy.convolve(self.lastUpdateTimes, numpy.ones((realWindow,))/realWindow, mode='valid')

        nmc = numpy.mean(comp)

        return ((1 / nmc) if nmc != 0 else 0)

    def GenerateAndSetHistory(self, predictionFunction, historyLength):
        self.history = []
        xhat = numpy.zeros((nVars))

        ## Create History
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

            # # Skip first loop, it's not valid for all predictions
            # if i > 0:
            #     _, xhat = predictionFunction( numpy.array(self.history)[-4:], xhat)
            #     print(xhat)

        self.history = numpy.array(self.history)

        tempHist = Utils.TailState(self.history.transpose(), Utils.offset).transpose()
        _, xhat = predictionFunction(tempHist, xhat, 0)

        return xhat

    def CoreLoop(self, loopLimit):
        return None

    def CoreSample(self, predictionFunction, localHistory, localXhat, sample, backOffset, arrLength, i, flt=None):
        # Set Next Value for Disturbs (Forecast)
        nextVal = self.history[-1][:4]

        #Concat
        # If Array is still got data
        if sample < backOffset:
            lh = self.history[arrLength - (Utils.seqLength + backOffset) + sample:(arrLength - backOffset) + sample]
            nextVal = self.history[arrLength - backOffset + sample][:4]
        elif sample < Utils.seqLength:
            lh = numpy.concatenate(
                [
                    self.history[arrLength - (Utils.seqLength + backOffset) + sample:(arrLength - backOffset) + sample][:Utils.seqLength-sample],
                    localHistory[:sample]
                ])
        else:
            lh = localHistory[sample-Utils.seqLength:sample]
        
        if(flt):
            return predictionFunction(flt(lh), localXhat, i), nextVal
        else:
            return predictionFunction(lh, localXhat, i), nextVal

    def TestRetrainLive(self, maxY, solvedSize, TargetDPI, iTime, color, predictionFunction, retrainCallback, thresholdFunction, loopLimit, labelOverrides=None, label2Overrides=None, backOffset = 15, futureOffset = 15, historyLength = 1000 + Utils.seqLength, detectorFunction=None, filterFunction=None, retrainFilter=None):
        dataP = []
        dataT = []
        dataS = []
        dataX = []
        dataClose = []
        dataFar = []
        dataDiff = []

        errorTracking = 0.0
        absErrorTracking = 0.0

        fig, ax, ax2, packedAxis1, packedAxis2 = MakeLiveMap(maxY, solvedSize, TargetDPI, iTime, color, labelOverrides=labelOverrides, label2Overrides=label2Overrides)

        dra, two, three, four, warn, warnfar, warndiff = packedAxis1
        dra2, two2, three2, four2, warn2 = packedAxis2


        ##### Actually do the loop!
        # xhat = numpy.zeros((nVars))

        print("Generating Pre Run History")

        xhat = self.GenerateAndSetHistory(predictionFunction, historyLength)

        # tempHist = Utils.TailState(self.history.transpose(), Utils.offset).transpose()
        # _, xhat = predictionFunction(tempHist, xhat)


        warningBar = []

        print(xhat)
        
        # xhat[0] = hist[4]
        # xhat[1] = hist[5]
        # xhat[2] = hist[6]
        localXhat = numpy.zeros((nVars))
        

        #!!!
        #self.boilerController.SetDisableDisturbance()
        
        arrLength = self.history.shape[0]

        localHistory = numpy.zeros((backOffset + futureOffset, self.history.shape[1]))
        stepErrorOracle = numpy.zeros((backOffset))# + futureOffset, self.history.shape[1]))

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

                        # BUG? This is trying to run *backOffset* behind but that's not always what we want
                        # When it's behind, it lags the live changes and fails to predict
                        #prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat)

                        if (filterFunction):
                            prediction, feedback = predictionFunction(filterFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset]), xhat, i)
                        else:
                            prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat, i)
                        
                        # Save this for the next iteration
                        xhat = feedback

                        # Prep for the loop
                        localHistory[0] = numpy.concatenate((self.history[arrLength - backOffset][:4], prediction))
                        localXhat = xhat

                        stepError = self.history[arrLength - backOffset] * Utils.ErrorWeights - localHistory[0] * Utils.ErrorWeights
                        stepErrorOracle[0] = numpy.abs(numpy.sum(stepError))

                        for sample in range(1, backOffset + futureOffset):
                            #print(localHistory[:sample].shape)
                            #print(history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample].shape)

                            (futurePrediction, futureFeedback), nextVal = self.CoreSample(predictionFunction, localHistory, localXhat, sample, backOffset, arrLength, i+sample, flt=filterFunction)

                            localXhat = futureFeedback

                            # Change Below when we can forecast
                            localHistory[sample] = numpy.concatenate((nextVal, futurePrediction))

                            if sample < backOffset:
                                stepError = self.history[arrLength - backOffset + sample] * Utils.ErrorWeights - localHistory[sample] * Utils.ErrorWeights
                                stepErrorOracle[sample] = numpy.abs(numpy.sum(stepError))

                        # #forecast = yo.transpose()[-1]
                        # forecast = localHistory[0]
                        # forecasterErrorFromSetpoint = forecast[4] - hist[4]

                        # closePoint = forecasterErrorFromSetpoint
                        # farPoint = localHistory[-1][4] - hist[4]

                        # ldiff = hist[4] - self.history[-len(localHistory)][4]
                        # rdiff = farPoint

                        # print(i, ldiff, rdiff)


                        # If < EPS
                        #delta = numpy.sum(forecast - numpy.array(hist))

                        # if delta < boiler.GetBoilerWaterTemp() * 0.05:
                        #     delta = 0


                        #preds.append(forecast)
                        #delta = forecast - tStat
                        #delta = delta * Utils.StateOnlyWeight[4]
                        #warningBar.append(delta)          
                        
                        ## Detector
                        if (detectorFunction):
                            detectorFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat, i)      

                        # ERROR
                        errorTracking += stepError
                        absErrorTracking += abs(stepError)

                        # Do we need to retrain?
                        if thresholdFunction(errorTracking, absErrorTracking):
                            if(retrainFilter):
                                retrainCallback(retrainFilter(self.history))
                            else:
                                retrainCallback(self.history)
                            errorTracking = 0.0
                            absErrorTracking = 0.0

                    # Add
                    self.history = self.history[1:]
                    self.history = numpy.concatenate((self.history, [numpy.array(hist)]))


                    # Update Everything
                    if i == self.dlp:
                        # Back
                        print("Setting {}".format(i))
                        self.boilerController.SetTarget(self.spTarg)

                    if i > self.dlp:
                        mod = math.sin(i * 0.05) * 10 #** 640 * 30
                        self.boilerController.SetTarget(self.spTarg - math.floor(mod))

                    ax.collections.clear()
                    ax2.collections.clear()
                    #ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)


                    # Second Set
                    #print(dataP.shape)
                    #print(localHistory.shape)
                    predDataP = numpy.concatenate( [dataP[:-backOffset], localHistory.transpose()[4]] )
                    predDataT = numpy.concatenate( [dataT[:-backOffset], localHistory.transpose()[6] * 0.001] )
                    predDataX = numpy.concatenate( [dataX[:-backOffset], localHistory.transpose()[2]] )
                    predDataS = numpy.concatenate( [dataS[:-backOffset], localHistory.transpose()[5]] )
                    offsetVal = 0 if len(dataClose) == 0 else dataClose[-1]
                    print(offsetVal)
                    predClose = numpy.concatenate( [dataClose[:-backOffset], (numpy.cumsum(stepErrorOracle)) * 0.01 + offsetVal] )
                    # print("\n\n")
                    # print(history[-1])
                    # print(localHistory[-1])
                    # print(boiler.GetBoilerWaterTemp())
                    #warn2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
                    #warn2.set_ydata(data5)


                    dataP = numpy.concatenate([dataP, [self.boiler.GetBoilerWaterTemp()]])
                    dataT = numpy.concatenate([dataT, [self.boiler.boilerPercent * self.boiler.boilerPerformance * 0.001]])
                    dataX = numpy.concatenate([dataX, [self.boilerController.temperatureSetPoint]])
                    dataS = numpy.concatenate([dataS, [self.boiler.waterVolCurrent]])
                    dataClose = numpy.concatenate([dataClose, [numpy.sum(absErrorTracking) * 0.01]])
                    dataFar = numpy.concatenate([dataFar, [0]])
                    #dataDiff = numpy.concatenate([dataDiff, [abs(ldiff - rdiff)]])
                    

                    #removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

                    #dra.set_ydata(dataP[removalCutter:])
                    at = 0#max((len(dataP) - 1) - iTime, 0)
                    dataP = dataP[at:]
                    dataT = dataT[at:]
                    dataS = dataS[at:]
                    dataX = dataX[at:]
                    dataClose = dataClose[at:]
                    dataFar = dataFar[at:]
                    dataDiff = dataDiff[at:]

                    #predat = max((len(predDataP) - 1) - iTime, 0)
                    predDataP = predDataP[at:]
                    predDataS = predDataS[at:]
                    predDataT = predDataT[at:]
                    predDataX = predDataX[at:]
                    predClose = predClose[at:]


                    dra2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    dra2.set_ydata(predDataP)
                    two2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    two2.set_ydata(predDataT)
                    three2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    three2.set_ydata(predDataS)
                    four2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    four2.set_ydata(predDataX)
                    warn2.set_xdata(numpy.arange(0, len(predClose)) * self.simulator.timeDilation)
                    warn2.set_ydata(predClose)

                    
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
                    # warndiff.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    # warndiff.set_ydata(dataDiff)



                    ax.set_xlim(left=-5, right=len(predDataP) * self.simulator.timeDilation +5)
                    #ax2.fill_between(x, len(dataP) * self.simulator.timeDilation +5, len(predDataP) * self.simulator.timeDilation +5, facecolor='green', alpha=0.5)
                    #ax2.fill_between(numpy.arange(0, len(dataP)) * self.simulator.timeDilation, 0, 1, where=x > (len(dataP) * self.simulator.timeDilation), facecolor='green', alpha=0.5)
                    ax2.fill_between(numpy.arange(len(dataP) - 1, len(predDataP)) * self.simulator.timeDilation, 0, maxY, facecolor='purple', alpha=0.25)
                    ax2.fill_between(numpy.arange(len(dataP) - (1 + backOffset), len(dataP)) * self.simulator.timeDilation, 0, maxY, facecolor='pink', alpha=0.25)

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
            raise(e)
            pass
        finally:
            #simulator.Shutdown()
            #fig.savefig("SPSH_{}.png".format(self.seed))
            #input("Press Any Key")
            pass

    def TestRetraining(self, predictionFunction, retrainCallback, thresholdFunction, loopLimit, backOffset = 1, futureOffset = 0, historyLength = 1000 + Utils.seqLength, detectorFunction=None, filterFunction=None, retrainFilter=None):
        offsetResults = {}
        retrainError={}
        errorTracking = 0.0
        absErrorTracking = 0.0

        for i in range(backOffset):
            offsetResults[i] = []

        xhat = self.GenerateAndSetHistory(predictionFunction, historyLength)

        localXhat = numpy.zeros((nVars))
        arrLength = self.history.shape[0]

        localHistory = numpy.zeros((backOffset + futureOffset, self.history.shape[1]))

        for i in range(loopLimit):
            # Begin Perf Timer
            beginPerfTime = time.perf_counter()

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

            if (filterFunction):
                prediction, feedback = predictionFunction(filterFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], timeBase=i), xhat, i)
            else:
                prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat, i)
            #prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat, i)
                        
            # Save this for the next iteration
            xhat = feedback

            # Prep for the loop
            localHistory[0] = numpy.concatenate((self.history[arrLength - backOffset][:4], prediction))
            localXhat = xhat



            stepError = self.history[arrLength - backOffset] * Utils.ErrorWeights - localHistory[0] * Utils.ErrorWeights
            #stepError = numpy.abs(stepError)
            offsetResults[0].append(stepError)

            ## Oracle
            for sample in range(1, backOffset):
                (futurePrediction, futureFeedback), nextVal = self.CoreSample(predictionFunction, filterFunction(localHistory, timeBase=i+sample), localXhat, sample, backOffset, arrLength, i+sample, flt=filterFunction)

                localXhat = futureFeedback

                # Change Below when we can forecast
                localHistory[sample] = numpy.concatenate((nextVal, futurePrediction))

                stepError = self.history[arrLength - backOffset + sample] * Utils.ErrorWeights - localHistory[sample] * Utils.ErrorWeights
                #stepError = numpy.abs(stepError)
                offsetResults[sample].append(stepError)

            #print(i, cosineSimilarity, currentTimeError, vecCurrent[4:])

            ## Detector
            if (detectorFunction):
                detectorFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat, i)

            # ERROR
            errorTracking += stepError
            absErrorTracking += abs(stepError)

            retrainError[i] = [numpy.sum(errorTracking), numpy.sum(absErrorTracking), numpy.sum(stepError)]

            # Do we need to retrain?
            if thresholdFunction(errorTracking, absErrorTracking):
                print("\tError {} at {}. Retrain.".format(absErrorTracking, i))
                if(retrainFilter):
                    retrainCallback(retrainFilter(self.history))
                else:
                    retrainCallback(self.history)
                #retrainCallback(self.history)
                errorTracking = 0.0
                absErrorTracking = 0.0

            # Add
            self.history = self.history[1:]
            self.history = numpy.concatenate((self.history, [numpy.array(hist)]))

            # Use last frame time because we want to include *this* computation in the avg
            totalFrameTime = self.lastFrameTime

            # Time for the first element
            baseTime = self.lastUpdateTimesRt[-1]
            self.lastUpdateTimes = numpy.concatenate([self.lastUpdateTimes, [totalFrameTime]])
            self.lastUpdateTimesRt = numpy.concatenate([self.lastUpdateTimesRt, [totalFrameTime] + baseTime])

            # Which elements are stale?
            removalCutter = numpy.argmax(self.lastUpdateTimesRt > (baseTime - self.perfWindow))

            # Update frame time list
            self.lastUpdateTimes = self.lastUpdateTimes[removalCutter:]

            # Rebase the Realtime list (Prevent accumulating large numbers)
            holdover = self.lastUpdateTimesRt[removalCutter]
            self.lastUpdateTimesRt = self.lastUpdateTimesRt[removalCutter:] - holdover



            if i % 10 == 0:
                framerate = self.ProcessAvgFramerate()
                if (framerate != 0):
                    print("{} - ETA: {:.2f}s. Rate: {:.2f} eops. (Covering {:.2f}s)".format(
                        i,
                        (loopLimit - i) / framerate,
                        framerate * backOffset,
                        framerate * Utils.dilation * Utils.step
                    ))
                else:
                    print("{} - ETA: INF. Rate: {:.2f} eops. (Covering {:.2f}s)".format(
                        i,
                        framerate * backOffset,
                        framerate * Utils.dilation * Utils.step
                    ))

            # Perf Timers
            endPerfTime = time.perf_counter()
            self.lastFrameTime = endPerfTime - beginPerfTime
    
        return offsetResults, retrainError

    def TestOffsetWidth(self, predictionFunction, loopLimit, backOffset = 150, futureOffset = 0, historyLength = 450 + Utils.seqLength):
        offsetResults = {}

        for i in range(backOffset):
            offsetResults[i] = []

        xhat = self.GenerateAndSetHistory(predictionFunction, historyLength)

        localXhat = numpy.zeros((nVars))
        arrLength = self.history.shape[0]

        localHistory = numpy.zeros((backOffset + futureOffset, self.history.shape[1]))

        for i in range(loopLimit):
            # Begin Perf Timer
            beginPerfTime = time.perf_counter()

            for x in range(150):
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
                    prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat, i*150+x)
                                
                    # Save this for the next iteration
                    xhat = feedback

                    # Prep for the loop
                    localHistory[0] = numpy.concatenate((self.history[arrLength - backOffset][:4], prediction))
                    localXhat = xhat



                    stepError = self.history[arrLength - backOffset] * Utils.ErrorWeights - localHistory[0] * Utils.ErrorWeights
                    #stepError = numpy.abs(stepError)
                    offsetResults[0].append(stepError)


                    ## Oracle
                    for sample in range(1, backOffset):
                        (futurePrediction, futureFeedback), nextVal = self.CoreSample(predictionFunction, localHistory, localXhat, sample, backOffset, arrLength, i*150+x)

                        localXhat = futureFeedback

                        # Change Below when we can forecast
                        localHistory[sample] = numpy.concatenate((nextVal, futurePrediction))

                        stepError = self.history[arrLength - backOffset + sample] * Utils.ErrorWeights - localHistory[sample] * Utils.ErrorWeights
                        #stepError = numpy.abs(stepError)
                        offsetResults[sample].append(stepError)


                    ## Error Handling
                    vecStart = self.history[arrLength - (Utils.seqLength + backOffset)] * Utils.ErrorWeights
                    vecCurrent = self.history[-1] * Utils.ErrorWeights
                    vecPrediction = localHistory[backOffset - 1] * Utils.ErrorWeights

                    ldiff = vecCurrent - vecStart
                    rdiff = vecPrediction - vecCurrent

                    ldiffn = normalize(ldiff)
                    rdiffn = normalize(rdiff)

                    cosineSimilarity = (1 + numpy.dot(ldiffn, rdiffn)) * 50
                    currentTimeError = math.sqrt(numpy.dot(rdiff, rdiff))

                #print(i, cosineSimilarity, currentTimeError, vecCurrent[4:])

                # Add
                self.history = self.history[1:]
                self.history = numpy.concatenate((self.history, [numpy.array(hist)]))

            # Use last frame time because we want to include *this* computation in the avg
            totalFrameTime = self.lastFrameTime

            # Time for the first element
            baseTime = self.lastUpdateTimesRt[-1]
            self.lastUpdateTimes = numpy.concatenate([self.lastUpdateTimes, [totalFrameTime]])
            self.lastUpdateTimesRt = numpy.concatenate([self.lastUpdateTimesRt, [totalFrameTime] + baseTime])

            # Which elements are stale?
            removalCutter = numpy.argmax(self.lastUpdateTimesRt > (baseTime - self.perfWindow))

            # Update frame time list
            self.lastUpdateTimes = self.lastUpdateTimes[removalCutter:]

            # Rebase the Realtime list (Prevent accumulating large numbers)
            holdover = self.lastUpdateTimesRt[removalCutter]
            self.lastUpdateTimesRt = self.lastUpdateTimesRt[removalCutter:] - holdover



            if i % 10 == 0:
                print("{} - ETA: {:.2f}s. Rate: {:.2f} eops. (Covering {:.2f}s)".format(i, (loopLimit - i) / self.ProcessAvgFramerate(), self.ProcessAvgFramerate() * backOffset, self.ProcessAvgFramerate() * Utils.dilation * Utils.step))

            # Perf Timers
            endPerfTime = time.perf_counter()
            self.lastFrameTime = endPerfTime - beginPerfTime
    
        return offsetResults

    def LiveUpdate(self, maxY, solvedSize, TargetDPI, iTime, color, predictionFunction, loopLimit, labelOverrides=None, label2Overrides=None, backOffset = 15, futureOffset = 15, historyLength = 450 + Utils.seqLength):
        dataP = []
        dataT = []
        dataS = []
        dataX = []
        dataClose = []
        dataFar = []
        dataDiff = []


        fig, ax, ax2, packedAxis1, packedAxis2 = MakeLiveMap(maxY, solvedSize, TargetDPI, iTime, color, labelOverrides=labelOverrides, label2Overrides=label2Overrides)

        dra, two, three, four, warn, warnfar, warndiff = packedAxis1
        dra2, two2, three2, four2, warn2 = packedAxis2


        ##### Actually do the loop!
        # xhat = numpy.zeros((nVars))

        print("Generating Pre Run History")

        xhat = self.GenerateAndSetHistory(predictionFunction, historyLength)

        # tempHist = Utils.TailState(self.history.transpose(), Utils.offset).transpose()
        # _, xhat = predictionFunction(tempHist, xhat)


        warningBar = []

        print(xhat)
        
        # xhat[0] = hist[4]
        # xhat[1] = hist[5]
        # xhat[2] = hist[6]
        localXhat = numpy.zeros((nVars))

        #!!!
        #self.boilerController.SetDisableDisturbance()
        
        arrLength = self.history.shape[0]

        localHistory = numpy.zeros((backOffset + futureOffset, self.history.shape[1]))

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

                        # BUG? This is trying to run *backOffset* behind but that's not always what we want
                        # When it's behind, it lags the live changes and fails to predict
                        #prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat)

                        prediction, feedback = predictionFunction(self.history[arrLength - (Utils.seqLength + backOffset):arrLength - backOffset], xhat, i)
                        
                        # Save this for the next iteration
                        xhat = feedback

                        # Prep for the loop
                        localHistory[0] = numpy.concatenate((self.history[arrLength - backOffset][:4], prediction))
                        localXhat = xhat

                        for sample in range(1, backOffset + futureOffset):

                            #print(localHistory[:sample].shape)
                            #print(history[arrLength - (Utils.seqLength + backOffset) + sample:arrLength - backOffset + sample][:Utils.seqLength-sample].shape)

                            # Set Next Value for Disturbs (Forecast)
                            nextVal = self.history[-1][:4]

                            #Concat
                            # If Array is still got data
                            if sample < backOffset:
                                lh = self.history[arrLength - (Utils.seqLength + backOffset) + sample:(arrLength - backOffset) + sample]
                                nextVal = self.history[arrLength - backOffset + sample][:4]
                            elif sample < Utils.seqLength:
                                lh = numpy.concatenate(
                                    [
                                        self.history[arrLength - (Utils.seqLength + backOffset) + sample:(arrLength - backOffset) + sample][:Utils.seqLength-sample],
                                        localHistory[:sample]
                                    ])
                            else:
                                lh = localHistory[sample-Utils.seqLength:sample]
                            

                            futurePrediction, futureFeedback = predictionFunction(lh, localXhat, i + sample)
                            
                            # if(sample == 1):
                            #     print(lh[-1])
                            #     print(self.history[arrLength - backOffset + 1])
                            #assert(sample != backOffset - 1 or lh[-1][1] == self.history[-1][1])

                            

                            localXhat = futureFeedback
                            # Change Below when we can forecast
                            localHistory[sample] = numpy.concatenate((nextVal, futurePrediction))


                        # #forecast = yo.transpose()[-1]
                        # forecast = localHistory[0]
                        # forecasterErrorFromSetpoint = forecast[4] - hist[4]

                        # closePoint = forecasterErrorFromSetpoint
                        # farPoint = localHistory[-1][4] - hist[4]

                        # ldiff = hist[4] - self.history[-len(localHistory)][4]
                        # rdiff = farPoint

                        # print(i, ldiff, rdiff)

                        vecStart = self.history[arrLength - (Utils.seqLength + backOffset)] * Utils.ErrorWeights
                        vecCurrent = self.history[-1] * Utils.ErrorWeights
                        vecPrediction = localHistory[backOffset - 1] * Utils.ErrorWeights

                        ldiff = vecCurrent - vecStart
                        rdiff = vecPrediction - vecCurrent

                        ldiffn = normalize(ldiff)
                        rdiffn = normalize(rdiff)

                        cosineSimilarity = (1 + numpy.dot(ldiffn, rdiffn)) * 50
                        currentTimeError = math.sqrt(numpy.dot(rdiff, rdiff))

                        print(i, cosineSimilarity, currentTimeError, vecCurrent[4:])

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

                    if i > self.dlp:
                        mod = math.sin(i * 0.05) * 10 #** 640 * 30
                        self.boilerController.SetTarget(self.spTarg - math.floor(mod))

                    ax.collections.clear()
                    ax2.collections.clear()
                    #ax.fill_between(dataHolderRt[:len(comp)], comp - (2 * err), comp + (2 * err), facecolor='blue', alpha=0.25)


                    # Second Set
                    #print(dataP.shape)
                    #print(localHistory.shape)
                    predDataP = numpy.concatenate( [dataP[:-backOffset], localHistory.transpose()[4]] )
                    predDataT = numpy.concatenate( [dataT[:-backOffset], localHistory.transpose()[6] * 0.001] )
                    predDataX = numpy.concatenate( [dataX[:-backOffset], localHistory.transpose()[2]] )
                    predDataS = numpy.concatenate( [dataS[:-backOffset], localHistory.transpose()[5]] )
                    # print("\n\n")
                    # print(history[-1])
                    # print(localHistory[-1])
                    # print(boiler.GetBoilerWaterTemp())
                    #warn2.set_xdata(numpy.arange(0, len(dataP)) * simulator.timeDilation)
                    #warn2.set_ydata(data5)


                    dataP = numpy.concatenate([dataP, [self.boiler.GetBoilerWaterTemp()]])
                    dataT = numpy.concatenate([dataT, [self.boiler.boilerPercent * self.boiler.boilerPerformance * 0.001]])
                    dataX = numpy.concatenate([dataX, [self.boilerController.temperatureSetPoint]])
                    dataS = numpy.concatenate([dataS, [self.boiler.waterVolCurrent]])
                    dataClose = numpy.concatenate([dataClose, [cosineSimilarity]])
                    dataFar = numpy.concatenate([dataFar, [currentTimeError]])
                    #dataDiff = numpy.concatenate([dataDiff, [abs(ldiff - rdiff)]])
                    

                    #removalCutter = numpy.argmax(dataP > (dataP[-1] - iTime))

                    #dra.set_ydata(dataP[removalCutter:])
                    at = max((len(dataP) - 1) - iTime, 0)
                    dataP = dataP[at:]
                    dataT = dataT[at:]
                    dataS = dataS[at:]
                    dataX = dataX[at:]
                    dataClose = dataClose[at:]
                    dataFar = dataFar[at:]
                    dataDiff = dataDiff[at:]

                    #predat = max((len(predDataP) - 1) - iTime, 0)
                    predDataP = predDataP[at:]
                    predDataS = predDataS[at:]
                    predDataT = predDataT[at:]
                    predDataX = predDataX[at:]


                    dra2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    dra2.set_ydata(predDataP)
                    two2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    two2.set_ydata(predDataT)
                    three2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    three2.set_ydata(predDataS)
                    four2.set_xdata(numpy.arange(0, len(predDataP)) * self.simulator.timeDilation)
                    four2.set_ydata(predDataX)

                    
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
                    # warndiff.set_xdata(numpy.arange(0, len(dataP)) * self.simulator.timeDilation)
                    # warndiff.set_ydata(dataDiff)



                    ax.set_xlim(left=-5, right=len(predDataP) * self.simulator.timeDilation +5)
                    #ax2.fill_between(x, len(dataP) * self.simulator.timeDilation +5, len(predDataP) * self.simulator.timeDilation +5, facecolor='green', alpha=0.5)
                    #ax2.fill_between(numpy.arange(0, len(dataP)) * self.simulator.timeDilation, 0, 1, where=x > (len(dataP) * self.simulator.timeDilation), facecolor='green', alpha=0.5)
                    ax2.fill_between(numpy.arange(len(dataP) - 1, len(predDataP)) * self.simulator.timeDilation, 0, maxY, facecolor='purple', alpha=0.25)
                    ax2.fill_between(numpy.arange(len(dataP) - (1 + backOffset), len(dataP)) * self.simulator.timeDilation, 0, maxY, facecolor='pink', alpha=0.25)

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
    ax.set_ylim(top=maxY, bottom=-5)
    ax2.set_ylim(top=maxY, bottom=-5)
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