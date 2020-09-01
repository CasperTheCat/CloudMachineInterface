#!/usr/bin/env python3

import threading
import time
import numpy

class CSimulator(object):
    def __init__(self, timeScale = 1.0, hertzLimit = 500):
        self.timeDilation = timeScale
        self.tickRateLimiter = hertzLimit - 0.001
        self.objects = []
        self.lastFrame = 0
        self.lastCollection = 0
        self.bIsPlaying = False
        self.objLock = threading.Lock()
        self.objCV = threading.Condition(self.objLock)
        self.shouldRun = True
        self.trSimThread = threading.Thread(target=self.RootTick, args=[])
        #self.trGarbageCollectionThread = threading.Thread(target=self.Reaper, args=[])

        self.lastUpdateTimes = numpy.array([0])
        self.lastUpdateTimesRt = numpy.array([0])
    

    def SpawnObject(self, nObj, *params):
        # Can we get control of the cv?
        obj = nObj(*params)

        with self.objLock:
            self.objects.append(obj)

        return obj


    def AddObject(self, nObj):
        # Can we get control of the cv?
        with self.objLock:
            self.objects.append(nObj)

        return nObj
            

    def BeginPlay(self):
        if not self.bIsPlaying:
            self.lastFrame = time.perf_counter()
            self.lastCollection = time.perf_counter()
            self.trSimThread.start()
            self.bIsPlaying = True


    def SimulateNTicks(self, nticks, timestep):
        limit = 0
        # As fast as possible
        delayedFramerate = numpy.zeros(10)
        delayCount = 0

        while limit < nticks:
            with self.objLock:
                # Update last frame (It's the start of the frame)
                # That way, time is consistent over the whole frame
                deltaTime = timestep * self.timeDilation


                # HelloWorld
                delayedFramerate[delayCount] = deltaTime
                delayCount += 1

                # Handle Updating Graph
                if delayCount == 10:
                    # Process before we reset!
                    baseTime = self.lastUpdateTimesRt[-1]
                    
                    self.lastUpdateTimes = numpy.concatenate([self.lastUpdateTimes, delayedFramerate])
                    self.lastUpdateTimesRt = numpy.concatenate([self.lastUpdateTimesRt, delayedFramerate + baseTime])

                    removalCutter = numpy.argmax(self.lastUpdateTimesRt > (deltaTime + baseTime - 30))

                    self.lastUpdateTimes = self.lastUpdateTimes[removalCutter:]
                    holdover = self.lastUpdateTimesRt[removalCutter]
                    self.lastUpdateTimesRt = self.lastUpdateTimesRt[removalCutter:] - holdover

                    delayCount = 0
                
                limit += 1

                # Tick the world
                for i in self.objects:
                    i.Tick(deltaTime)


    def SetTimeDilation(self, val):
        with self.objLock:
            self.timeDilation = val


    def ProcessAvgFramerate(self):
        realWindow = min(30, len(self.lastUpdateTimes))

        comp = numpy.convolve(self.lastUpdateTimes, numpy.ones((realWindow,))/realWindow, mode='valid')

        return 1 / numpy.mean(comp)


    def Reaper(self):
        while self.shouldRun:
            with self.objCV:
                while time.time() - self.lastCollection < 30.0:
                    self.objCV.wait()

                for i in self.objects:
                    if i.isDead:
                        del i


    def Shutdown(self):
        self.shouldRun = False
        if self.trSimThread:
            self.trSimThread.join()
       # self.trGarbageCollectionThread.join()


    def RootTick(self):
        # As fast as possible
        delayedFramerate = numpy.zeros(10)
        delayCount = 0
        timeFudge = 0

        while self.shouldRun:
            with self.objLock:
                # Time since last update
                lastUpdateTime = self.lastFrame
                thisUpdateTime = time.perf_counter()
                if timeFudge + (thisUpdateTime - lastUpdateTime) < 1.0/self.tickRateLimiter:
                    #print("On limiter. {} vs {}".format(thisUpdateTime - lastUpdateTime, 1.0/self.tickRateLimiter))
                    time.sleep((1.0/self.tickRateLimiter) - ( timeFudge + thisUpdateTime - lastUpdateTime))
                    #print("SLEEP: {}".format((1.0/self.tickRateLimiter) - (thisUpdateTime - lastUpdateTime)))
                    #continue

                # Update, we're under the limiter
                self.lastFrame = thisUpdateTime

                # Update last frame (It's the start of the frame)
                # That way, time is consistent over the whole frame
                deltaTime = (self.lastFrame - lastUpdateTime) * self.timeDilation

                timeFudge += (self.lastFrame - lastUpdateTime) - 1.0/self.tickRateLimiter

                # HelloWorld
                delayedFramerate[delayCount] = deltaTime
                delayCount += 1

                # Handle Updating Graph
                if delayCount == 10:
                    # Process before we reset!
                    baseTime = self.lastUpdateTimesRt[-1]
                    
                    self.lastUpdateTimes = numpy.concatenate([self.lastUpdateTimes, delayedFramerate])
                    self.lastUpdateTimesRt = numpy.concatenate([self.lastUpdateTimesRt, delayedFramerate + baseTime])

                    removalCutter = numpy.argmax(self.lastUpdateTimesRt > (deltaTime + baseTime - 30))

                    self.lastUpdateTimes = self.lastUpdateTimes[removalCutter:]
                    holdover = self.lastUpdateTimesRt[removalCutter]
                    self.lastUpdateTimesRt = self.lastUpdateTimesRt[removalCutter:] - holdover

                    delayCount = 0



                # Tick the world
                for i in self.objects:
                    i.Tick(deltaTime)

            


