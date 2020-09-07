#!/usr/bin/env python3

from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler
import time

simulator = CSimulator(5.0, 5000)

boiler = simulator.SpawnObject(ABoiler, None)

simulator.BeginPlay()

for i in range(10):
    time.sleep(1)
    print("Average Simulation Rate (Dilated): {:.04f} hz".format(simulator.ProcessAvgFramerate()))
    print("Boiler Water Level is {:.03f}L. WOut: {:.02f}.".format(boiler.GetWaterLevel(), boiler.waterOutPres))

simulator.Shutdown()