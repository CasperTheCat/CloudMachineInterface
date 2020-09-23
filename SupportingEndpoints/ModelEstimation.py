#!/usr/bin/env python3

import control
import modred
from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import matplotlib
import matplotlib.pyplot
matplotlib.interactive(True)
matplotlib.use("TkAgg") 
import numpy as np




simulator = CSimulator(30, 600000)
#simulator = CSimulator(1, 200000)

spTemp = 65

boiler = simulator.SpawnObject(ABoiler, 10000, 30, 80, 30)
boilerController = simulator.SpawnObject(ABoilerController, 5, 75, spTemp) # Heating to 95
#boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
boilerController.Possess(boiler)

boiler.SetInflowWaterTemp(24)
boiler.SetInflowRate(0.0)
boiler.SetOutflowRate(0.0001)

measurements = 100
ins = [[],[],[]]
outs = [[]]
for i in range(measurements):
    simulator.SimulateNTicks(1000, 1/1000)

    ins[0].append(boiler.boilerPercent)
    ins[1].append(boiler.waterInRatePerSecond)
    ins[2].append(boiler.waterOutRatePerSecond)

    outs[0].append(boiler.GetBoilerWaterTemp())


print(control.era(OKID(ins, outs, 50), 50, 50, 3, 1, 5))




