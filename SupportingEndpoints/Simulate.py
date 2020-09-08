#!/usr/bin/env python3

from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time

simulator = CSimulator(600, 60000)

boiler = simulator.SpawnObject(ABoiler, 4200, 0, 80, 30)
boilerController = simulator.SpawnObject(ABoilerController, 5, 75, 65)
boilerController.Possess(boiler)

boiler.SetInflowWaterTemp(25)
boiler.SetInflowRate(0.0)
boiler.SetOutflowRate(0.0)

# boiler.SetBoilerPower(100)



simulator.BeginPlay()

try:
    for i in range(100):
        time.sleep(1)
        #boiler.SetBoilerPower((i + 1) * 10)
        print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
        print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
        print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
except:
    pass
finally:
    simulator.Shutdown()

