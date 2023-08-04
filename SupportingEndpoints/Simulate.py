#!/usr/bin/env python3

##### ##### LICENSE ##### #####
# Copyright (C) 2021 Chris Anderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time

simulator = CSimulator(600, 600000)
#simulator = CSimulator(1, 200000)

boiler = simulator.SpawnObject(ABoiler, 10000, 0, 80, 30)
boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 65) # Heating to 95
#boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
boilerController.Possess(boiler)

boiler.SetInflowWaterTemp(24)
boiler.SetInflowRate(0.0)
boiler.SetOutflowRate(0.000)

# boiler.SetBoilerPower(100)



simulator.BeginPlay()

try:
    for i in range(100):
        time.sleep(1) # Not optional. This reads every second

        #simulator.SetTimeDilation(20 * (i + 1))
        #boiler.SetBoilerPower((i + 1) * 10)
        print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
        print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
        print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
        print("[TIME {:.02f}s] Power Perc: {:.02f}%".format((i + 1) * simulator.timeDilation, boiler.boilerPercent * 100))
        print("[TIME {:.02f}s] PID: {:.02f}i".format((i + 1) * simulator.timeDilation, boilerController.PID.iVal))
        print("[TIME {:.02f}s] PIDdbg: {}".format((i + 1) * simulator.timeDilation, boilerController.PID.dbgLastReturn))
except:
    pass
finally:
    simulator.Shutdown()

