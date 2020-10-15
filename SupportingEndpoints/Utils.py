from ProcessSimulation import CSimulator
from ProcessSimulation import AActor, ABoiler, ABoilerController
import time
import datetime
import numpy
import math

def TimeNow():
    print("WEEEEEE")
    return datetime.datetime.now().isoformat()


#https://stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray
def window_stack(a, stepsize=1, width=3, nummax=None):
    if not nummax:
        indexer = numpy.arange(width)[None, :] + stepsize * numpy.arange(a.shape[0] - (width))[:, None]
    else:
        indexer = numpy.arange(width)[None, :] + stepsize * numpy.arange(nummax)[:, None]

    return a[indexer]
    #return numpy.hstack( a[i:1 + i-width or None:stepsize] for i in range(0, width))


def MakeData(x,y, td, width, modRange, disable, boilerPower=10000, tankageLimits=(5,75,80), initWaterTemp=30, initCapacity=35, step=1, stack=True):
    simulator = CSimulator(td, 600000)
    #simulator = CSimulator(1, 200000)

    spTemp = y

    lowLevelAlarm, highLevelAlarm, totalCapacity = tankageLimits

    boiler = simulator.SpawnObject(ABoiler, boilerPower, initCapacity, totalCapacity, initWaterTemp)
    boilerController = simulator.SpawnObject(ABoilerController, lowLevelAlarm, highLevelAlarm, spTemp) # Heating to 95
    #boilerController = simulator.SpawnObject(ABoilerController, 50, 75, 95) # Heating to 95
    boilerController.Possess(boiler)

    boiler.SetInflowWaterTemp(24)
    boiler.SetInflowRate(0.0)
    boiler.SetOutflowRate(0.0001)

    if(disable):
        boilerController.SetDisableDisturbance()

    measurements = x // step
    #ins = [[],[],[],[],[],[]]

    # In Temperature, Inflow Rate, Outflow Rate
    user_disturbances = [[],[],[]] 

    # Water Level, Boiler Setpoint PID, Water Temperature + disturbances
    stateInformation = [[],[],[],[]]

    # Nothing here!
    outputs = []

    outs = [[]]

    for i in range(measurements):
        if i % 100 == 0:
            print("Time {}: {}".format(i * td * step, boiler.GetBoilerWaterTemp()))
        
        pastTickTemp = boiler.GetBoilerWaterTemp()

        # Simulate `step` seconds
        simulator.SimulateNTicks(step * 10, 1/10)

        if(not disable):
            mod = math.sin(i * 0.005) * modRange #** 640 * 30
            boilerController.SetTarget(spTemp - math.floor(mod))

        user_disturbances[0].append(boiler.waterInRatePerSecond)
        user_disturbances[1].append(boiler.GetInflowWaterTemp())
        user_disturbances[2].append(boiler.waterOutRatePerSecond)

        # stateInformation[0].append(boiler.waterInRatePerSecond)
        # stateInformation[1].append(boiler.GetInflowWaterTemp())
        # stateInformation[2].append(boiler.waterOutRatePerSecond)
        stateInformation[0].append(boilerController.temperatureSetPoint)
        stateInformation[1].append(boiler.waterVolCurrent)
        stateInformation[2].append(boiler.GetBoilerWaterTemp())
        #stateInformation[3].append(boiler.boilerPercent * boiler.boilerPerformance)
        stateInformation[3].append(boiler.boilerPercent * boiler.boilerPerformance)
        
        # print("[TIME {:.02f}s][{:.02f}h] Average Simulation Rate (Dilated): {:.04f} hz".format((i + 1) * simulator.timeDilation, ((i + 1) * simulator.timeDilation) / 3600, simulator.ProcessAvgFramerate()))
        # print("[TIME {:.02f}s] Boiler Water Level is {:.03f}L @ {:.02f}°C".format((i + 1) * simulator.timeDilation, boiler.GetWaterLevel(), boiler.GetBoilerWaterTemp()))
        # print("[TIME {:.02f}s] Power Used: {:.02f} kWh".format((i + 1) * simulator.timeDilation, boiler.GetPowerUse() / 3600000 ))
        # print("[TIME {:.02f}s] Power Perc: {:.02f}%".format((i + 1) * simulator.timeDilation, boiler.boilerPercent * 100))
        # print("[TIME {:.02f}s] PID: {:.02f}i".format((i + 1) * simulator.timeDilation, boilerController.PID.iVal))
        # print("[TIME {:.02f}s] PIDdbg: {}".format((i + 1) * simulator.timeDilation, boilerController.PID.dbgLastReturn))

        # #ins[3].append(boiler.waterVolCurrent)
        # ins[0].append(boiler.waterInRatePerSecond)
        # ins[1].append(boiler.GetInflowWaterTemp())
        # ins[2].append(boiler.waterOutRatePerSecond)
        # ins[3].append(boilerController.temperatureSetPoint)
        # ins[4].append(boiler.waterVolCurrent)
        # #ins[5].append(boiler.boilerPercent)
        # ins[5].append(boiler.GetBoilerWaterTemp())
        # #ins[4].append((i * 10) * simulator.timeDilation)

        # #ins[3].append(boiler.boilerPercent)


        #ins[3].append(pastTickTemp)
        #ins[3].append(boiler.boilerPercent)
        #ins[4].append(boiler.waterVolCurrent)

        outs[0].append(boiler.GetBoilerWaterTemp())


    disturbs = numpy.array([numpy.array(xi) for xi in user_disturbances]).transpose()
    states = numpy.array([numpy.array(xi) for xi in stateInformation]).transpose()

    if stack:
        disturbStacked = window_stack(disturbs, stepsize=1, width=width)
        stateStacked = window_stack(states, stepsize=1, width=width)

        return disturbStacked, stateStacked, disturbs[width:], states[width:]

    else:
        return disturbs, states, None, None

    # nins = numpy.array([numpy.array(xi) for xi in ins]).transpose()
    # nouts = numpy.array([numpy.array(xi) for xi in outs]).transpose()
    # rnins = nins#window_stack(nins, stepsize=1, width=width)
    # #nouts = window_stack(nouts, stepsize=1, width=width)
    # nouts = nouts#nouts[width:]

    # #print(nstack.shape)
    # print(rnins.shape)
    # print(nouts.shape)

    # ninstest = numpy.expand_dims(rnins[0], 0)
    # #nouts = numpy.expand_dims(nouts, 1)



    # return rnins, nouts, ninstest, nins[width:]

ran = numpy.arange(7)
ranStack = window_stack(ran, stepsize=1, width=3)
rO = ran[3:]

print(ranStack)
print(rO)

print(TimeNow())


