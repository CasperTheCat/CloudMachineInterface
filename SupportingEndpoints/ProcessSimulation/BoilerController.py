#!/usr/bin/env python3

# Simulator for the water boiler
# No dontrol done here

from .Controller import AController
from .PID import PIDController
from .StdPID import StdPIDController
import math

class ABoilerController(AController):
    def __init__(self, lowWaterLevel, highWaterLevel, setpoint):
        self.lowWaterAlert = lowWaterLevel
        self.highWaterAlert = highWaterLevel
        self.temperatureSetPoint = setpoint
        self.Pawn = None
        #self.PID = PIDController(setpoint, 100, 0, 0)
        self.PID = PIDController(setpoint, 32, 0.00125, 8)
        #self.PID = StdPIDController(setpoint, 2, 4000, 8000)
        self.accTime = 0
        self.bRunDisturb = True
        super().__init__()

    def SetTarget(self, target):
        temp = self.temperatureSetPoint
        self.temperatureSetPoint = target

        # Update Controllers
        self.PID.UpdateSetpoint(target)

        return temp

    def SetEnableDisturbance(self):
        self.bRunDisturb = True

    def SetDisableDisturbance(self):
        self.bRunDisturb = False

    def Tick(self, DeltaTime: float):
        if self.Pawn:
            self.accTime += DeltaTime

            if self.bRunDisturb:
                boilerDegrader = max(0, self.accTime - 100000) / 100000 # After 200000, we reach peak bad boilerism
                self.Pawn.SetBoilerPerformancePercentage(1 - boilerDegrader * 0.5)

                boilerStickyControls = (max(0, self.accTime - 50000) / 100000)
                self.Pawn.SetBoilerControlTimeOffset(600 * boilerStickyControls)

            if self.bRunDisturb:
                pinOut = (math.sin(self.accTime * 0.01) * math.sin(self.accTime * 0.0001) * math.sin(self.accTime * 0.000001) + 1) * 0.1
                pinOut = math.sin(self.accTime * 0.01) * ((math.sin(self.accTime * 0.01) + math.sin(self.accTime * 0.01 * 1/3)) * 0.5) ** 2 * 1
                pinOut *= math.sin((self.accTime + 0.01) * 0.001)
                pinOut *= math.sin((self.accTime + 0.05) * 0.005)
                pinOut *= math.sin((self.accTime * 0.00003))
                #pinOut = (pinOut + 1.0) * 0.5
                pinOut = (pinOut) + 0.25
                pinOut = max(pinOut, 0)
                self.Pawn.SetOutflowRate(pinOut ** 2.2 * .65)

                pinIn = ((math.sin(self.accTime * 0.001) * math.sin(self.accTime * 0.0021)) + 1) * 0.1
                pinIn = math.sin(self.accTime * 0.001)
                pinIn *= math.sin(self.accTime * 0.000314)
                pinIn *= math.sin((self.accTime + 1) * 0.001)
                pinIn = (pinIn + 1) * 0.5
                self.Pawn.SetInflowRate(pinIn * 0.5 * 0.5)

            else:
                self.Pawn.SetInflowRate(0.01)
                self.Pawn.SetOutflowRate(0.001)

            temps = math.cos(self.accTime * 0.001)
            self.Pawn.SetInflowWaterTemp(((temps * 0.5) + 0.5) * 30)

            # # Waterlevels
            if self.Pawn.GetWaterLevel() < self.lowWaterAlert:
                self.Pawn.SetOutflowRate(0)

            if self.Pawn.GetWaterLevel() > self.highWaterAlert:
                self.Pawn.SetInflowRate(0)

            # Water Power Level
            # Dumb System
            #if self.Pawn.GetBoilerWaterTemp() < self.temperatureSetPoint:
            
            r = self.Pawn.SetBoilerPower(self.PID.Update(self.Pawn.GetBoilerWaterTemp(), DeltaTime))
            #print(r)
            #r = self.Pawn.SetBoilerPower((self.temperatureSetPoint - self.Pawn.GetBoilerWaterTemp()) * 100)
            #print("Boiler set to {}%".format(r *  100))


    def Possess(self, AActor):
        AActor.owningActor = self
        self.Pawn = AActor
