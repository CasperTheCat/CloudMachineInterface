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
        self.PID = PIDController(setpoint, 16, 0.0025, 8)
        #self.PID = StdPIDController(setpoint, 2, 2000, 1000)
        self.accTime = 0
        super().__init__()


    def Tick(self, DeltaTime: float):
        if self.Pawn:
            self.accTime += DeltaTime

            pinOut = (math.sin(self.accTime * 0.01) * math.sin(self.accTime * 0.0001) * math.sin(self.accTime * 0.000001) + 1) * 0.1
            pinOut = math.sin(self.accTime * 0.01) * ((math.sin(self.accTime * 0.01) + math.sin(self.accTime * 0.01 * 1/3)) * 0.5) ** 2 * 1
            pinOut *= math.sin((self.accTime + 0.01) * 0.001)
            pinOut *= math.sin((self.accTime + 0.05) * 0.005)
            pinOut *= math.sin((self.accTime * 0.00003))
            #pinOut = (pinOut + 1.0) * 0.5
            pinOut = (pinOut) + 0.25
            pinOut = max(pinOut, 0)
            self.Pawn.SetOutflowRate(pinOut ** 2 * .75)


            pinIn = ((math.sin(self.accTime * 0.001) * math.sin(self.accTime * 0.0021)) + 1) * 0.1
            pinIn = math.sin(self.accTime * 0.001)
            pinIn *= math.sin(self.accTime * 0.000314)
            pinIn *= math.sin((self.accTime + 1) * 0.001)
            pinIn = (pinIn + 1) * 0.5
            self.Pawn.SetInflowRate(pinIn * 0.5)

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
