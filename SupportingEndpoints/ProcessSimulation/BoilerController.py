#!/usr/bin/env python3

# Simulator for the water boiler
# No dontrol done here

from .Controller import AController

class ABoilerController(AController):
    def __init__(self, lowWaterLevel, highWaterLevel, setpoint):
        self.lowWaterAlert = lowWaterLevel
        self.highWaterAlert = highWaterLevel
        self.temperatureSetPoint = setpoint
        self.Pawn = None
        super().__init__()


    def Tick(self, DeltaTime: float):
        if self.Pawn:
            # Waterlevels
            if self.Pawn.GetWaterLevel() < self.lowWaterAlert:
                self.Pawn.SetInflowRate(1)
            elif self.Pawn.GetWaterLevel() > self.highWaterAlert:
                self.Pawn.SetInflowRate(0)

            # Water Power Level
            # Dumb System
            #if self.Pawn.GetBoilerWaterTemp() < self.temperatureSetPoint:
            r = self.Pawn.SetBoilerPower((self.temperatureSetPoint - self.Pawn.GetBoilerWaterTemp()) * 100)
            #print("Boiler set to {}%".format(r *  100))


    def Possess(self, AActor):
        AActor.owningActor = self
        self.Pawn = AActor
