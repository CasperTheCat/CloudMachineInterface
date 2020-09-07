#!/usr/bin/env python3

# Simulator for the water boiler
# No dontrol done here

from .Actor import AActor

class ABoiler(AActor):
    def __init__(self, kwargs):
        self.setTemp = 0
        self.waterInRatePerSecond = 10
        self.waterOutRatePerSecond = 0.2
        self.waterCapacity = 100
        self.waterVolCurrent = 0
        self.waterOutPres = 0

    def GetWaterLevel(self):
        return self.waterVolCurrent

    def Tick(self, DeltaTime: float):
        # In
        self.waterVolCurrent += self.waterInRatePerSecond * DeltaTime
        self.waterOutPres = self.waterOutRatePerSecond * self.waterVolCurrent
        self.waterVolCurrent -= self.waterOutPres * DeltaTime
        
        #print("Ticking Boiler. WL: {}, {:.02f}%".format(self.waterVolCurrent, (self.waterVolCurrent / self.waterCapacity) * 100))

    