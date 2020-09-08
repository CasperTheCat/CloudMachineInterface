#!/usr/bin/env python3

# Simulator for the water boiler
# No dontrol done here

from .Actor import AActor

class ABoiler(AActor):
    def __init__(self, initBoilerPowerWatts, initWaterVol, initWaterCap, initTemperature):
        self.waterTemperature = initTemperature

        self.waterInRatePerSecond = 2
        self.waterInTemperature = 0

        self.waterOutRatePerSecond = 0.2
        self.waterOutPres = 0

        # Litres and Watts
        self.dissipationRateCPerSecond = 1 / 3600
        self.waterCapacity = initWaterCap
        self.waterVolCurrent = initWaterVol
        self.boilerPercent = 0
        self.boilerCapacity = initBoilerPowerWatts

        self.powerUseWatts = 0

        super().__init__()
        

    def GetWaterLevel(self):
        return self.waterVolCurrent

    def GetBoilerWaterTemp(self):
        return self.waterTemperature

    def GetInflowWaterTemp(self):
        return self.waterInTemperature

    def GetPowerUse(self):
        return self.powerUseWatts


    # Setters for controlling the world
    def SetInflowWaterTemp(self, SetPoint: float):
        self.waterInTemperature = SetPoint

    def SetInflowRate(self, SetPoint: float):
        self.waterInRatePerSecond = SetPoint

    def SetOutflowRate(self, SetPoint: float):
        self.waterOutRatePerSecond = SetPoint

    def SetBoilerPower(self, Percent: float):
        sanitisedPercent = max(min(Percent / 100, 1), 0)
        self.boilerPercent = sanitisedPercent

        return self.boilerPercent

    def Tick(self, DeltaTime: float):
        # In Handle Water!
        lWaterIncoming = self.waterInRatePerSecond * DeltaTime

        self.waterOutPres = self.waterOutRatePerSecond * self.waterVolCurrent
        lWaterLeaving = self.waterOutPres * DeltaTime

        self.waterVolCurrent -= lWaterLeaving * 0.5

        if (self.waterVolCurrent > 0):
            s1 = (self.waterTemperature * self.waterVolCurrent + lWaterIncoming * self.waterInTemperature) / (self.waterVolCurrent + lWaterIncoming)
            s2 = (self.boilerPercent * self.boilerCapacity * DeltaTime) / (4200 * self.waterVolCurrent)
            newTemp = (s1 + s2) - self.dissipationRateCPerSecond * DeltaTime
            self.waterTemperature = newTemp


        self.waterVolCurrent += lWaterIncoming
        self.waterVolCurrent -= lWaterLeaving * 0.5

        self.powerUseWatts += self.boilerPercent * self.boilerCapacity * DeltaTime

        #s2 = (4200 * lWaterIncoming * self.GetInflowWaterTemp()) / (4200 * self.waterVolCurrent)
        # # Heat
        # # We lost as much energy as the energy leaving was carrying
        # outEnergy = 4.2 * lWaterLeaving * self.GetBoilerWaterTemp()
        # inEnergy = 4.2 * lWaterIncoming * self.GetInflowWaterTemp()
        # boilerEnergy = self.boilerPercent * self.boilerCapacity

        # print((inEnergy - outEnergy) + boilerEnergy)

        
        #print("Ticking Boiler. WL: {}, {:.02f}%".format(self.waterVolCurrent, (self.waterVolCurrent / self.waterCapacity) * 100))

    