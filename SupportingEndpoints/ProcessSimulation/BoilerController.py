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

from .Controller import AController
from .PID import PIDController
from .StdPID import StdPIDController
import random
import math

class ABoilerController(AController):
    def __init__(self, lowWaterLevel, highWaterLevel, setpoint, seed=0):
        self.lowWaterAlert = lowWaterLevel
        self.highWaterAlert = highWaterLevel
        self.temperatureSetPoint = setpoint
        self.Pawn = None
        #self.PID = PIDController(setpoint, 100, 0, 0)

        # Below PID is used for most tests
        self.PID = PIDController(setpoint, 32, 0.00125, 8)

        # Uncomment this one for the Chapter 7 tests
        self.PID = PIDController(setpoint, 32, 0.0025, 8)

        #self.PID = StdPIDController(setpoint, 2, 4000, 8000)
        self.accTime = 0
        self.bRunDisturb = True
        self.uDisturbanceSeed = seed

        self.rng = random.Random(seed)

        self.boilerDegraderSeconds = 80000 + max(0, self.rng.gauss(1, 0.5)) * 40000
        self.boilerStickyStart = 40000 + max(0, self.rng.gauss(1, 0.5)) * 25000
        self.boilerStickyEnd = self.boilerStickyStart + max(0, self.rng.gauss(1, 0.5)) * 30000
        self.boilerStickyAmount = 0 + max(0, self.rng.gauss(1, 0.5)) * 1200
        
        self.boilerPOutScale = 0.5 + max(0, self.rng.gauss(1, 0.5))
        self.boilerPInScale = 0.5 + max(0, self.rng.gauss(1, 0.5))
        self.boilerPOutOffsetScale1 = 30 + max(0, self.rng.gauss(1, 0.5)) * 1000
        self.boilerPOutOffsetScale2 = 30 + max(0, self.rng.gauss(1, 0.5)) * 1000
        self.boilerPOutOffsetScale3 = 30 + max(0, self.rng.gauss(1, 0.5)) * 1000
        self.boilerPInOffsetScale1 = 30 + max(0, self.rng.gauss(1, 0.5)) * 1000
        self.boilerPInOffsetScale2 = 30 + max(0, self.rng.gauss(1, 0.5)) * 1000
        self.boilerPInOffsetScale3 = 30 + max(0, self.rng.gauss(1, 0.5)) * 1000

        self.inWaterBase = max(0, self.rng.gauss(1, 0.5)) * 20
        self.inWaterFlux = 5 + max(0, self.rng.gauss(1, 0.5)) * 10
        self.inWaterTemp = max(0, self.rng.gauss(1, 0.5)) * self.inWaterFlux

        self.boilerGoneSeconds = 80000 + max(0, self.rng.gauss(1, 0.5)) * 40000

        print("self.boilerDegraderSeconds is {}".format(self.boilerDegraderSeconds))
        print("self.boilerStickyStart is {}".format(self.boilerStickyStart))
        print("self.boilerStickyEnd is {}".format(self.boilerStickyEnd))
        print("self.boilerStickyAmount is {}".format(self.boilerStickyAmount))
        print("self.boilerPOutScale is {}".format(self.boilerPOutScale))
        print("self.boilerPInScale is {}".format(self.boilerPInScale))
        print("self.boilerPOutOffsetScale1 is {}".format(self.boilerPOutOffsetScale1))
        print("self.boilerPOutOffsetScale2 is {}".format(self.boilerPOutOffsetScale2))
        print("self.boilerPOutOffsetScale3 is {}".format(self.boilerPOutOffsetScale3))
        print("self.boilerPInOffsetScale1 is {}".format(self.boilerPInOffsetScale1))
        print("self.boilerPInOffsetScale2 is {}".format(self.boilerPInOffsetScale2))
        print("self.boilerPInOffsetScale3 is {}".format(self.boilerPInOffsetScale3))
        print("self.inWaterBase is {}".format(self.inWaterBase))
        print("self.inWaterTemp is {}. Flux is {}".format(self.inWaterTemp,self.inWaterFlux))
        print("self.boilerGoneSeconds is {}".format(self.boilerGoneSeconds))

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
                boilerDegrader = min(1, max(0, self.accTime - self.boilerDegraderSeconds) / self.boilerGoneSeconds) # After 200000, we reach peak bad boilerism
                self.Pawn.SetBoilerPerformancePercentage(1 - boilerDegrader * 0.60)

                boilerStickyControls = (max(0, self.accTime - self.boilerStickyStart) / self.boilerStickyEnd)
                self.Pawn.SetBoilerControlTimeOffset(300 + self.boilerStickyAmount * boilerStickyControls)

            if self.bRunDisturb:
                pinOut = (math.sin(self.accTime * 0.01) * math.sin(self.accTime * 0.0001) * math.sin(self.accTime * 0.000001) + 1) * 0.1
                pinOut = math.sin(self.accTime * 0.01) * ((math.sin(self.accTime * 0.01) + math.sin(self.accTime * 0.01 * 1/3)) * 0.5) ** 2 * 1
                pinOut *= math.sin((self.accTime + self.boilerPOutOffsetScale1) * 0.001)
                pinOut *= math.sin((self.accTime + self.boilerPOutOffsetScale2) * 0.005)
                pinOut *= math.sin((self.accTime + self.boilerPOutOffsetScale3 * 0.00003))
                #pinOut = (pinOut + 1.0) * 0.5
                pinOut = (pinOut) + 0.25
                pinOut = max(pinOut, 0)
                self.Pawn.SetOutflowRate(self.boilerPOutScale * (pinOut ** 2.2 * .65))

                pinIn = ((math.sin(self.accTime * 0.001) * math.sin(self.accTime * 0.0021)) + 1) * 0.1
                pinIn = math.sin(self.accTime * 0.001)
                pinIn *= math.sin(self.accTime * 0.000314)
                pinIn *= math.sin((self.accTime + self.boilerPInOffsetScale1) * 0.001)
                pinIn = (pinIn + 1) * 0.5
                self.Pawn.SetInflowRate(self.boilerPInScale * (pinIn * 0.5 * 0.5))

            else:
                self.Pawn.SetInflowRate(0.01)
                self.Pawn.SetOutflowRate(0.001)

            temps = math.cos(self.accTime * 0.001)
            self.Pawn.SetInflowWaterTemp(
                self.inWaterBase + ((temps * 0.5) + 0.5) * self.inWaterTemp)

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
