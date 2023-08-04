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

class PIDController(object):
    def __init__(self, sp, kp, ki, kd):
        self.setPoint = sp
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.iVal = 0
        self.lastValue = 0

        self.dbgLastReturn = 0
        
    def UpdateSetpoint(self, value):
        self.setPoint = value

    def Update(self, value, DeltaTime):
        # Get Prop
        error = self.setPoint - value

        # Make positive error higher
        # if error < 0:
        #     error = error * 10

        self.iVal = self.iVal + error * DeltaTime
        derivative = (error - self.lastValue) / DeltaTime

        self.lastValue = error

        self.dbgLastReturn = [error * self.kp, self.iVal * self.ki, derivative * self.kd]

        return error * self.kp + self.iVal * self.ki + derivative * self.kd

