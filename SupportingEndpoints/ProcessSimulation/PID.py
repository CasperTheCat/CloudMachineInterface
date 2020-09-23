#!/usr/bin/env python3

class PIDController(object):
    def __init__(self, sp, kp, ki, kd):
        self.setPoint = sp
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.iVal = 0
        self.lastValue = 0

        self.dbgLastReturn = 0
        

    def Update(self, value, DeltaTime):
        #Get Prop
        error = self.setPoint - value

        self.iVal = self.iVal + error * DeltaTime
        derivative = (error - self.lastValue) / DeltaTime

        self.lastValue = error

        self.dbgLastReturn = [error * self.kp, self.iVal * self.ki, derivative * self.kd]

        return error * self.kp + self.iVal * self.ki + derivative * self.kd
