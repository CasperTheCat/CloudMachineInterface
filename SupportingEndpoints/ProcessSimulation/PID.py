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

