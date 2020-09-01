#!/usr/bin/env python3

class StdPIDController(object):
    def __init__(self, sp, kp, ti, td):
        self.setPoint = sp
        self.kp = kp
        self.ki = ti
        self.kd = td

        self.iVal = 0
        self.lastValue = 0

        self.dbgLastReturn = 0
        
    def UpdateSetpoint(self, value):
        self.setPoint = value

    def Update(self, value, DeltaTime):
        # Get Prop
        error = self.setPoint - value
        self.iVal = self.iVal + error * DeltaTime
        derivative = (error - self.lastValue) / DeltaTime
        self.lastValue = error
        self.dbgLastReturn = [self.kp * (error + (1/self.ki) * self.iVal + (1/self.kd) * derivative), 0, 0]

        return self.kp * (error + (1/self.ki) * self.iVal + (1/self.kd) * derivative)
