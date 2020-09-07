#!/usr/bin/env python3

class AActor(object):
    def __init__(self, kwargs):
        self.isDead = False

    def Tick(self, DeltaTime: float):
        # Hello
        #print("RECV TICK: {}".format(DeltaTime))
        pass
