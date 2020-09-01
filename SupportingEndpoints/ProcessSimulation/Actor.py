#!/usr/bin/env python3

class AActor(object):
    def __init__(self, *argv):
        self.isDead = False
        self.owningController = None

    def Tick(self, DeltaTime: float):
        # Hello
        #print("RECV TICK: {}".format(DeltaTime))
        raise NotImplementedError
