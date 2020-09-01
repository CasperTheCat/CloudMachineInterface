#!/usr/bin/env python3

class AController(object):
    def __init__(self, *argv):
        self.isDead = False

    def Tick(self, DeltaTime: float):
        raise NotImplementedError

    def Possess(self, AActor):
        raise NotImplementedError
