#!/usr/bin/env python3

class FeedInterface(object):
    def GetStream(self):
        raise NotImplementedError

    def GetTransmission(self):
        raise NotImplementedError