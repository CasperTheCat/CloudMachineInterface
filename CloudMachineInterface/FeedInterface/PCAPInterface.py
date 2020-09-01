#!/usr/bin/env python3

from CloudMachineInterface import FeedInterface

class PCAPFeed(FeedInterface.FeedInterface):
    def GetStream(self):
        raise NotImplementedError

    def GetTransmission(self):
        raise NotImplementedError