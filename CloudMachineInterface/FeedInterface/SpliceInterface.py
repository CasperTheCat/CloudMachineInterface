#!/usr/bin/env python3

import os
import struct
from CloudMachineInterface import FeedInterface

class SpliceFeed(FeedInterface.FeedInterface):
    def GetStream(self):
        raise NotImplementedError

    def GetTransmission(self):
        raise NotImplementedError