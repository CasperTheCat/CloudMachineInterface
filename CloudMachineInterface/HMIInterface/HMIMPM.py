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

# For urandom
import os

# Encryption and Auth Packages
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
#from cryptography.hazmat.primitives.asymmetric.ec

# Networking with HTTPS
import requests


## Human-Machine Interface Message Passing Module
## Handles message passing by passing JSON

class HMIMPM(object):
    """We do not receive data on this line!"""
    def __init__(self, endpoint, user, auth):
        self.endpoint = endpoint
        self.username = user
        self.token = auth

    def Send(self, tag):
        """Send arbitrary JSON Tags"""
        r = requests.post(self.endpoint, json=tag)
        return r.status_code
