#!/usr/bin/env python3

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
