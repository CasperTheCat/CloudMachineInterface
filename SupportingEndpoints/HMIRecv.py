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

from flask import Flask, render_template, request, make_response
import random

# Uncomment for actually running in production, but this is dummy code...
import waitress

CONST_FLASK_SIMULATEFAILURES = False
CONST_FAKE_RETCODE = [404, 403, 500, 522]

## Flask Header Overrides
class lyqFlask(Flask):
    def process_response(self, response):
        tresp = super().process_response(response)
        tresp.headers['Server'] = "Flask HMI Dummy"
        tresp.headers['Cache-Control'] = "max-age=0, nocache, nostore"
        return(tresp)

def ObfuscateRetCode(NominalCode: int):
    if CONST_FLASK_SIMULATEFAILURES:
        if random.random() < 0.05:
            return random.choice(CONST_FAKE_RETCODE)
    return NominalCode

app = lyqFlask(__name__)

@app.route("/", methods=['GET', 'DELETE', 'HEAD', 'OPTIONS'])
def nullRoute():
    return make_response("FAIL!")

@app.route("/", methods=['POST'])
def slashRoute():
    postreq = request.get_json()
    print(postreq)

if __name__ == "__main__":
    waitress.serve(app, listen="*:8080")