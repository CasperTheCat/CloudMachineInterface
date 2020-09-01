#!/usr/bin/env python3

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
    return make_response("Epic FAIL!")

@app.route("/", methods=['POST'])
def slashRoute():
    postreq = request.get_json()
    print(postreq)

if __name__ == "__main__":
    waitress.serve(app, listen="*:8080")