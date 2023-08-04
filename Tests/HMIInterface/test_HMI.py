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

from CloudMachineInterface.HMIInterface import HMIMPM
import SupportingEndpoints.HMIRecv
import waitress
import json
import threading
import socket

def waitressRT():
    # iamboolean = True

    # serv = waitress.create_server(SupportingEndpoints.HMIRecv.app, host="127.0.0.1", port=3090)
    # serv.run()

    # # Spinlock hahaha
    # while iamboolean:
    #     continue

    # serv.close()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 30090))
    s.listen(1)
    conn, addr = s.accept()
    with conn:
        print('Connected to ', addr)
        data = conn.recv(1024)
        if not data:
            raise Exception("Hi")
        assert(data == b'POST / HTTP/1.1\r\nHost: 127.0.0.1:30090\r\nUser-Agent: python-requests/2.23.0\r\nAccept-Encoding: gzip, deflate\r\nAccept: */*\r\nConnection: keep-alive\r\nContent-Length: 91\r\nContent-Type: application/json\r\n\r\n"{\\"WaterBoilerTemperatureSetPoint\\": 79, \\"WaterIngressRate\\": 0, \\"WaterEgressRate\\": 0}"')

        conn.send(b"HTTP/1.1 200 OK\nserver: nullable\ncontent-type: text/html\nx-clacks-overhead: GNU Terry Pratchett\naccept-ranges: bytes\nage: 0\nvary: Accept-Encoding\ncontent-length: 0\nX-Firefox-Spdy: h2")

    s.shutdown(socket.SHUT_RDWR)
    s.close()
        

def test_hmisend():
    waitressThread = threading.Thread(target=waitressRT)
    waitressThread.start()
    #waitress.serve(SupportingEndpoints.HMIRecv.app, listen="127.0.0.1:3090")


    connection = HMIMPM("http://127.0.0.1:30090/", None, None)

    hereisadict = {
        "WaterBoilerTemperatureSetPoint":79,
        "WaterIngressRate": 0,
        "WaterEgressRate": 0
    }

    connection.Send(json.dumps(hereisadict))

    waitressThread.join()

    assert(2==2)
