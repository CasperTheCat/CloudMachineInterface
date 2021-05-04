#!/usr/bin/env python3

import pickle
import pandas
import numpy
import os
import sys
import matplotlib
import matplotlib.pyplot
import Utils

def ComputeRise(x, percent=0.65):
    minimum = numpy.min(x)
    maximum = numpy.max(x) - minimum

    riseLevel = numpy.percentile(x, percent)

    return numpy.argmax(x > maximum * percent)

def ComputeMedianSig(x):
    med = numpy.median(x)
    sig = numpy.std(x)
    
    back = x[::-1]

    return x.shape[0] -  numpy.argmax(back > (med-sig))




comboBox = [
    "DMDc.dat",
    "DMDc_Raw.dat",
    "DMDc_FilterBoth.dat",
    "Sindy.dat",
    "OKIDERA.dat",
    "BaseCase.dat",
    "FollowCase.dat",
    "OKIDERA_FilterData.dat",
    "Sindy_FilterData.dat",
    "DMDc_FilterData.dat",
    "BaseCase_FilterData.dat",
    "Sindy_FilterTrain.dat",
    "DMDc_FilterTrain.dat",
    "BaseCase_FilterTrain.dat",
    "Recurrent_FilterTrain.dat",
    "Recurrent_FilterData.dat",
    "Recurrent.dat"
]

# Create results directory
if not os.path.exists("./Results/"):
    os.makedirs("./Results/")

for runType in comboBox:
    print("Processing {}".format(runType))
    
    if not os.path.exists("Error_" + runType):
        continue

    RunError = []
    RunInclusiveTime = []
    RunEvaluationTime = []
    RunRetrainTime = []

    with open("Error_" + runType, "rb+") as f:
        RunError = pickle.load(f)
    
    with open("IncTime_" + runType, "rb+") as f:
        RunInclusiveTime = pickle.load(f)

    with open("EvalTime_" + runType, "rb+") as f:
        RunEvaluationTime = pickle.load(f)

    with open("RetrainTime_" + runType, "rb+") as f:
        RunRetrainTime = pickle.load(f)

    with open("RetrainReason_" + runType, "rb+") as f:
        RtReason = pickle.load(f)

    # Handle Run Error
    arrayOfThings = []
    for i in RunError.keys():
        arrayOfThings.append(RunError[i])

    # Handle Inc Time
    arrayOfIncl = []
    for i in RunInclusiveTime.keys():
        arrayOfIncl.append(RunInclusiveTime[i])


    RunError = numpy.array(arrayOfThings).astype(numpy.float)
    RunInclusiveTime = numpy.array(arrayOfIncl).astype(numpy.float) * 1000
    RunEvaluationTime = numpy.array(RunEvaluationTime).astype(numpy.float) * 1000
    RunRetrainTime = numpy.array(RunRetrainTime).astype(numpy.float) * 1000
    RtReason = numpy.array(RtReason)

    hostData = numpy.abs(RunError).cumsum(axis=0)
    _range = numpy.arange(hostData.shape[0])

    with open("GRAD_" + runType + ".json", "w+") as f:
        f.write("{ \"Runs\": [")
        for i in range(hostData.shape[1]):
            limit = hostData.transpose()[i].transpose()

            fit = numpy.polyfit(numpy.log(limit), _range, 1)
            grad = fit[0]

            index = ComputeRise(limit)
            rise = index * Utils.GetTimeStep()
            
            print("{{ \"Index\" : \"{}\"".format(index))
            print("\"Gradient\" : \"{}\"".format(fit[0]))
            print("\"Metrix\" : \"{}\"".format(rise / grad))
            print("")



            f.write("{")
            f.write("\"Index\" : \"{}\",\n".format(index))
            f.write("\"Gradient\" : \"{}\",\n".format(fit[0]))
            f.write("\"Metric\" : \"{}\",\n".format(rise / grad))
            f.write("\"RiseFraction\": \"{}\", \n".format(index / limit.shape[0]))
            f.write("\"MaxError\": \"{}\", \n".format(numpy.max(limit)))
            f.write("\"Median\" : \"{}\", \n".format(numpy.median(limit)))
            f.write("\"Time to Reach Med\" : \"{}\", \n".format(ComputeMedianSig(limit)))
            f.write("\"RiseFraction\": \"{}\", \n".format(index / limit.shape[0]))
            f.write("},")

        f.write("],}")

