# Cloud Machine Interface
The Cloud Machine Interface (CMI) formed a starting point for this research. The goal of the CMI is to be a trusted, secure platform for retrofitting Industry 4.0.

### Jargon

#### PLC

The Programmable Logic Controller (PLC) is a realtime system that monitors connected sensors and actuators and makes control decisions based on its programming.

These devices may be distant from the operators that run the site.

Updates to these devices are done via an engineering workstation. Commonly, PLCs are programmed with ladder logic.

#### HMI

The Human-Machine Interface (HMI) is the primary way that operators can control the site. Additionally, alarms and warnings are visible on the HMI.

HMIs can read values and set target on the PLC but not update the logic.

#### Process

The process is the physical system under control. For our project, this is a boiler.

#### Splice

The splice is abstraction of a T-Junction with a one-way connection to the CMI. This simply prevents the CMI from being to interrupt to datastream between the HMI and the PLC. 

By preventing this connection at a physical level or electronic level, we can build trust in the system to not interfere with the plant.

## Diagram

Below is a diagram of the Cloud Machine Interface.
![Architecture Diagram](Diagram.png)

In this diagram, the CMI connects to the site via a splice between an HMI and a PLC. This provides us with any and all data that the HMI can see.


## Twin Interface

One of the requirements, from an engineering perspective, is a common, abstract interface that describes a twin.
This interface needs to provide a way to retrieve values from the twin alongside some representation of what the value means.

In PLC land, there are tags that represent what the value is, so providing a name in the interface should be permissable.


## Digital Twinning

A digital twin is a form of high-fidelity simulation that uses sensor data from a physical twin to inform its state.

One of the key benefits of a digital twin is being able to evaluate the result of a scenario without risking the physical twin.


## System Identification

Python forms the base of this project up to this point, and Python 3.6+ should be installed on your computer.

We use system identification to construct a digital twin based on historical data from a "physical" twin. At present, this twin is a simulation.

The two libraries used are `control`, which contains a large array of control, and `modred`. The latter is required at present due to `control` not implementing eigensystem realisation (`control.era`) yet. 

### Examples and Helpful Tips




## Project Resources

* [Steve Brunton's Youtube](https://www.youtube.com/channel/UCm5mt-A4w61lknZ9lCsZtBw)
* [Tensorflow](https://www.tensorflow.org/)
* [Digital Twin: Mitigating Unpredictable, Undesirable Emergent Behavior in Complex Systems](https://doi.org/10.1007/978-3-319-38756-7_4)
* [The Digital Twin Paradigm](https://arc.aiaa.org/doi/10.2514/6.2012-1818)
* [Towards Security-Aware Virtual Environments for Digital Twins](http://doi.acm.org/10.1145/3198458.3198464)