import matplotlib
import matplotlib.pyplot
import math
import numpy

nSeconds = 10
nHertz = 35
nSamplingRate = 1000

y = numpy.arange(nSamplingRate * nSeconds) * (1 / nSamplingRate)
sm = numpy.sin(y * nHertz * 2 * 3.1415926535)


# This is the Nyquist Rate
nSecondSampleRate = 14

# Half of this is the nyquist frequency
# nHertz has to be less or equal to half of this number

# Technically, this is actually in half-hertz increments
# nHertz up top requires the *2 for similar reasons
# One full cycle is 2Pi radians
nSecondSampleRate = nSecondSampleRate * 2

ss = numpy.arange(nSecondSampleRate * nSeconds) * (1 / nSecondSampleRate)

# Compute sample time to index
st = (ss * (nSamplingRate)).astype(numpy.uint32) 
print(max(st))
sx = sm[st]

print(sx)




maxY = 105
maxTDPI = 200
resolution = numpy.array((1920, 1080))
TargetDPI = maxTDPI
solvedSize = resolution / TargetDPI

#fig = matplotlib.pyplot.figure(0)
fig, ((ax1), (ax2)) = matplotlib.pyplot.subplots(2, 1, sharex=True, sharey=True, dpi=TargetDPI, figsize=solvedSize)

ax1.scatter(ss, sx, s=6.5, c='r',marker='x', zorder=2)
ax1.plot(y, sm, linewidth=2.0, zorder=1)
ax2.plot(ss, sx)

ax2.set_xlabel("Time (Seconds)")
ax1.set_ylabel("Arbitrary Axis")
ax2.set_ylabel("Arbitrary Axis")

ax1.set_title("Aliased Signal")


fig.show()

input()
