
'''
This rk code follows closely to numerical recipes. I highly recommend
reading the chapter on this to better understand why this code is
the way it is.

I have four examples laid out, each with a stand alone function
to run the example. At the very end of the code are the function
calls to run the examples, but all but the Bessel example are commented out.

The examples are:
exponential decay
SHO (with dampenning if you give it a different value in the code)
Bessel function (order zero, but you can also change this with a single value in the code)
free particle in polar coordinates (to explore order of magnitude difference of variables)
'''

#fraction time coefficients
a2 = 1/5
a3 = 3/10
a4 = 3/5
a5 = 1
a6 = 7/8

#previous value weights
b21 = 1/5

b31 = 3/40
b32 = 9/40

b41 = 3/10
b42 = -9/10
b43 = 6/5

b51 = -11/54
b52 = 5/2
b53 = -70/27
b54 = 35/27

b61 = 1631/55296
b62 = 175/512
b63 = 575/13824
b64 = 44275/110592
b65 = 253/4096

#fifth order coefs
c1 = 37/378
c2 = 0
c3 = 250/621
c4 = 125/594
c5 = 0
c6 = 512/1771

#fourth order minus fifth order coefs
d1 = 2825/27648-c1
d2 = 0
d3 = 18575/48384-c3
d4 = 13525/55296-c4
d5 = 277/14336
d6 = 1/4-c6

def rk4Step(state, stateDerivative, stateDerivativeFunct, t, dt):
	'''
	This uses and "embedded" runge kutta algorithm, which
	is about twice as fast as the half step algorithm.
	It finds six intermediate steps (instead of the standard four)
	which may be combined with specific coefficients to get a final
	step accurate to fifth order (instead of the standard four)
	or to fourth order with a different set of coefficients.
	The difference of these will be the fifth order term which
	the fourth order RK misses, and so is used for error estimation.
	The fact that error estimates are valid for the fourth order
	part, but the fifth order part is actually used is what makes
	this a "45" algorithm, having parts both fourth and fifth order.
	'''

	ds1 = dt * stateDerivative
	ds2 = dt * stateDerivativeFunct(
		state + b21 * ds1,
		t + a2 * dt)
	ds3 = dt * stateDerivativeFunct(
		state + b31 * ds1 + b32 * ds2,
		t + a3 * dt)
	ds4 = dt * stateDerivativeFunct(
		state + b41 * ds1 + b42 * ds2 + b43 * ds3,
		t + a4 * dt)
	ds5 = dt * stateDerivativeFunct(
		state + b51 * ds1 + b52 * ds2 + b53 * ds3 + b54 * ds4,
		t + a5 * dt)
	ds6 = dt * stateDerivativeFunct(
		state + b61 * ds1 + b62 * ds2 + b63 * ds3 + b64 * ds4 + b65 * ds5,
		t + a6 * dt)

	newState = state + (c1 * ds1 + c3 * ds3 + c4 * ds4 + c6 * ds6)
	deltaState = (d1 * ds1 + d3 * ds3 + d4 * ds4 + d5 * ds5 + d6 * ds6)

	return (newState, deltaState)

def rk4FullStep(state, stateDerivativeFunct, t, dt, numericalErrorFunct):
	initialDerivative = stateDerivativeFunct(state, t)
	'''
	Completing a full step properly with dynamic step size is not as simple
	as just calling the above function. The deltaState return must
	be sent to the numericalErrorFunct to check if it is within tolerance.
	If it is above tolerance (out of tolerance) then the step must be repeated
	with a smaller step size. If it is below tolerance (within tolerance) then
	the step is successful, but we should still adjust the next step to be larger
	so that we don't waist computer resources.

	For this, I have a for loop to 20 (This is only a fail safe.
	a better code would implement a better check for the integrator getting
	stuck, and this would be a while(true) instead of a for loop.)
	This loop tries the integration step until succesful, which is
	decided by whether numericalErrorFunct returns a value greater than
	or less than 1. (See the numericalError function in the besselExample
	function below for more details on this function.)
	'''
	for i in range(20):
		(newState, deltaState) = rk4Step(
			state, initialDerivative, stateDerivativeFunct, t, dt)

		ds = numericalErrorFunct(newState, deltaState, initialDerivative, dt)

		if (ds <= 1):
			# print(t)
			'''
			If we are under tolerance, then it was a success,
			but we can afford to make dt larger. We say this by
			returning a new dt value (last return slot).
			On top of the expected value for what the new dt
			should be, we tack on a 0.9 to give a little wiggle room.
			(Other wise the failures would be very common and
			the algorithm may even get caught in a loop.)
			read numerical recipes to see why 1/5 is used here
			'''
			# return (newState, dt, dt * ds**(-1/5))
			return (newState, dt, 0.9 * dt * ds**(-1/5))
		else:
			#print is unnecessary, just here to keep tabs on how often things
			#are out of spec. Since this is a failed step, we adjust dt and try again.
			# print('failed at t = ' + str(t) + ' with dt = ' + str(dt) + ' and ds = ' + str(ds))
			dt *= max(0.9 * ds**(-1/4),.1)
			continue

def rk4(initialState, stateDerivativeFunct, numericalErrorFunct, t0, t1, init_dt, useDynamicTimeStep = True):
	t = t0
	timeList = [t]
	stateList = [initialState.copy()]
	
	if (useDynamicTimeStep):

		new_dt = init_dt

		while (t < t1):
			(newState, init_dt, new_dt) = rk4FullStep(
				stateList[-1], stateDerivativeFunct, t, new_dt, numericalErrorFunct)
			'''
			At this point, init_dt is the time step which was used
			and should be added to the current time, while new_dt is
			the time step we should use next time. Hence, in the rk4FullStep call,
			new_dt is used.
			'''

			stateList.append(newState)

			t += init_dt

			timeList.append(t)

	else:
		while (t < t1):
			(newState, deltaState) = rk4Step(
				stateList[-1], stateDerivativeFunct(stateList[-1],t),
				stateDerivativeFunct, t, init_dt)

			stateList.append(newState)

			t += init_dt

			timeList.append(t)

	return timeList, stateList

#####-------------------------Examples---------------------------#####

import numpy as np
from matplotlib import pyplot as plt
from scipy import special

def exponentialExample():
	'''
	model e^-x by solving
	dy/dx = -y
	'''

	initState = np.array([1])
	#RK4 code uses "copy" function, supported by np array,
	#and requires addition and scalar multiplication, also supported by np array
	tolerance = 1e-6
	t0 = 0
	t1 = 10
	initDT = 0.08
	'''
	Choosing initDT to be too large or too small can be problematic.
	Too large and the first step will accumulate error outside of tolerance,
	too small and the second step may accumulate too much error.
	This is because the dynamic time step may overcompensate for having
	too small of a time step and make the time step too large for the second
	step. (For the most part, this isn't a big problem, just something
	to keep in mind.)
	'''

	def stateDerivative(inputState, inputTime):
		return -inputState

	def errorFunct(currentState, deltaState, derivative, dt):
		return np.abs(deltaState[0]/currentState[0])/tolerance
		#This shouldn't cross y=0, so we don't need absolute error, only relative error.

	timeList, stateList = rk4(initState, stateDerivative, errorFunct, t0, t1, initDT)

	'''
	Since we all know what e^-x looks like, lets just
	plot the difference between the exact answer
	and what our integrator gives to see how close we are.
	'''
	errorList = []
	for i in range(len(timeList)):
		errorList.append(stateList[i]-np.exp(-timeList[i]))

	plt.figure(1)
	plt.plot(timeList, errorList)
	plt.grid(True)
	plt.title('Accumulative Absolute Error\nof Exponential Integration')

	'''
	You'll notice that the above gives a dwindling value.
	Both the analytic and integrated curves limit to zero.
	So we should look at relative error instead.
	'''
	for i in range(len(timeList)):
		errorList[i] *= np.exp(timeList[i])

	plt.figure(2)
	plt.plot(timeList, errorList)
	plt.grid(True)
	plt.title('Accumulative Relative Error\nof Exponential Integration')
	plt.show()

def SHO_Example():
	'''
	Create a simple harmonic oscillator.
	Equation of motion is
	xdotdot + a * xdot + k/m * x = 0
	with 'a' being the dampening coefficient
	divided by m. (We will refer to this
	as 'dampeningTerm'.)

	Analytically, this should look like
	exp[(-a/2 +- i*sqrt[4k/m - a^2]/2)t].

	For our initial conditions, this is
	exp(-at/2)sin(t*sqrt[4k/m - a^2]/2)
	'''

	absTol = 1e-8
	relTol = 1e-6

	dampeningTerm = 0.0
	kOverM = 1

	initState = np.array([1., 0.])#xdot and x
	t0 = 0
	t1 = 20
	initDT = 0.03

	def stateDerivative(inputState, inputTime):
		#inputTime can be used for external force terms etc.
		xdot = inputState[0]
		x = inputState[1]
		xdotdot = -dampeningTerm * xdot - kOverM * x
		return np.array([xdotdot, xdot]) #derivative of xdot is xdotdot, derivative of x is xdot.

	def errorFunct(currentState, deltaState, derivative, dt):
		# return .9**5
		'''
		we assume the dimension of the state are independent, so to get the error we
		add the relative of each dimension in quadrature and take the square root.
		This will make it so that any tolerance you specify should be thought of as "one part in blank",
		so if you want accuracy to one part in a million, the tolerance will be 1e-6.

		Note: Numerical Recipes does not suggest using summing in quadrature for error.
		It suggests finding errors for each coordinate and then taking a max at the end.
		'''
		errorParameter = 0
		for pair in zip(currentState, deltaState):
			errorParameter += (pair[1]/(pair[0]*relTol + absTol))**2
		return np.sqrt(errorParameter)

	timeList, stateList = rk4(initState, stateDerivative, errorFunct, t0, t1, initDT)

	'''
	We only want the position, not the velocity. The following line
	grabs only the first element from every state. If this line doesn't make sense
	to you, then look up what these symbols do in python.
	'''
	xList = list(zip(*stateList))[1]
	plt.figure(1)
	plt.plot(timeList, xList)
	plt.title('Numerically Integrated\nSimple Harmonic Oscillator')
	plt.grid(True)

	#And we would like to see the difference as before
	errorList = []
	for i in range(len(timeList)):
		errorList.append(stateList[i][1]
			-np.sin(timeList[i]*np.sqrt(4*kOverM - dampeningTerm**2) / 2)
			*np.exp(-dampeningTerm * timeList[i] / 2))

	plt.figure(2)
	plt.plot(timeList, errorList)
	plt.grid(True)
	plt.title('Accumulative Absolute Error\nof SHO Integration')
	plt.show()

def BesselExample():
	'''
	Equation of motion:

	xdotdot + xdot/t + (1 - n^2/t^2)x = 0

	where n is the order.
	Notice that this involves dividing by the free parameter,
	so we can't start at t = 0.
	'''

	absTol = 1e-6
	relTol = 1e-6

	order = 0

	t0 = 0.001
	t1 = 500
	initDT = 0.01

	'''
	This problem benifits from an extra small initial step due to
	the division by the time parameter.
	(This is not necessary since the algorithm will refine the step
	size dramatically at the start, I just wanted to put it here.)

	We will use the analytic solution for the initial conditions,
	since we are not starting at t = 0. This can be
	partially avoided by using a small t approximation,
	but it won't be quite right and will give us
	errors which are not sourced from the integrator
	but rather from the initial conditions. So I get
	rid of it for simplicity and so we can focus on just
	the integrator itself.
	'''

	initState = np.array([
		0.5*(special.jv(order-1, t0) - special.jv(order+1, t0)),#xdot
		special.jv(order,t0)])#x

	def stateDerivative(inputState, inputTime):
		#inputTime can be used for external force terms etc.
		xdot = inputState[0]
		x = inputState[1]
		xdotdot = -xdot/inputTime + (order**2/inputTime**2-1)*x
		return np.array([xdotdot, xdot]) #derivative of xdot is xdotdot, derivative of x is xdot.

	def errorFunct(currentState, deltaState, derivative, dt):
		# return 1
		'''
		Here we are adding the error of each coordinate in quadrature.
		(The coordinates are x and xdot.) Each error is actually a ratio
		of two errors: the first is the error which the
		algorithm finds, deltaState, while the other is the
		maximum error our tolerances allow. If we are using
		a step-by-step tolerance, then this will
		be (y * relTol + absTol), as numpy/scipy uses. If we
		are using total simulation tolerance, then this will
		be (dy/dt * relTol + absTol) * (dt/(t1-t0)). This results
		in fractional tolerance on the increment error (dy) rather than
		the total error (y). This makes it so that if you integrate
		the error itself, you are gaurenteed to be within tolerance at
		the end.

		The commented code is using step-by-step tolerance,
		while the uncommented code uses total simulation tolerance.
		The total simulation tolerance bugs out if you try to use t1 > ~200
		with absTol = ~1e-8, probably due to machine precision limits.
		'''
		# errorParameter = 0
		# for pair in zip(currentState, deltaState):
		# 	errorParameter += (pair[1]/(pair[0]*relTol + absTol))**2
		# return np.sqrt(errorParameter)
		errorParameter = 0
		for pair in zip(deltaState, derivative):
			errorParameter += (pair[0]/((absTol + pair[1] * relTol)*dt/t1))**2
		return np.sqrt(errorParameter)

	timeList, stateList = rk4(initState, stateDerivative, errorFunct, t0, t1, initDT)

	errorList = []
	for i in range(len(timeList)):
		errorList.append(stateList[i][1]-special.jv(order, timeList[i]))

	plt.plot(timeList, errorList)
	plt.grid(True)
	plt.title('Accumulative Absolute Error of Bessel Integration\nwith Step-by-Step Error Tolerances')
	# plt.title('Accumulative Absolute Error of Bessel Integration\nwith Total Runtime Error Tolerances')
	plt.show()

def freeParticlePolarExample():
	'''
	We would like to integrate a particle moving in a
	straight line in polar coordinates.

	Metric:
	ds^2 = r^2 dphi^2 + dr^2

	From Euler-Lagrange:
	d/dt(2rDot)=2r*phiDot^2
	d/dt(r^2*2phiDot)=0
	=>
	rDotDot=r*phiDot^2 (centrifugal force)
	phiDotDot=-2rDot*phiDot/r (coriolis effect)

	We will keep track of the state with this scheme: [r,rDot, phi,phiDot].
	'''

	absTol = 1e-8
	relTol = 1e-6
	angularTol = 1e-8

	initState = np.array([1, 0, 0, 1])#start on x axis heading in y direction
	t0 = 0.
	t1 = 10
	initDT = 0.01

	def stateDerivative(inputState, inputTime):
		r = inputState[0]
		rDot = inputState[1]
		phi = inputState[2]
		phiDot = inputState[3]
		return np.array([rDot, r * phiDot**2, phiDot, -2*rDot*phiDot/r])

	def errorFunct(currentState, deltaState, derivative, dt):
		'''
		Everything except phi should be done with relative error,
		including phiDot. (Since phi is the only compact coordinate.)
		Phi should be handled with absolute error.
		'''
		errorParameter = 0

		for i in [0,1,3]:#skip phi
			errorParameter += (deltaState[i]/(currentState[i]*relTol + absTol))**2

		errorParameter += deltaState[2]**2*angularTol**2
		return np.sqrt(errorParameter)

	timeList, stateList = rk4(initState, stateDerivative, errorFunct, t0, t1, initDT)

	rList = list(zip(*stateList))[0]
	phiList = list(zip(*stateList))[2]

	plt.polar(phiList, rList)
	plt.ylim([0,12])
	plt.grid(True)
	plt.title('Numerically Integrated Path of a Free Particle in Polar Coordinates')
	plt.show()

	#This also has an analytic result we could compare against, but I'm
	#too lazy to do that part.

# exponentialExample()
# SHO_Example()
BesselExample()
# freeParticlePolarExample()
