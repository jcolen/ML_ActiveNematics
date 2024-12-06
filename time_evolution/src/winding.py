import numpy as np
import torch

from numba import jit, vectorize

PI = np.pi
PIHALF = PI / 2.
PI2 = np.pi * 2.

@vectorize
def subtract_theta(theta1, theta2):
	dth = theta1 - theta2
	if dth < -PIHALF:
		dth += PI
	if dth > PIHALF:
		dth -= PI
	
	return dth

@jit(cache=True, nopython=True)
def gradient_y(theta, periodic=False):
	dth = np.empty_like(theta)
	dth[..., :-1, :] = subtract_theta(theta[..., 1:, :], theta[..., :-1, :])

	if periodic:
		dth[..., -1, :] = subtract_theta(theta[..., 0, :], theta[..., -1, :])

	else:
		dth[..., -1, :] = subtract_theta(theta[..., -1, :], theta[..., -2, :])
	
	return dth

@jit(cache=True, nopython=True)
def gradient_x(theta, periodic=False):
	dth = np.empty_like(theta)
	dth[..., :-1] = subtract_theta(theta[..., 1:], theta[..., :-1])
	
	if periodic:
		dth[..., -1] = subtract_theta(theta[..., 0], theta[..., -1])

	else:
		dth[..., -1] = subtract_theta(theta[..., -1], theta[..., -2])
	
	return dth

'''
Compute the winding number around each pixel in an image
'''
@jit(cache=True, nopython=True)
def winding(theta, radius=3, periodic=False):
	#Compute gradient in x, y direction
	gx = gradient_x(theta, periodic)
	gy = gradient_y(theta, periodic)
	
	#Compute winding around each point
	wind = np.zeros_like(theta)
	rad2 = 2 * radius
	for i in range(rad2):
		bnd = rad2-i
		wind[..., radius:-radius, radius:-radius] += gx[..., rad2:, i:-bnd]
		wind[..., radius:-radius, radius:-radius] -= gx[..., :-rad2, i:-bnd]
		wind[..., radius:-radius, radius:-radius] -= gy[..., i:-bnd, rad2:]
		wind[..., radius:-radius, radius:-radius] += gy[..., i:-bnd, :-rad2]
	wind /= PI2
	return wind

if __name__=='__main__':
	#index = (slice(0, 24), slice(0, 24))
	index = (slice(0, 200), slice(0, 200))
	charges = np.loadtxt('../sharpening/images/ch700')[index]
	nx = np.loadtxt('../sharpening/images/nx700')[index]
	ny = np.loadtxt('../sharpening/images/ny700')[index]

	theta = np.arctan2(ny, nx)
	theta[theta < 0] += np.pi
	wind = winding(theta, periodic=True)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(charges)
	ax[1].imshow(wind)
	plt.show()


