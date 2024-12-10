import numpy as np
import torch
from skimage import measure
from scipy.ndimage.filters import gaussian_filter, maximum_filter1d

from numba import jit, vectorize

from winding import winding

def defect_coordinates(theta, thresh=0.2, radius=3, periodic=False):
	charge = winding(theta, radius=radius, periodic=periodic)
	charge[np.abs(charge) < thresh] = 0
	return prominent_peaks(gaussian_filter(np.abs(charge), 1), min_xdistance=2, min_ydistance=2)

def prominent_peaks(image, min_xdistance, min_ydistance, threshold=None):
	img = image.copy()
	rows, cols = img.shape
	if threshold is None:
		threshold = 0.5 * np.max(img)
	yc_size = 2 * min_ydistance + 1
	xc_size = 2 * min_xdistance + 1
	img_max = maximum_filter1d(img, size=yc_size, axis=0, mode='constant', cval=0)
	img_max = maximum_filter1d(img_max, size=xc_size, axis=1, mode='constant', cval=0)
	mask = (img == img_max)
	img *= mask
	img_t = img > threshold

	label_img = measure.label(img_t)
	props = measure.regionprops(label_img, img_max)
	props = sorted(props, key=lambda x: x.max_intensity)[::-1]
	coords = np.array([np.round(p.centroid) for p in props], dtype=int)
	yc_peaks = []
	xc_peaks = []

	yc_ext, xc_ext = np.mgrid[-min_ydistance:min_ydistance+1,
							  -min_xdistance:min_xdistance+1]
	for yc_idx, xc_idx in coords:
		accum = img_max[yc_idx, xc_idx]
		if accum > threshold:
			yc_nh = yc_idx + yc_ext
			xc_nh = xc_idx + xc_ext

			yc_in = np.logical_and(yc_nh > 0, yc_nh < rows)
			yc_nh = yc_nh[yc_in]
			xc_nh = xc_nh[yc_in]

			xc_low = xc_nh < 0
			yc_nh[xc_low] = rows - yc_nh[xc_low]
			xc_nh[xc_low] += cols
			xc_high = xc_nh >= cols
			yc_nh[xc_high] = rows - yc_nh[xc_high]
			xc_nh[xc_high] -= cols

			img_max[yc_nh, xc_nh] = 0
			yc_peaks.append(yc_idx)
			xc_peaks.append(xc_idx)

	return np.transpose(np.vstack((np.array(xc_peaks), np.array(yc_peaks)))).astype(int)

if __name__=='__main__':
	index = (slice(0, 24), slice(0, 24))
	index = (slice(0, 200), slice(0, 200))
	charges = np.loadtxt('../sharpening/images/ch700')[index]
	nx = np.loadtxt('../sharpening/images/nx700')[index]
	ny = np.loadtxt('../sharpening/images/ny700')[index]

	theta = np.arctan2(ny, nx)
	theta[theta < 0] += np.pi
	
	import matplotlib.pyplot as plt
	ax = plt.figure().gca()
	charge = winding(theta)
	defects = defect_coordinates(theta)
	charges = charge[defects[:, 1], defects[:, 0]]
	colors = [[1, 0, 0] if abs(c+0.5) < 0.1 else [0, 0.5, 1] for c in charges]
	ax.imshow(theta)
	ax.scatter(defects[:, 0], defects[:, 1], c=colors)
	plt.show()
