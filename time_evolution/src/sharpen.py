from scipy.ndimage.filters import gaussian_filter, maximum_filter1d
from skimage import measure
from ctypes import c_int
import numpy.ctypeslib as npct
import numpy as np
import torch

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')

'''
Loading of C libraries
'''

libsharpen = npct.load_library('libsharpen', '../src')
libsharpen.sharpen.restype = None
libsharpen.sharpen.argtypes = [
    array_1d_double,	#theta
    array_1d_int,		#coords
    c_int,				#ncoords
    c_int,				#x_max
    c_int,				#y_max
    c_int,				#fixed_border
    c_int				#n
]
libsharpen.sharpen_periodic.restype = None
libsharpen.sharpen_periodic.argtypes = [
    array_1d_double,	#theta
    array_1d_int,		#coords
    c_int,				#ncoords
    c_int,				#x_max
    c_int,				#y_max
    c_int,				#fixed_border
    c_int				#n
]


libwinding = npct.load_library('libwinding', '../src')
libwinding.winding.restypes = None
libwinding.winding.argtypes = [
    array_1d_double,	#theta
    c_int,				#x_max
    c_int,				#y_max
    c_int				#box_size
]

'''
Methods for sharpening ML model predictions
'''
def convert_sincos_theta(sincos):
    sincos[sincos < -1] = -1
    sincos[sincos > 1] = 1
    twot = np.arctan2(sincos[0], sincos[1])
    theta = twot / 2
    theta[theta < 0] += np.pi
    return theta

def defect_coordinates(theta, thresh=0.2):
    charge = theta.flatten().astype(np.float64)
    libwinding.winding(charge, theta.shape[0], theta.shape[1], 2)
    charge = charge.reshape(theta.shape)
    charge[np.abs(charge) < thresh] = 0
    return prominent_peaks(gaussian_filter(np.abs(charge), 1), min_xdistance=2, min_ydistance=2)

def sharpen(theta, border=3, n=1, periodic=False):
    coords = defect_coordinates(theta)
    sharp = theta.flatten().astype(np.float64)
    if periodic:
        libsharpen.sharpen(sharp, coords.flatten().astype(np.int32), coords.shape[0],
            theta.shape[0], theta.shape[1], border, n)
    else:
        libsharpen.sharpen(sharp, coords.flatten().astype(np.int32), coords.shape[0],
            theta.shape[0], theta.shape[1], border, n)
    return sharp.reshape(theta.shape)

'''
The following method is copied from the publically-accessible scikit-image source code (skimage.feature.peak.py, method _prominent_peaks). The method is used for converting winding masks to defect coordinates. For some unknown reason, this routine was not importable despite appearing in their public source code at the time of writing this program (this may change in the future). As this is then redistribution of scikit-image source code, we include the following in accordance with the scikit-image license:

Copyright (C) 2019, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

'''
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

    return np.transpose(np.vstack((np.array(xc_peaks), np.array(yc_peaks))))

'''
Methods for stitching forecasting patches into a full frame and looping predictions
'''

def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size//2:size//2, -size//2:size//2]
    xy2 = np.power(x, 2) + np.power(y, 2)
    return np.exp(xy2 / (2 * sigma * sigma))

def stitch_frame_prediction(model, sequence, device, input_size=48, stitch_ratio=0.2):
    overlap = int(input_size * stitch_ratio)
    inc = input_size - overlap

    shape = list(sequence.shape)
    shape[-4] = 1 #Forward compatibility
    #del shape[-4]
    pred = torch.zeros(shape, device=device)
    counts = torch.zeros(shape, device=device)

    kernel = torch.tensor(gaussian_kernel(input_size, input_size)).float().to(device)

    y0, y1 = 0, input_size
    while y1 < sequence.shape[-1]:
        x0, x1 = 0, input_size
        while x1 < sequence.shape[-2]:
            counts[..., x0:x1, y0:y1] += kernel
            pred[..., x0:x1, y0:y1] += model(sequence[..., x0:x1, y0:y1]) * kernel
            x0, x1 = x0 + inc, x1 + inc
        counts[..., -input_size:, y0:y1] += kernel
        pred[..., -input_size:, y0:y1] += model(sequence[..., -input_size:, y0:y1]) * kernel
        y0, y1 = y0 + inc, y1 + inc

    x0, x1 = 0, input_size
    while x1 < sequence.shape[-2]:
        counts[..., x0:x1, -input_size:] += kernel
        pred[..., x0:x1, -input_size:] += model(sequence[..., x0:x1, -input_size:]) * kernel
        x0, x1 = x0 + inc, x1 + inc
    counts[..., -input_size:, -input_size:] += kernel
    pred[..., -input_size:, -input_size:] += \
        model(sequence[..., -input_size:, -input_size:]) * kernel

    return pred / counts

def loop_frame_prediction(model, sequence, device, input_size=48, n=1, 
                          stitch_ratio=0.05, border=3, nsharp=1):
    outputs = np.zeros((n,)+ sequence.shape[-2:])
    full = stitch_ratio >= 0
    sequence = sequence.to(device)
    for i in range(n):
        if full:
            nextframe = stitch_frame_prediction(model, sequence.unsqueeze(0), 
                device, input_size, stitch_ratio)[0,0] #Forward compatibility
        else:
            nextframe = model(sequence.unsqueeze(0))[0,0] #Forward compatibility

        theta = convert_sincos_theta(nextframe.cpu().numpy())
        sharp = sharpen(theta, border=border, n=nsharp, periodic=full)
        sharp[np.isnan(sharp)] = theta[np.isnan(sharp)]
        theta = sharp

        outputs[i] = theta
        sequence[:-1] = sequence[1:].clone()
        sequence[-1] = torch.tensor(np.array([np.sin(2*theta), np.cos(2*theta)]), device=device).float()

    return outputs
