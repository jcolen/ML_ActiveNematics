import os
import numpy as np
import torch
import argparse

from torchvision import transforms
from simulation_datasets import NematicsSequenceDataset
import data_processing as dp
import sharpen
from models import FramePredictor

def dot_product(theta1, theta2):
	return np.average(
		np.abs(np.cos(theta1) * np.cos(theta2) + np.sin(theta1) * np.sin(theta2)))

def spatial_correlation_qij(theta, d=1.):
	nx = np.cos(theta)
	ny = np.sin(theta)

	qxx = nx * nx
	qyy = ny * ny
	qxy = nx * ny

	Ftxx = np.fft.fft2(qxx)
	Ftyy = np.fft.fft2(qyy)
	Ftxy = np.fft.fft2(qxy)

	prodxx = Ftxx * np.conjugate(Ftxx)
	prodyy = Ftyy * np.conjugate(Ftyy)
	prodxy = Ftxy * np.conjugate(Ftxy)

	tp = (np.fft.ifft2(prodxx + prodyy + 2 * prodxy).real / (theta.shape[0] * theta.shape[1]) - 0.5) * 2
	
	return vec_to_scalar_func(tp, d=d)[:int(theta.shape[0]/2), 1]

def mean_defect_spacing(theta):
	defects = sharpen.defect_coordinates(theta)
	return defects.shape[0]

def compare(t1, t2, func, l05=True, norm=False, out=False):
	f1 = func(t1)
	f2 = func(t2)

	outstr = ''
	if l05:
		try:
			outstr += '%d\t' % np.argwhere(f1 < 0.5)[0][0]
		except:
			outstr += '-1\t'
		try:
			outstr += '%d\t' % np.argwhere(f2 < 0.5)[0][0]
		except:
			outstr += '-1\t'
	else:
		try:
			outstr += '%d\t' % f1
			outstr += '%d\t' % f2
		except:
			outstr += '-1\t-1\t'
	return outstr	

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--directory', type=str, default='/home/jcolen/data/deltat_10s')
	parser.add_argument('--num_frames', type=int, default=7)
	parser.add_argument('--nmax', type=int, default=50)
	parser.add_argument('--niters', type=int, default=300)
	parser.add_argument('--stitch_ratio', type=float, default=0.1)
	parser.add_argument('--border', type=int, default=3)
	parser.add_argument('--nsharp', type=int, default=1)
	args = parser.parse_args()

	#GPU or CPU
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pin_memory = True
	print(device)

	#Load dataset
	transform = transforms.Compose([
		dp.SinCos(),
		dp.ToTensor()
	])
	dataset = NematicsSequenceDataset(args.directory, args.num_frames+args.nmax, transform=transform, 
		validation_split=1.0)
	loader = dataset.get_loader(dataset.test_indices, 1, 2, True)
	sampler = iter(loader.batch_sampler)

	input_size = 48
	model = FramePredictor([2,4,6],
						   recurrent='residual',
						   #recurrent='linear_res',
						   input_size=input_size,
						   num_lstms=2)
	model.load_state_dict(torch.load('models/%s' % model.name)['state_dict'])
	model.eval().to(device)

	'''
	Comparing metrics
	1. Pixelwise dot product
	2. Spatial correlation function
	3. Defect density
	'''

	with torch.no_grad():
		for x in range(args.niters):
			indices = next(sampler)
			samples = loader.collate_fn([dataset[i] for i in indices[:1]])
			images = samples['image']
			if args.stitch_ratio < 0:
				images = images[..., :input_size, :input_size]

			sequence = images[0, :args.num_frames]
			act_sc = images[0, args.num_frames:].numpy()

			outstr = '%g\t%d\t' % (samples['label'][0, -1], indices[0])

			pre = sharpen.loop_frame_prediction(model, sequence, device, input_size,
				n=args.nmax, stitch_ratio=args.stitch_ratio, border=args.border, nsharp=args.nsharp)
			#Convert actuals to theta
			act = np.zeros(act_sc.shape[:1] + act_sc.shape[-2:])
			for i in range(act.shape[0]):
				act[i] = sharpen.convert_sincos_theta(act_sc[i])

			for i in range(pre.shape[0]):
				dot = dot_product(act[i], pre[i])
				outstr = '\t%g\t' % (dot)
				outstr += compare(act[i], pre[i], spatial_correlation_qij)
				outstr += compare(act[i], pre[i], mean_defect_spacing, l05=False)
				print(outstr, flush=True)
