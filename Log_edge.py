import numpy as np
import cv2


def log_fspecial(p1=13, p2=2):
	# first calculate Gaussian
	siz = (p1 - 1) / 2
	std2 = np.power(p2, 2)
	x, y = np.meshgrid(np.arange(-siz, siz + 1, 1), np.arange(-siz, siz + 1, 1))
	arg = -(x * x + y * y) / (2 * std2)
	h = np.zeros_like(arg)
	h = np.exp(arg)
	eps = 2.2204e-16
	h[h < (eps * (np.max(h)))] = 0
	hsum = np.sum(h)
	if hsum != 0:
		h = h / hsum
	# now calculate Laplace
	h1 = h * (x * x + y * y - 2 * std2) / (np.power(std2, 2))
	h = h1 - np.sum(h1) / np.prod(np.array([p1, p1]))
	return h


def edge_log(I, sigma=2):
	I = I / 256.0
	m, n = I.shape
	Out = np.zeros_like(I)
	rr = np.arange(1, m)  # 1:m-1
	cc = np.arange(1, n)
	Out = cv2.filter2D(I, -1, log_fspecial())
	Out[Out < 0] = 0
	Out = Out * 256
	return Out
