import readFile
import cv2
from skimage.feature import local_binary_pattern
import labels
import numpy as np

# folder="D:\data-1"
# data=labels.test_image
# images=readFile.resize_images(folder,data)
# gray_images=readFile.grayscale_images(folder,data)
lbp_radius=3
lbp_numpoints=8*lbp_radius
eps=1e-7

def calculate_lbp(gray_images):
	lbp_features=[]
	for im in gray_images:
		lbp = local_binary_pattern(im, lbp_numpoints, lbp_radius, method='uniform')
		(hist,hist_len)=np.histogram(lbp.ravel(),bins=np.arange(0,lbp_numpoints + 3), range=(0, lbp_numpoints + 2))

		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		lbp_features.append(hist)
	return lbp_features

def calculate_hog(images):
	winSize = (128,64)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	winStride = (8,8)
	padding = (8,8)
	hog_features=[]
	for im in images:
		hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
		hist=hog.compute(im)
		hog_features.append(hist)
	return hog_features

# lbp_feat=calculate_lbp()
# print lbp_feat

# hog_feat=calculate_hog()
# print hog_feat[0]




