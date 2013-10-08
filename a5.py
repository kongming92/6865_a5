# Assignment 5 for 6.815/865
# Submission:
# Deadline: 9/9
# Your name: Charles Liu
# Reminder:
# - Don't hand in data
# - Don't forget README.txt

import numpy as np
import scipy
import bilagrid

def computeWeight(im, epsilonMini=0.002, epsilonMaxi=0.99):
	out = np.ones(im.shape)
	out[im < epsilonMini] = 0.0
	out[im > epsilonMaxi] = 0.0
	return out

def computeFactor(im1, w1, im2, w2):
	valid = np.logical_and(w1, w2) # channels of pixels where both w1 and w2 are nonzero
	return np.median((im2[valid] / im1[valid]).flatten())

def makeHDR(imageList, epsilonMini=0.002, epsilonMaxi=0.99):
	out = np.zeros(imageList[0].shape)
	weightSum = np.zeros(imageList[0].shape)
	lastW = np.zeros(imageList[0].shape)
	lastK = 0
	for i, image in enumerate(imageList):
		# Compute w, with special cases
		# Set min or max to beyond [0,1] so it doesn't get clamped
		if i == 0:
			w_i = computeWeight(image, epsilonMini, 2.0)
		elif i == len(imageList) - 1:
			w_i = computeWeight(image, -1.0, epsilonMaxi)
		else:
			w_i = computeWeight(image, epsilonMini, epsilonMaxi)

		# Compute k, starting at 1
		if i == 0:
			k_i = 1
		else:
			k_i = computeFactor(imageList[i-1], lastW, image, w_i) * lastK
		lastW = w_i
		lastK = k_i
		weightSum += w_i
		out += w_i * image / k_i
	weightSum[weightSum == 0] = 1e-12
	return out / weightSum

def toneMap(im, targetBase=100, detailAmp=1, useBila=False):
	imL, imC = lumiChromi(im)
	imL[imL == 0] = np.min(imL[imL.nonzero()])
	logimL = np.log10(imL)
	domainSigma = max(im.shape[0], im.shape[1]) / 50.0
	if useBila:
		rangeSigma = 0.4
		largeScale = bilagrid.bilateral_grid(logimL, domainSigma, rangeSigma)
	else:
		largeScale = scipy.ndimage.filters.gaussian_filter(logimL, [domainSigma, domainSigma, 0])

	detail = logimL - largeScale
	k = np.log10(targetBase) / (np.max(largeScale) - np.min(largeScale))
	logOut = detailAmp * detail + k * (largeScale - np.max(largeScale))

 	return (10 ** logOut) * imC

def BW(im, weights=[0.3, 0.6, 0.1]):
    img = im.copy()
    (height, width, rgb) = np.shape(img)
    for y in xrange(height):
        for x in xrange(width):
            img[y][x] = np.dot(img[y][x], weights)
    return img

def lumiChromi(im):
    imcopy = im.copy()
    bw = BW(imcopy)
    bwNonzero = bw.copy()
    bwNonzero[bw == 0] = 1e-12
    return (bw, imcopy / bwNonzero)
