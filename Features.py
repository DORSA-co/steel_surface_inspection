import cv2
import numpy as np
from skimage import feature
import os
import time

from tensorflow.python.ops.gen_array_ops import split


# ______________________________________________________________________________________________________________________________________________
# explain:
#   get histogram of Gradiant from gray image
#
# arg:
#   gray: gray image ( np.array shape=(w,h), dtype = np.uint8)
#
#   bin_n: number of bins in histogram
#
#   split: This parameter specifies that the image be divided into how many pieces
#   to calculate the histogram , for split = 3the image of divided into 9 pices (3x3)
#
# return:
#   hist
#   hist: histogram of color (np.array shape(bin_n,))
# ______________________________________________________________________________________________________________________________________________
def get_hog(bin_n=24, split_h=2, split_w=2):
    def extractor(gray):
        h, w = gray.shape
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)

        bin_cells = []
        mag_cells = []

        for i in range(split_h):
            for j in range(split_w):
                bin_cells.append(bins[h // split_h * i:h // split_h * (i + 1),
                                 w // split_w * j:w // split_w * (j + 1)])

                mag_cells.append(mag[h // split_h * i:h // split_h * (i + 1),
                                 w // split_w * j:w // split_w * (j + 1)])

        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)  # hist is a 64 bit vector
        # hist = hist.astype(np.float32) / ( (h//split) * (w//split)  )
        hist = hist - hist.mean()
        hist = hist / hist.std()
        return hist

    return extractor


# ______________________________________________________________________________________________________________________________________________
# explain:
#   get histogram of color from gray image
#
# arg:
#   gray: gray image ( np.array shape=(w,h), dtype = np.uint8)
#   bin_n: number of bins in histogram
#
# return:
#   hist
#   hist: histogram of color (np.array shape(bin_n,))
# ______________________________________________________________________________________________________________________________________________
def get_hoc(bin_n=25, split_h=2, split_w=2):
    def extractor(gray):
        h, w = gray.shape
        hists = []
        for i in range(split_h):
            for j in range(split_w):
                roi = gray[h // split_h * i:h // split_h * (i + 1), w // split_w * j:w // split_w * (j + 1)]

                hists.append(cv2.calcHist([roi], [0], None, [bin_n], [0, 255]).reshape(-1))
        hist = np.concatenate(hists)
        hist = hist - hist.mean()
        hist = hist / hist.std()
        return hist

    return extractor


# ______________________________________________________________________________________________________________________________________________
# explain:
#   get local binary pattern from gray image
#
# arg:
#   image: gray image ( np.array shape=(w,h), dtype = np.uint8)
#   P: Number of circularly symmetric neighbour set points (quantization of the angular space)
#	R: Radius of circle (spatial resolution of the operator)
# 	method: Method to determine the pattern.
#	‘default’: original local binary pattern which is gray scale but not rotation invariant.
#	‘ror’: extension of default implementation which is gray scale and rotation invariant.
#	‘uniform’: improved rotation invariance with uniform patterns and finer quantization of
##	the angular space which is gray scale and rotation invariant.
#	‘nri_uniform’: non rotation - invariant uniform patterns variant which is only gray scale
##	invariant.
#	‘var’: rotation invariant variance measures of the contrast of local image texture
##	which is rotation but not gray scale invariant.
#
# return:
#   hist
#   hist: histogram of LBP
# ______________________________________________________________________________________________________________________________________________
def get_lbp(P, R, method):
    def extractor(image):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, P, R, method)
        lbp = lbp.astype(int)
        hist = np.bincount(lbp.ravel(), minlength=lbp.max() + 1)
        hist = hist - hist.mean()
        hist = hist / hist.std()
        return hist

    return extractor


if __name__ == '__main__':
    img = cv2.imread('/home/reyhane/Desktop/f.jpg', 0)
    start = time.time()
    func = get_lbp(8, 1, 'default')
    h = func(img)
    print(time.time() - start)
    print(h, h.shape)
    pass