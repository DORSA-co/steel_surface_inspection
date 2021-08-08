import cv2
import numpy as np



#______________________________________________________________________________________________________________________________________________
#explain:
#   get histogram of Gradiant from gray image
#
#arg:
#   gray: gray image ( np.array shape=(w,h), dtype = np.uint8)
#
#   bin_n: number of bins in histogram
#
#   split: This parameter specifies that the image be divided into how many pieces to calculate the histogram , for split = 3
#   the image of divided into 9 pices (3x3)
#
#return:
#   hist
#   hist: histogram of color (np.array shape(bin_n,))
#______________________________________________________________________________________________________________________________________________
def get_hog(gray, bin_n = 24, split=2):
    h,w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi)) #quantizing binvalues in (0...16)

    bin_cells = []
    mag_cells = []
    
    for i in range(split):
        for j in range(split):
            bin_cells.append( bins[h//split*i:h//split*(i+1) ,
                                   w//split*j:w//split*(j+1)] )

            mag_cells.append( mag[h//split*i:h//split*(i+1) ,
                                   w//split*j:w//split*(j+1)] )
        
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists) #hist is a 64 bit vector
    return hist



#______________________________________________________________________________________________________________________________________________
#explain:
#   get histogram of color from gray image
#
#arg:
#   gray: gray image ( np.array shape=(w,h), dtype = np.uint8)
#   bin_n: number of bins in histogram
#
#return:
#   hist
#   hist: histogram of color (np.array shape(bin_n,))
#______________________________________________________________________________________________________________________________________________
def get_hoc(gray, bin_n = 25):
    hist = cv2.calcHist( [gray],[0], None, [24],[0,255]).reshape(-1)
    return hist