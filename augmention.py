from typing import MutableSequence
import cv2
import numpy as np
from numpy.core.fromnumeric import resize
from numpy.lib.twodim_base import tri




class augmention():

    def __int__(self):
        pass
    
    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   Rotate an image 
    #atribiut:
    #   imgs: iist of image
    #   angle: rotation angle
    #return:
    #   reses: list of rotated images
    #______________________________________________________________________________________________________________________________________________
    def rotate(self, imgs, rotate_agnle):
        reses = []
        for img in imgs:
            center = img.shape[0]//2,  img.shape[1] //2
            mtx = cv2.getRotationMatrix2D( center, rotate_agnle, 1 )

            h,w = img.shape[:2]
            cos = np.abs(mtx[0, 0])
            sin = np.abs(mtx[0, 1])
            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            # adjust the rotation matrix to take into account translation
            mtx[0, 2] += (nW / 2) - center[0]
            mtx[1, 2] += (nH / 2) - center[1]
            # perform the actual rotatio

            reses.append(cv2.warpAffine(img, mtx, (nW,nH)))
        return reses
    
    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   Rotate an image such a way that entire image keep
    #atribiut:
    #   imgs: iist of image
    #   angle: rotation angle deg
    #   keep_size: if True, the rotated image resize into orginal size
    #return:
    #   reses: list of rotated images
    #______________________________________________________________________________________________________________________________________________
    def rotate_bound(self,imgs, angle, keep_size = False):
        # grab the dimensions of the image and then determine the
        # center
        reses = []
        for img in imgs:
            (h, w) = img.shape[:2]
            (cx, cy) = (w // 2, h // 2)

            mtx = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
            cos , sin= np.abs(mtx[0, 0]) , np.abs(mtx[0, 1])
            # compute the new bounding dimensions of the image
            res_w , res_h = int((h * sin) + (w * cos)) , int((h * cos) + (w * sin))
            # adjust the rotation matrix to take into account translation
            mtx[0, 2] += (res_w / 2) - cx
            mtx[1, 2] += (res_h / 2) - cy
            # perform the actual rotation and return the image
            res = cv2.warpAffine(img, mtx, (res_w, res_h)  )
            if keep_size:
                res = cv2.resize( res, dsize=(w,h))
            reses.append( res )
        return reses

    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   transpose image in x and y axis
    #atribiut:
    #   imgs: iist of image
    #   tx: shift value for x axis in pixel
    #   ty: shift value for y axis in pixel
    #return:
    #   reses: list of shifted images
    #______________________________________________________________________________________________________________________________________________
    def shift(self, imgs, tx, ty):
        mtx =np.array([[1,0,tx],[0,1,ty]], dtype=np.float32)
        reses = []
        for img in imgs:
            reses.append( cv2.warpAffine(img, mtx, None) )
        return reses

    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   changed brightness of images
    #atribiut:
    #   imgs: iist of image
    #   leved: value of brighness ( negetive for darker , posetive for lighter) #recomend( -30< value < +30)
    #return:
    #   reses: list of modified images
    #______________________________________________________________________________________________________________________________________________
    def brightnessControl(self,imgs, level):
        reses = []
        for img in imgs:
            reses.append( cv2.convertScaleAbs(img, beta=level) )
        return reses


    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   verticaly flip a list of images 
    #atribiut:
    #   imgs: iist of image
    #return:
    #   reses: list of filped images
    #______________________________________________________________________________________________________________________________________________
    def hflip(self,imgs):
        reses = []
        for img in imgs:
            reses.append( cv2.flip(img,0) )
        return reses

    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   horizontali flip a list of images 
    #atribiut:
    #   imgs: iist of image
    #return:
    #   reses: list of filped images
    #______________________________________________________________________________________________________________________________________________
    def wflip(self,imgs):
        reses = []
        for img in imgs:
            reses.append( cv2.flip(img,1) )
        return reses


    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   hshear list of images 
    #atribiut:
    #   imgs: iist of image
    #   value: value of shearning ( recommend -2<shear<2)
    #return:
    #   reses: list of sheared images
    #______________________________________________________________________________________________________________________________________________
    def shear( self, imgs, value=0):
        reses = []
        for img in imgs:
            h, w = img.shape[0:2]
            mtx = np.float32([[1, 0, 0], [value, 1, 0]])
            mtx[0,2] = -mtx[0,1] * w/2
            mtx[1,2] = -mtx[1,0] * h/2
            
            reses.append( cv2.warpAffine(img, mtx, None) )
        return reses

    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   zoomout or zoomin an y and x axis 
    #atribiut:
    #   imgs: iist of image
    #   zoom_x: zoom in x axis ( recommend 0.7<shear<1.2)
    #   zoom_y: zoom in y axis ( recommend 0.7<shear<1.2)
    #return:
    #   reses: list of zoomed images
    #______________________________________________________________________________________________________________________________________________
    def zoom(self, imgs, zoom_x, zoom_y):
        reses = []
        for img in imgs:
            res = np.zeros_like(img )
            img = cv2.resize(img, None, fx=zoom_x, fy=zoom_y)

            h,w = img.shape[:2]
            h_res, w_res = res.shape[:2]
            sy , sx = abs(h-h_res)//2 , abs(w-w_res)//2

            if h>= h_res and w>=w_res:
                res =  img[sy:sy+h_res, sx:sx+w_res ]
            
            elif h>= h_res and w < w_res:
                res[:,sx:sx+w ] =  img[sy:sy+h_res, :]

            elif h< h_res and w >= w_res:
                res[sy:sy+h, : ] =  img[ :, sx:sx+w_res]
            
            elif h< h_res and w < w_res:
                res[sy:sy+h, sx:sx+w] =  img[ :, :]    
            reses.append(res)
                
        return reses





aug  = augmention()

#for i in range(20):
#    print(i/10)
img = cv2.imread('0a2c9f2e5.jpg')
mask = cv2.imread('m0a2c9f2e5.jpg',0)

#img , mask = aug.rotate_bound([mask,img], -10)
#img , mask = aug.shift([mask,img], 1500, -100)
#img , mask = aug.hflip([mask,img] )
#img , mask = aug.shear([mask,img], -i/10   )
#img , mask = aug.zoomout( [mask,img], zoom_x=.8, zoom_y=0.9)
img , mask = aug.zoom( [mask,img], zoom_x=.5, zoom_y=0.5 )

cv2.imshow('img',img)   
cv2.imshow('org',mask)
cv2.waitKey(0)