from typing import MutableSequence
import cv2
import numpy as np
from numpy.core.fromnumeric import resize
from numpy.lib.function_base import copy
from numpy.lib.twodim_base import tri
from numpy.random.mtrand import random_integers
from scipy.interpolate import UnivariateSpline



class augmention():

    def __int__(self, shift_range=(-100, 100),
                    rotation_range=(-10,10),
                    zoom_range=(0.9,1.1),
                    shear_range=(-0.1,0.1),
                    hflip=True, 
                    wflip = True, 
                    color_filter=True,
                    chance=0.3  ):

        self.shift_range = shift_range
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.hflip = hflip
        self.wflip = wflip
        self.color_filter = color_filter
        self.chance = chance
    
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
            reses.append(cv2.warpAffine(img, mtx, None))
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
    #   value: value of shearning ( recommend -0.2<shear<0.2)
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

    

    
    def __spreadLookupTable__(self,x, y):
        print(x)
        print(y)
        print('---------------------')
        spline = UnivariateSpline(x, y)
        return spline(range(256))
        
    
    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   Apply a random color filter( color effect ) on a list of images
    #atribiut:
    #   imgs: iist of image
    #   max_shift: maximum value for shift color
    #return:
    #   reses: result images
    #______________________________________________________________________________________________________________________________________________
    def color_filter(self, imgs, max_shift=30):

        reses = []
        for img in imgs:
            res = np.zeros_like(img)
            if len(img.shape) == 2:
                origin = np.linspace(0, 255 , 4)
                destnation = origin + np.random.randint(0,max_shift, len(origin)) * np.random.choice([1,-1])
                destnation = np.clip(destnation, 0,255)
                destnation[0], destnation[-1] = 0,255
                lookup_tabel = self.__spreadLookupTable__(origin, destnation)
                res = cv2.LUT( img, lookup_tabel).astype(np.uint8)
                
            else:
                for i in range(3):
                    origin =  list(np.linspace(0, 255 , 4 ))
                    destnation = list(origin + np.random.randint(0,max_shift, len(origin)) * np.random.choice([1,-1]))
                    destnation[0], destnation[-1] = 0,255
                    destnation = np.clip(destnation, 0,255)
                    lookup_tabel = self.__spreadLookupTable__(origin, destnation)
                    res[:,:,i] = cv2.LUT( img[:,:,i],lookup_tabel ).astype(np.uint8)

            reses.append(res)
        return reses


    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   augment an image randomly
    #atribiut:
    #   img: image
    #   shift_range:  acceptable range for tansform
    #   rotation_range: acceptable range for rotation
    #   zoom_range:  acceptable range for zoom
    #   shear_range:  acceptable range for shear
    #   hflip:  if True, hflip may occure
    #   wflip:  if True, wflip may occure
    #   color_filter:  if True, color_filter may occure
    #   chance : possible of augment
    #return:
    #   reses: result images
    #______________________________________________________________________________________________________________________________________________
    def augment_single(self,img ):
        
        func_chance = 0.5
        imgs = [img ] 
        if np.random.rand() < self.chance or True:
            if np.random.rand() < func_chance and self.shift_range is not None:
                tx = np.random.randint(self.shift_range[0], self.shift_range[1])
                ty = np.random.randint(self.shift_range[0], self.shift_range[1])
                imgs = self.shift(imgs, tx, ty)
                #print('shift', tx,ty)
                    

            if np.random.rand() < func_chance and self.zoom_range is not None:
                zoom_x = np.random.random() * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
                zoom_y = np.random.random() * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
                imgs = self.zoom(imgs, zoom_x, zoom_y)
                #print('zoom', zoom_x, zoom_y)

            if np.random.rand() < func_chance and self.rotation_range is not None:
                angle = np.random.random() * (self.rotation_range[1] - self.rotation_range[0]) + self.rotation_range[0]
                imgs = self.rotate(imgs, angle)
                #print('rotate', angle)

            if np.random.rand() < func_chance and self.shift_range is not None:
                value = np.random.random() * (self.shear_range[1] - self.shear_range[0]) + self.shear_range[0]
                imgs = self.shear(imgs, value)
                #print('shear', value)

            if np.random.rand() < func_chance and self.wflip:
                imgs = self.wflip(imgs)
                #print('wfilp')

            if np.random.rand() < func_chance and self.hflip:
                imgs = self.hflip(imgs)
                #print('hfilp')

            if np.random.rand() < func_chance and self.color_filter:
                imgs = self.color_filter(imgs)
                #print('color effect')

            #print('------------------augment-------------------')
        return imgs[0]

    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   augment an image and it's mask randomly
    #atribiut:
    #   img: image
    #   mask: mask label
    #   shift_range:  acceptable range for tansform
    #   rotation_range: acceptable range for rotation
    #   zoom_range:  acceptable range for zoom
    #   shear_range:  acceptable range for shear
    #   hflip:  if True, hflip may occure
    #   wflip:  if True, wflip may occure
    #   color_filter:  if True, color_filter may occure
    #   chance : possible of augment
    #return:
    #   img, mask
    #   img: result image
    #   mask: result mask
    #______________________________________________________________________________________________________________________________________________
    def augment_single_byMask(self,img, mask):
        
        func_chance = 0.5
        
        if np.random.rand() < self.chance or True:
            if np.random.rand() < func_chance and self.shift_range is not None:
                tx = np.random.randint(self.shift_range[0], self.shift_range[1])
                ty = np.random.randint(self.shift_range[0], self.shift_range[1])
                [img, mask ]  = self.shift([img, mask ] , tx, ty)
                #print('shift', tx,ty)
                    

            if np.random.rand() < func_chance and self.zoom_range is not None:
                zoom_x = np.random.random() * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
                zoom_y = np.random.random() * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
                [img, mask ]  = self.zoom([img, mask ] , zoom_x, zoom_y)
                #print('zoom', zoom_x, zoom_y)

            if np.random.rand() < func_chance and self.rotation_range is not None:
                angle = np.random.random() * (self.rotation_range[1] - self.rotation_range[0]) + self.rotation_range[0]
                [img, mask ]  = self.rotate([img, mask ] , angle)
                #print('rotate', angle)

            if np.random.rand() < func_chance and self.shift_range is not None:
                value = np.random.random() * (self.shear_range[1] - self.shear_range[0]) + self.shear_range[0]
                [img, mask ]  = self.shear([img, mask ] , value)
                #print('shear', value)

            if np.random.rand() < func_chance and self.wflip:
                [img, mask ]  = self.wflip([img, mask ] )
                #print('wfilp')

            if np.random.rand() < func_chance and self.hflip:
                [img, mask ]  = self.hflip([img, mask ] )
                #print('hfilp')

            if np.random.rand() < func_chance and self.color_filter:
                [img] = self.color_filter([img])
                #print('color effect')
            #print('------------------augment-------------------')
            _,mask = cv2.threshold(mask,180, 255, cv2.THRESH_BINARY)
        return img, mask


    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   augment a list of images randomly
    #atribiut:
    #   img: image
    #   shift_range:  acceptable range for tansform
    #   rotation_range: acceptable range for rotation
    #   zoom_range:  acceptable range for zoom
    #   shear_range:  acceptable range for shear
    #   hflip:  if True, hflip may occure
    #   wflip:  if True, wflip may occure
    #   color_filter:  if True, color_filter may occure
    #   chance : possible of augment
    #return:
    #   reses: result images
    #______________________________________________________________________________________________________________________________________________
    def augment_batch(self,imgs ):
        
        func_chance = 0.5
        reses=copy(imgs)
        if np.random.rand() < self.chance or True:
            if np.random.rand() < func_chance and self.shift_range is not None:
                tx = np.random.randint(self.shift_range[0], self.shift_range[1])
                ty = np.random.randint(self.shift_range[0], self.shift_range[1])
                reses = self.shift(reses, tx, ty)
                #print('shift', tx,ty)
                    

            if np.random.rand() < func_chance and self.zoom_range is not None:
                zoom_x = np.random.random() * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
                zoom_y = np.random.random() * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
                reses = self.zoom(reses, zoom_x, zoom_y)
                #print('zoom', zoom_x, zoom_y)

            if np.random.rand() < func_chance and self.rotation_range is not None:
                angle = np.random.random() * (self.rotation_range[1] - self.rotation_range[0]) + self.rotation_range[0]
                reses = self.rotate(reses, angle)
                #print('rotate', angle)

            if np.random.rand() < func_chance and self.shift_range is not None:
                value = np.random.random() * (self.shear_range[1] - self.shear_range[0]) + self.shear_range[0]
                reses = self.shear(reses, value)
                #print('shear', value)

            if np.random.rand() < func_chance and self.wflip:
                reses = self.wflip(reses)
                #print('wfilp')

            if np.random.rand() < func_chance and self.hflip:
                reses = self.hflip(reses)
                #print('hfilp')

            if np.random.rand() < func_chance and self.color_filter:
                reses = self.color_filter(reses)
                #print('color effect')

            #print('------------------augment-------------------')
        return reses
    






if __name__ == "__main__":
    aug  = augmention()

    for i in range(500):
        print(i/10)
        img = cv2.imread('0a2c9f2e5.jpg')
        mask = cv2.imread('m0a2c9f2e5.jpg',0)

        #img , mask = aug.rotate([mask,img], -10)
        #img , mask = aug.shift([mask,img], 1500, -100)
        #img , mask = aug.hflip([mask,img] )
        #img , mask = aug.shear([mask,img], -i/10   )
        #img , mask = aug.zoomout( [mask,img], zoom_x=.8, zoom_y=0.9)
        #img , mask = aug.zoom( [mask,img], zoom_x=.5, zoom_y=0.5 )
        #img , mask = aug.color_fliter( [mask,img] )
        img,mask = aug.augment_single_byMask(img, mask)
        
        cv2.imshow('img',img)   
        cv2.imshow('org',mask)
        cv2.waitKey(0)