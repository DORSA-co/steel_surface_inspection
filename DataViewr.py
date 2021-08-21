from statistics import binary_hist
import cv2
from matplotlib import pyplot as plt
import sys

from numpy.lib.function_base import percentile
import DataReader
import numpy as np


#______________________________________________________________________________________________________________________________________________
#explain:
#   show Dataset. Just instance an object and it will show images and labels
#
#arg:
#   gen: generator function ( from DataReader.py)
#
#return:
#   plot Dataset  automaticly in proper view
#______________________________________________________________________________________________________________________________________________
class Viewer():

    def __init__(self, gen):
        self.gen = gen
        self.batch_img = None
        self.batch_lbl = None
        self.batch_size = None
        self.idx = -1
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        plt.show()
    

    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   keabord press Event. DONT NEED CALL THIS FUNCTION
    #
    #arg:
    #   event: it obtain when pressed a key on keyboard
    #
    #______________________________________________________________________________________________________________________________________________
    def on_press(self, event):
        print('press', event.key)
        
        sys.stdout.flush()
        if event.key =='escape':
            plt.close()
            return None
        
        img,lbl = None, None
        if event.key == 'left':
            img, lbl = self.sample('back')

        elif event.key == 'right':
            img, lbl = self.sample('next')
        
        if lbl is not None and len(lbl.shape)==0: #binary Labe shape is ()
            self.binary(img,lbl)
        
        elif lbl is not None and len(lbl.shape)==1: #class Label shape is (n,)
            self.classification(img,lbl)

        elif lbl is not None and len(lbl.shape)>1: #class Label shape is (n, h,w)
            self.mask(img,lbl)
            
            
        
        

    
    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   function for plot an image and binary label
    #
    #arg:
    #   img: numpy array of image in BGR format
    #   lbl: binary label that is a numpy number ( 0 or 1)
    #______________________________________________________________________________________________________________________________________________
    def binary(self, img, lbl):
        H = 50
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        lbl_img = np.zeros((H,w,3), dtype=np.uint8)
        

        if lbl==1:
            lbl_img[:,:,1] = 255
            cv2.putText( lbl_img, "Have Object", ( int(w*0.44), H//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0))
        else:
            lbl_img[:,:,2] = 255
            cv2.putText( lbl_img, "No Object", ( int(w*0.45), H//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255))
        
        lbl_img = cv2.cvtColor(lbl_img, cv2.COLOR_BGR2RGB)


        ax = self.fig.add_axes([0.025, 0.25, 0.95, 0.5])
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = self.fig.add_axes([0.025, 0.2, 0.95, 0.2])
        ax.imshow(lbl_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()


    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   function for plot an image and classification label
    #
    #arg:
    #   img: numpy array of image in BGR format
    #   lbl: classification label in format of ONE_HOT_CODE that is a numpy number ( shape= (n,) )
    #______________________________________________________________________________________________________________________________________________
    def classification(self, img, lbl):
        H = 50
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        lbl_img = np.zeros((H,w,3), dtype=np.uint8)
        class_num = len(lbl)
        class_width = w//class_num

        for i in range(len(lbl)):
            if lbl[i]==1:
                lbl_img[:, class_width*i+5: class_width*(i+1)-5 ,1] = 255
                cv2.putText( lbl_img, "CLASS_" + str(i), ( int(class_width*i + class_width*0.4), H//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0))
            else:
                lbl_img[:, class_width*i+5: class_width*(i+1)-5 ,2] = 255
                cv2.putText( lbl_img, "CLASS_" + str(i), ( int(class_width*i + class_width*0.4), H//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255))
        
        lbl_img = cv2.cvtColor(lbl_img, cv2.COLOR_BGR2RGB)


        ax = self.fig.add_axes([0.025, 0.25, 0.95, 0.5])
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = self.fig.add_axes([0.025, 0.2, 0.95, 0.2])
        ax.imshow(lbl_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()



    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   function for plot an image and mask label
    #
    #arg:
    #   img: numpy array of image in BGR format
    #   lbl: mask label in that is a numpy number ( shape= (n,h,w) )
    #______________________________________________________________________________________________________________________________________________
    def mask(self, img, lbl):
        H = 50
        border = 0.025
        num_mask = len(lbl)
        h_mask = (1 - border * ( num_mask + 1 ))/( num_mask + 1 )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_axes = [0.025,1 - h_mask, 0.95, h_mask]
        ax = self.fig.add_axes(img_axes)
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        

        for i in range(num_mask):
            mask = lbl[i]
            mask = cv2.bitwise_and(img,img,mask=mask)
            img_axes[1] -= (h_mask + border )
            img_axes[-1] = h_mask
            ax = self.fig.add_axes(img_axes)
            ax.imshow(mask)
            ax.get_xaxis().set_visible(False)
            ax.set_yticklabels([])
            ax.set_ylabel('CLASS_' + str(i))
        plt.show()
    


    
    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   sample an data row (img,lbl) from generator
    #
    #arg:
    #   state: if state=='next' returned the next data row. else if state=='back' it returned perivous data row
    #______________________________________________________________________________________________________________________________________________
    def sample(self, state ):
        if state =='next':
            self.idx+=1
        elif state == 'back':
            self.idx-=1

        if self.batch_img is None:
            self.batch_img, self.batch_lbl = next( self.gen )
            self.batch_size = len( self.batch_img )
            self.idx = 0


        if self.idx > self.batch_size - 1:
            self.batch_img, self.batch_lbl = next( self.gen )
            self.idx = 0
        
        elif self.idx < 0:
            self.idx = 0

        img = self.batch_img[ self.idx ]
        lbl = self.batch_lbl[ self.idx ]
        img = (img * (255/img.max()) ).clip(0,255)
        img = (img).astype(np.uint8)
        #if lbl is mask
        if len(lbl.shape) > 2:
            lbl *= 255
            lbl = lbl.astype(np.uint8)
            lbl = np.moveaxis(lbl, [0,1,2],[1,2,0]) #from (h,w,channel) to (channel,h,w)

        return img, lbl
        







if __name__ == '__main__':
    lbls_path = 'severstal-steel-defect-detection/annotations'

    extractor_func1 = DataReader.extact_binary()
    extractor_func2 = DataReader.extract_class(4,consider_no_object=False)
    extractor_func3 = DataReader.extract_mask(4, (256,1600), consider_no_object=False)

    gen = DataReader.generator( lbls_path, extractor_func3, annonations_name=None, batch_size=32, aug=None, rescale=255)

    viewer = Viewer(gen)
