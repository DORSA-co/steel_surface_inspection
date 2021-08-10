import csv
import os
from typing_extensions import Annotated
import numpy as np
import cv2
from numpy.core.fromnumeric import reshape
from numpy.lib.shape_base import split
import random
from sys import getsizeof

import json

lbls_path = 'severstal-steel-defect-detection/annotations'
imgs_path = 'severstal-steel-defect-detection/train_images'


#______________________________________________________________________________________________________________________________________________
#explain:
#   a class for  masks of image. it contains maskes and their classes
#atribiut:
#   classes: array of classes Id
#   codedMaskes_: array of masks in format of [[start px, end px],...]
#______________________________________________________________________________________________________________________________________________
class Mask():
    def __init__(self):
        self.class_ = None
        self.codedMask_ = None
        


#______________________________________________________________________________________________________________________________________________
#explain:
#   a class for read json anntotion label
#
#atribiut:
#   annotation: dict of json file
#
#methods:
#
#   -------------------------------
#   __init__:
#       explain:
#           set the annotation atribiut
#       args:
#           path: path of json file
#       
#   -------------------------------
#   __read__:
#       explain:
#           read jason file and convert to dict
#       args:
#           path: path of json file
#       returns:
#          annotation: dictionary of json file
#   -------------------------------
#   get_fname:
#       explain:
#           return image file name image
#       returns:
#          fname: string of image's name
#   -------------------------------
#   get_path:
#       explain:
#           return path of image
#       returns:
#          fname: string of image's path
#   -------------------------------
#   get_fullpath:
#       explain:
#           return full path of image ( path + image_full_name)
#   -------------------------------
#   get_img_size:
#       explain:
#           return size of image in tuple (w,h)
#   -------------------------------
#   get_classes:
#       explain:
#           return classes id of all objects in image in np.array
#   -------------------------------
#   get_masks:
#       explain:
#           return list of objects label contain masks and classes in format of Label() class
#   -------------------------------
#   get_bboxs:
#       TBD
#   -------------------------------
#   is _color:
#       explain:
#           return True, if image is colored
#   -------------------------------
#   is _gray:
#       explain:
#           return True, if image is gay
#   -------------------------------
#   is _mask:
#       explain:
#           return True, type of localisation's label is mask format
#   -------------------------------
#   is is_lbl_bbox:
#       explain:
#           return True, type of localisation's label is bounding box format
#   -------------------------------
#______________________________________________________________________________________________________________________________________________
class Annotation():

    def __read__(self, path):
        with open(path) as jfile:
            file = json.load(jfile)
        return file
        

    def __init__(self, path):
        self.annotation = self.__read__(path)

    def get_fname(self):
        return self.annotation['name']

    def get_path(self):
        return self.annotation['path']

    def get_fullpath(self):
        return os.path.join(self.annotation['path'], self.annotation['fname'] )

    def get_img(self):
        return cv2.imread( self.get_fullpath() )

    def get_img_size(self):
        return tuple( self.annotation['size'] )

    def get_classes(self):
        assert self.have_object(), "There is no object"
        classes = []
        labels = self.annotation['labels']
        for lbl in labels:
            classes.append( lbl['class'])
        return np.array(classes)
    
    def get_masks(self):
        assert self.have_object(), "There is no object"
        assert self.is_lbl_mask(), "Label type is not mask"
        labels = self.annotation['labels']
        mask_list = []
        for lbl in labels:
            print(lbl.keys())
            msk_obj = Mask()
            msk_obj.class_ = int(lbl['class'])
            msk_obj.mask_ = np.array( lbl['mask'] ).reshape((-1,2)).astype(np.int32)
            mask_list.append(msk_obj)
        return mask_list




    
    def get_bboxs(self):
        assert self.have_object(), "There is no object"
        assert self.is_lbl_bbox(), "Label type is not bounding box"


    def is_color(self):
        return self.annotation['color_mode'] == 'COLOR'
    
    def is_gray(self):
        return self.annotation['color_mode'] == 'GRAY'

    def is_lbl_mask(self):  
        assert self.have_object(), "There is no object"
        return self.annotation['label_type'] == 'MASK'

    def is_lbl_bbox(self):  
        assert self.have_object(), "There is no object"
        return self.annotation['label_type'] == 'BBOX'

    def have_object(self):  
        return self.annotation['included_object'] == 'YES'
    
    
    
        
       






#______________________________________________________________________________________________________________________________________________
#explain:
#   get path of labels and split into val and train lbl_file_name list
#
#arg:
#   path: path of json labels
#   split: a float number that determine amount of split
#   shuffle: if True, the labels list shuffle
#
#return:
#   lbls_train_list, lbls_train_list
#   lbls_train_list: list of lbl_file_name for train
#   lbls_train_list: list of lbl_file_name for validation
#______________________________________________________________________________________________________________________________________________
def get_labels_name_list(lbls_path, split=0.2, shuffle=True):
    lbls_list = os.listdir(lbls_path)
    if shuffle:
        random.shuffle(lbls_list)

    lbls_count = len(lbls_list)
    lbls_val_list   = lbls_list[ : int(lbls_count * split)]
    lbls_train_list = lbls_list[ int(lbls_count * split) : ]
    return lbls_train_list, lbls_val_list



#______________________________________________________________________________________________________________________________________________
#explain:
#   get path of labels and a list of labels and return list of labels in annotation() class format
#
#arg:
#   lbls_list: list of labels name
#   lbls_path: path of folder of labels 
#
#return:
#   labels
#   labels : a annoation of label in annotation() class format
#
#______________________________________________________________________________________________________________________________________________
def read_label(lbls_list, lbls_path):
    labels = []
    for lbl_name in lbls_list:
        labels.append( Annotation( os.path.join( lbls_path, lbl_name ))  )
    return labels


#______________________________________________________________________________________________________________________________________________
#explain:
#   get anonations list and return images and binary labels
#   0: no object
#   1: has object
#arg:
#   annotations: a batch or full list of instance of annonation() class
#
#return:
#   imgs, lbls
#   imgs: batch of images
#   lbls: batch of binary labels
#______________________________________________________________________________________________________________________________________________
def get_binary_datasets( annotations ):
    
    lbls = []
    imgs = []
    for annotation in annotations:
        lbls.append( int(annotation.have_object()) )
        imgs.append( annotation.get_img())

    return np.array(imgs),np.array(lbls )



#______________________________________________________________________________________________________________________________________________
#explain:
#   get a list of anonations( instance of Anonation() class ) and return images and class labels
#
#arg:
#   annotations , class_num, consider_no_object
#   annotations: a batch or full list of instance of anonations( instance of Anonation() class ) it obtaon from read_label() function
#   class_num: number of class. no_object class shouldn't acount
#   consider_no_object: if True, it Allocates a new class to no object. it's class is 0 class. defuat is False
#
#return:
#   class_lbl, imgs_list
#   class_lbl: classification label for images_name_list
#   imgs_list: it is excatly imgs_list aurguman
#______________________________________________________________________________________________________________________________________________
def get_class_datasets(annotations, class_num, consider_no_object=False):

    lbls = []
    imgs = []
    for annotation in annotations:
        lbl =  np.zeros((class_num,))

        if annotation.have_object():
            classes = annotation.get_clases()
            classes - 1 #in json file class started ferm numer 1
            lbl[classes] = 1
        
        if consider_no_object:
            #if no defect, no_defect class value should be 1 else 0
            if np.sum(_class_) == 0:
                _class_ = np.insert(_class_,0,1)
            else:
                _class_ = np.insert(_class_,0,0)
        
        lbls.append( lbl )
        imgs.append( annotation.get_img())


    return np.array(imgs),np.array(lbls )





#______________________________________________________________________________________________________________________________________________
#explain:
#   takes a label from train.csv file and converts it to image mask. This function can be used for different width and height
#
#arg:
#   lbl = a label with the same style as labels in csv2labelDict. An np.array which shows the [n * [start_pix, column_spacing]].
#   width = image width
#   height = image height
#
#return:
#   image_mask = a mask that shows the location of the certain defects
#______________________________________________________________________________________________________________________________________________
def _encoded_mask(coded_mask , height = 256 , width = 1600):
    # Lbl is an np.array
    lbl_mod = mask_raw.copy()
    lbl_mod[:,1] += lbl_mod[:,0]
  
    mask = np.zeros((height * width))

    def assign_val(coded_)
    for lbl in lbl_mod:
        mask[lbl[0]:lbl[1]] = 255

    return mask.reshape((width,height)).T

#______________________________________________________________________________________________________________________________________________
#explain:
#   takes one element of csv2labelDict output (the value of an element in the dictionary) and convert the desiered lable to image mask.
#   If nothing is passed for cls, all labels are converted to masks and returned as an array of masks.
#   If considerBackground is set to True, all classes will increment by one an the background mask is calculated if necessary.
#
#
#arg:
#   dict_lbl = a tuple ([classes] , [ [label]s , ... ])
#   cls = class number. If ConsiderBackground == True, the main classes start from 1 and the 0 class is the background maske.
#       if not, the classes will start at 0.
#   considerBackground = If true, the 0 class is considered as the background mask class and if necessary, background mask
#       is calculated.
#   width = image width
#   height = image height
#
#return:
#   (class , mask)
#       class = an array of classes
#       mask = an array of masks 
#______________________________________________________________________________________________________________________________________________
def get_img_mask(dic , cls = None ,  considerBackground = False , height = 256 , width = 1600):

    classes = dic[0].copy()
    labels = dic[1].copy()

    def calc_background_mask(arg_labels = []):

        if len(arg_labels) == 0:

            arg_labels = np.array(
                    list( map( lambda x: conv_label_to_mask(x , width = width , height = height)  , labels ) )
                )

        all_masks = np.sum(arg_labels , axis = 0).clip(0 , 255)
        all_masks -= 255
        all_masks *= -1

        return all_masks

    if considerBackground:
        classes = list( map(lambda x: x+1 , classes))
        classes.append(0)
    
        if cls == None:

            labels = np.array(
                list( map( lambda x: _conv_label_to_mask(x , width = width , height = height)  , labels ) )
            )

            np.insert( labels , 0 , calc_background_mask(labels) , axis = 0)

            return classes , labels

        elif cls == 0:
            return [cls] , [calc_background_mask()]

        else:
            return [cls] , [_conv_label_to_mask(labels[cls - 1])]

    else:
        if cls == None:

            labels = np.array(
                list( map( lambda x: _conv_label_to_mask(x , width = width , height = height)  , labels ) )
            )
            return classes , labels

        else:
            return [cls] , [_conv_label_to_mask(labels[cls])]



if __name__ == '__main__':
    '''

    csv_list = csv_reader(csv_path)
    dict_lbl = csv2labelDict(csv_list)
    imgs_list,b =  get_imgs_list(img_path)

    bin_lbl,_ = get_binary_labels(dict_lbl, imgs_list)
    classes_lbl,_ = get_class_labels(dict_lbl,imgs_list,4)


    '''
    lbls_train_list,lbls_val_list = get_labels_name_list(lbls_path)
    annotations = read_label(lbls_train_list,lbls_path)
    imgs,lbls = get_binary_datasets(annotations[:64])
    js = Annotation('Json_sample.json')