import csv
import os
import numpy as np
import cv2
from numpy.lib.shape_base import split
import random
csv_path = 'severstal-steel-defect-detection/train.csv'
img_path = 'severstal-steel-defect-detection/train_images'
from sys import getsizeof

#______________________________________________________________________________________________________________________________________________
#explain:
#   get path of images and split into val and train images_name list
#
#arg:
#   path: path of images
#   split: a float number that determine amount of split
#   shuffle: if True, the images list shuffle
#
#return:
#   train_list, val_list
#   train_list: list of images_name for train
#   val_list: list of images_name for validation
#______________________________________________________________________________________________________________________________________________
def get_imgs_list(path, split=0.2, shuffle=True):
    imgs_list = os.listdir(img_path)

    if shuffle:
        random.shuffle(imgs_list)
    imgs_count = len(imgs_list)
    val_list   = imgs_list[ : int(imgs_count * split)]
    train_list = imgs_list[ int(imgs_count * split) : ]
    return train_list, val_list





#______________________________________________________________________________________________________________________________________________
#explain:
#   get dict_lbl( from csv2labelDict ) and images_name_list (from get_imgs_list()) and return binary label array
#   0: no defect
#   1: has defect
#arg:
#   dict_lbl: dictionary label that returned from csv2labelDict function
#   imgs_list: a batch or full list of imags_name that we want get their binary label
#
#return:
#   binary_lbl, imgs_list
#   binary_lbl: binary label of imgs_list that is a 1d numpy array 
#   imgs_list: it is excatly imgs_list aurguman
#______________________________________________________________________________________________________________________________________________
def get_binary_labels( dict_lbl, imgs_list ):
    binary_lbl_func = lambda x: 1 if x in dict_lbl.keys() else 0
    binary_lbl = np.array( list(map( binary_lbl_func , imgs_list)) )
    return binary_lbl,imgs_list



#______________________________________________________________________________________________________________________________________________
#explain:
#   get dict_lbl( from csv2labelDict ) and images_name_list (from get_imgs_list()) and return mult classfication label in one hot code
#
#arg:
#   dict_lbl: dictionary label that returned from csv2labelDict function
#   imgs_list: a batch or full list of imags_name that we want get their binary label
#   class_num: number of class. background class shouldn't acount
#   no_defect: if True, it Allocates a new class to no defect. it's class is 0 class
#
#return:
#   class_lbl, imgs_list
#   class_lbl: classification label for images_name_list
#   imgs_list: it is excatly imgs_list aurguman
#______________________________________________________________________________________________________________________________________________
def get_class_labels(dict_lbl, imgs_list, class_num, no_defect=False):
    def class_lbl_func(img_name):
        _class_ = np.zeros((class_num,))
        if img_name in dict_lbl.keys():
            img_class = np.array( dict_lbl[img_name][0] )
            _class_[ img_class ] = 1
        if no_defect:
            #if no defect, no_defect class value should be 1 else 0
            if np.sum(_class_) == 0:
                _class_ = np.insert(_class_,0,1)
            else:
                _class_ = np.insert(_class_,0,0)
        return _class_

    classes_lbl = list( map( class_lbl_func, imgs_list))
    classes_lbl = np.array( classes_lbl)
    return classes_lbl,imgs_list





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
def _conv_label_to_mask(lbl , height = 256 , width = 1600):
    # Lbl is an np.array
    lbl_mod = lbl.copy()
    lbl_mod[:,1] += lbl_mod[:,0]
  
    mask = np.zeros((height * width))

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





csv_list = csv_reader(csv_path)
dict_lbl = csv2labelDict(csv_list)
imgs_list,b =  get_imgs_list(img_path)

bin_lbl,_ = get_binary_labels(dict_lbl, imgs_list)
classes_lbl,_ = get_class_labels(dict_lbl,imgs_list,4)


