import csv
import os
import numpy as np
import cv2
from numpy.lib.shape_base import split
import random
csv_path = 'severstal-steel-defect-detection/train.csv'
img_path = 'severstal-steel-defect-detection/train_images'

#______________________________________________________________________________________________________________________________________________
#expalin:
#   read csv file from path and return csv list
#
#arg:
#   csv_path: path of csv label
#
#return:
#   csv list: 
#       col_0: image_name, col_1: Class_ID, col_2: Encoded_pixel
#______________________________________________________________________________________________________________________________________________
def csv_reader( csv_path):
    with open( csv_path, newline='') as csvfile :
            csv_iter = csv.reader( csvfile)
            csv_file = list(csv_iter)
            return csv_file[1:]  




#______________________________________________________________________________________________________________________________________________
#explain:
#   ->Convert csv list to dictionay that each key is a image_name and value is a tupel contatin classes_list and Encodded_pixel_list
#   ->'image_name': ([classes], [encoded_pilxes_array])
#   ->class: an int number
#   ->encoded_pilxes_array: an array with shape(n,2) that first col is start pixel and second col is count pix ( order is form top to down then lft to right)
#
#arg:
#   csv_list: output of csv_reader
#
#return:
#   dict label
#______________________________________________________________________________________________________________________________________________
def csv2labelDict( csv_list):

    dict_lbl = {}
    #Row -> image_name, class_id, mask_row_lenght_code
    for row in csv_list:
        img_name, class_id, encoded_pixel = row
        class_id = int(class_id) - 1 #in csv class start from 1
        encoded_pixel = encoded_pixel.split(' ')
        encoded_pixel = list(  map( int, encoded_pixel ))
        encoded_pixel = np.array(encoded_pixel).reshape( (-1,2) ) #firrst col-> start pix , second col -> count pix

        if dict_lbl.get( img_name ) is None:
            dict_lbl[ img_name ] = ( [class_id], [encoded_pixel] )
        
        else:
            classes_list, encodedPx_list = dict_lbl.get( img_name )
            classes_list.append( class_id )
            encodedPx_list.append( encoded_pixel )
            dict_lbl[img_name] = (classes_list, encodedPx_list)
    return dict_lbl



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



csv_list = csv_reader( csv_path)
dict_lbl = csv2labelDict(csv_list)
imgs_list,b =  get_imgs_list(img_path)

bin_lbl,_ = get_binary_labels(dict_lbl, imgs_list)
classes_lbl,_ = get_class_labels(dict_lbl,imgs_list,4)



    
    
    




    
m=0