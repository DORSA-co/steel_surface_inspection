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
        class_id = int(class_id)
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







    




    
m=0










#______________________________________________________________________________________________________________________________________________
#author: Alireza Khalilian
#
#explain:
#   given an element number, (considering that elements are numbered from top to bottom in a column) returns the column and row number of the pixels
#   in the (row, colm) format. Note that the images are sized 1600 x 256
#
#arg:
#   elm_num: coresponding element number
#
#return:
#   a tupe in (row, column) format.
#______________________________________________________________________________________________________________________________________________

def get_row_col(elm_num):
    return ( elm_num % 256 , int(elm_num / 256) ,  )



def expandLabels(dict_lbl):

    modified_csv_list = {}
    dict_lbl_bcup = dict_lbl.copy()

    for key, val  in dict_lbl_bcup.items():
        np_lbl_array = val[1][0]            ## Extract the lables from the dictionary
        elm_cord = list( map(get_row_col , np_lbl_array[:,0]) )     ## Convert element-number to (row,col)
        elm_cord = np.array(elm_cord)
        elm_cord = np.insert( elm_cord , 2 , np_lbl_array[:,1] , axis= 1)       ## insert the end_columns to the array
        dict_lbl_bcup[key][1][0] = elm_cord


        get_image_mask(elm_cord)


    return dict_lbl_bcup


def get_image_mask(label):

    print(label)
    mask = np.zeros((256 , 1600))
    begin_cols = label[:,1]
    end_cols = label[:,2] + label[:,1]
    row = label[:,0]
    
    # cols = list( map(lambda x , y , z: np.arange(x , y) , begin_cols , end_cols ) )
    # cols = np.array(cols).reshape((-1 , 1))

    mask[ np.arange(begin_cols[:] , end_cols[:]) , row ] = 1

    cv2.waitKey(0)

    return mask



print(os.curdir)
csv_list = csv_reader( csv_path)
dict_lbl = csv2labelDict(csv_list)
dict_lbl_bcup = expandLabels(dict_lbl)
# a,b =  get_imgs_list(img_path)

