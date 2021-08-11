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
        self.refrenced_size_ = None

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
    def encode_mask(self):
        
        ref_width = self.refrenced_size_[1]
        ref_height = self.refrenced_size_[0]

        coded_mask_mod = self.codedMask_.copy()
        coded_mask_mod[:,1] += coded_mask_mod[:,0]

        mask = np.zeros((ref_height * ref_width))

        for raw_mask in coded_mask_mod:
            mask[raw_mask[0]:raw_mask[1]] = 255

        return mask.reshape( (ref_width, ref_height) ).T
        


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
        return os.path.join(self.annotation['path'], self.annotation['name'] )

    def get_img(self):
        return cv2.imread( self.get_fullpath() )

    def get_img_size(self):
        return tuple( self.annotation['size'] )

    def get_classes(self):
        assert self.have_object(), "There is no object"
        classes = []
        labels = self.annotation['labels']
        for lbl in labels:
            classes.append( int(lbl['class']) )
        return np.array(classes)
    
    def get_encoded_mask(self, cls = None):
        assert self.have_object(), "There is no object"
        assert self.is_lbl_mask(), "Label type is not mask"

        if cls != None:
            assert self.is_class_valid(cls) , "Class Not Valid" 

        labels = self.annotation['labels']
        mask_list = []

        for lbl in labels:
            print(lbl.keys())
            msk_obj = Mask()
            msk_obj.refrenced_size_ = self.get_img_size()
            msk_obj.class_ = int(lbl['class'])
            msk_obj.mask_ = np.array( lbl['mask'] ).reshape((-1,2)).astype(np.int32)

            if (cls == msk_obj.class_):
                return [msk_obj]
            
            mask_list.append(msk_obj)

        return mask_list

    def is_class_valid(self , cls ):
        classes = self.get_classes()
        
        if cls in classes:
            return True

        else:
            return False

    
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
#   filter annonation_name list base of filter_arg 
#
#arg:
#   annonation_name: list of annonation's name
#
#   path: path of annonations folder
#
#   filter_arg: it is a dictionay. it's key are same as annonation jason file, e.g "label_type", "color_mode" and etc, and
#   their values are a list of acceptabel values for related features. this Values are or by each other. it's also accept
#   the "class" key and don't accept "label" key for e.g if the filter_arg be { 'label_type':["MASK"], "class":[1,2] }
#   it pass annonations that their label_type are "MASK" and their object's class consist one of 1 or 2 class or both of them
#   
#
#return:
#   filtered
#   filtered: list of annontions_file_name that pass filters
#______________________________________________________________________________________________________________________________________________
def filter_annonations(annonations_name, path,filter_arg):

    def filter_func(annotation_name):

        with open(os.path.join(path,annotation_name)) as jfile:
            annonation_dict = json.load(jfile)
        
        for filter_key, filter_values in filter_arg.items():
            if filter_key == 'class':
                classes =[]
                labels = annonation_dict['labels']
                for lbl in labels:
                    classes.append( lbl['class'])

                flag = False
                for c in classes:
                    if c  in filter_values:
                        flag =  True
                if not flag:
                    return False                  
            else :
                if annonation_dict[filter_key] not in filter_values:
                    return False
        
        return True

    filtered = list( filter( filter_func, annonations_name))
    return filtered




#______________________________________________________________________________________________________________________________________________
#explain:
#   get path of annonations and return list of annonations_name ( json file's name )
#
#arg:
#   path: path of json labels
#   shuffle: if True, the labels list shuffle
#
#return:
#   annonations_name_list
#   annonations_name_list: list of annontions_file_name ( jason file's name)
#______________________________________________________________________________________________________________________________________________
def get_annonations_name(path, shuffle=True):
    annonations_name_list = os.listdir(path)
    if shuffle:
        random.shuffle(annonations_name_list)
    return annonations_name_list


#______________________________________________________________________________________________________________________________________________
#explain:
#   get list of annonation_file_name ( jason file's name) and split into val and train annonation_file_name list
#
#arg:
#   annonations_name_list: list of an_file_name ( jason file's name)
#   split: a float number that determine amount of split
#   shuffle: if True, the labels list shuffle
#
#return:
#   annonations_train_list, annonations_val_list
#   annonations_train_list: list of list of lbl_file_name for validation_file_name for validation
#   annonations_val_list: list of annontions_file_name for validation
#______________________________________________________________________________________________________________________________________________
def split_annonations_name(annonations_name_list, split=0.2, shuffle=True):
    lbls_count = len(annonations_name_list)
    annonations_val_list   = annonations_name_list[ : int(lbls_count * split)]
    annonations_train_list = annonations_name_list[ int(lbls_count * split) : ]
    return annonations_train_list, annonations_val_list



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
def read_annotations(lbls_list, lbls_path):
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
            classes = annotation.get_classes()
            classes - 1 #in json file class started ferm numer 1
            lbl[classes] = 1
        
        if consider_no_object:
            #if no defect, no_defect class value should be 1 else 0
            if np.sum(lbl) == 0:
                lbl = np.insert(lbl,0,1)
            else:
                lbl = np.insert(lbl,0,0)
        
        lbls.append( lbl )
        imgs.append( annotation.get_img())


    return np.array(imgs),np.array(lbls )


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
#
#      ATTENTION
#______________________________________________________________________________________________________________________________________________


if __name__ == '__main__':
    '''
    csv_list = csv_reader(csv_path)
    dict_lbl = csv2labelDict(csv_list)
    imgs_list,b =  get_imgs_list(img_path)

    bin_lbl,_ = get_binary_labels(dict_lbl, imgs_list)
    classes_lbl,_ = get_class_labels(dict_lbl,imgs_list,4)
    '''
    
        
    annonations_name = ['Json_sample.json']
    path = 'severstal-steel-defect-detection\\annotations'
    
    # filter_arg={'label_type':["BBOX","MASK"], 'class':[3]}
    # filtered = filter_annonations(annonations_name, path, filter_arg)
    # annontions_names = get_annonations_name(lbls_path)

    # annontions_names_train, annontions_names_val = split_annonations_name(annontions_names)
    # annotations = read_annotations(annontions_names_train,lbls_path)
    # imgs,lbls = get_class_datasets(annotations[:1000],4, consider_no_object=True)

    js = Annotation(os.path.join(path , '470a96423.json' ))
    print(js.get_encoded_mask(0)[0].mask_)