import os
from typing_extensions import Annotated
import numpy as np
import cv2
from numpy.core.fromnumeric import reshape
from numpy.lib.shape_base import split
import random
from sys import getsizeof
import json



CLASSIFICATION_TYPE = 1
BINARY_TYPE = 2
MASK_TYPE = 3

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
#   get_encoded_mask:
#       explain:
#           returns the encoded masks of an anotation. If cls = None (or not passed any), the function will return ALL the masks
#           for the annotation. If passed an integer, if available, the method will return the mask with the passed integer class. 
#           If the integer is not valid, an assertation will rise.
#   -------------------------------
#   get_decoded_masks:
#       explain:
#           Uses the encoded masks available in the annotation and decoedes the requested masks.
#           If nothing is passed for cls, all encoded masks are decoded and returned as an array of masks.
#           If considerBackground is set to True, all classes will increment by one and the background mask is calculated if necessary.
#
#           cls = class number. If ConsiderBackground == True, the main classes start from 1 and the 0 class is the background maske.
#                   if not, the classes will start at 0.
#           considerBackground = If true, the 0 class is considered as the background mask class and if necessary, background mask
#                   is calculated.
#
#           return:
#               (class , mask)
#               class = an array of classes
#               mask = an array of masks 
#           
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
#   is_class_valid:
#       explain:
#           returns true if the mask class is available in the anotation. False if not.
#   --------------------------------
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
            
            msk_obj = Mask()
            msk_obj.refrenced_size_ = self.get_img_size()
            msk_obj.class_ = int(lbl['class'])
            msk_obj.codedMask_ = np.array( lbl['mask'] ).reshape((-1,2)).astype(np.int32)

            if (cls == msk_obj.class_):
                return [msk_obj]
            
            mask_list.append(msk_obj)

        return mask_list



    def get_decoded_masks(self,  cls = None , considerBackground = False , **kwargs ):

        height , width = self.get_img_size()

        raw_masks = self.get_encoded_mask()

        def calc_background_mask():

            raw_masks_decoded = np.array(
                    list( map( lambda x: x.encode_mask()  , raw_masks ) )
                )

            all_masks = np.sum(raw_masks_decoded , axis = 0).clip(0 , 255)
            all_masks -= 255
            all_masks *= -1

            return all_masks

        if considerBackground:
            # Add all classes by one

            if cls == None:
                classes = list(
                    map(lambda x: x.class_ + 1 , raw_masks)
                )

                encoded_masks = np.array(
                    list( map( lambda x: x.encode_mask() , raw_masks ) )
                )

                classes.append(0)

                np.append( encoded_masks , 0 , calc_background_mask() , axis = 0)

                return classes , encoded_masks

            elif cls == 0:
                return [0] , [calc_background_mask()]

            else:

                assert self.is_class_valid(cls - 1), "Requested mask-class is not valid!"

                return [cls] , [self.get_encoded_mask(cls = cls - 1)[0].encode_mask()]

        else:
            if cls == None:
                
                classes = self.get_classes()

                devcoded_masks = np.array(
                    list( map( lambda x: x.encode_mask()  , raw_masks ) )
                )
                return classes , devcoded_masks

            else:
                assert self.is_class_valid(cls), "Requested mask-class is not valid!"

                return [cls] , [self.get_encoded_mask(cls = cls)[0].encode_mask()]



    
    def get_bboxs(self):
        assert self.have_object(), "There is no object"
        assert self.is_lbl_bbox(), "Label type is not bounding box"

    def is_class_valid(self , cls ):
        classes = self.get_classes()
        
        if cls in classes:
            return True

        else:
            return False

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
#   get path of labels and a annonation_name and return annotation() class format
#
#arg:
#   annonations_path: path of folder of annonations 
#   annonation_name: name of specific annonation
#
#return:
#   labels
#   labels : a annoation of label in annotation() class format
#
#______________________________________________________________________________________________________________________________________________
def read_annotation(annonations_path, annonation_name):
    return  Annotation( os.path.join( annonations_path, annonation_name ) )


#______________________________________________________________________________________________________________________________________________
#explain:
#   return binary dataset extractor
#
#arg:
#
#return:
#   func: extractor function, that get an annonation ( Instance if Annonation() class ) and return image, binary_lbl
#______________________________________________________________________________________________________________________________________________
def extact_binary():
    def func(annotation):
        lbl = int(annotation.have_object()) 
        img = annotation.get_img()
        return np.array(img),np.array(lbl )
    return func



#______________________________________________________________________________________________________________________________________________
#explain:
#   get an anonations( instance of Anonation() class ) and return its image and class label
#
#arg:
#   class_num, consider_no_object
#   class_num: number of class. no_object class shouldn't acount
#   consider_no_object: if True, it Allocates a new class to no object. it's class is 0 class. defuat is False
#
#return:
#   func: extractor function, that get an annonation ( Instance if Annonation() class ) and return image, classificaiotn_label ( in one_hot_code format )
#______________________________________________________________________________________________________________________________________________
def extract_class( class_num, consider_no_object=False):

    def func(annotation):
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
        
        img = annotation.get_img()
        return np.array(img),np.array(lbl )
    return func

#______________________________________________________________________________________________________________________________________________
#explain:
#   genreat inputs and labels batch
#
#
#arg:
#   annonations_path: path of annonations file
#   extractor_func: an exrtracor function that get an annonation and returns its image and label
#   annonations_name: list of file name of desire  annonations. if None , it load all the annonations in path
#   rescale: rescale value that images and masks divided on it (defualt = 255)
#   batch_size: size of batchs
#   aug: augmention object( instance of Augmention() class from augmention.py file. if None there is no augmention
#   infinit: if False, it returns batchs just for one epoch
#
#return:
#   (batch_inputs , batch_lbls)
#   batch_inputs: batch of images that are ready for train
#   batch_lbls: batch of labels that are ready for train
#
#______________________________________________________________________________________________________________________________________________
def generator(annonations_path, extractor_func, annonations_name=None,rescale=255, batch_size = 32, aug = None, infinit=True):
    
    batch_inputs = []
    batch_lbls = []
    if annonations_name is None:
        annonations_name = os.listdir(annonations_path)
    
    while infinit:
        for name in annonations_name:    
            annonation = read_annotation( annonations_path, name)
            img, lbl = extractor_func(annonation)
            if aug is not None:
                if len(lbl.shape) < 2: #binary or classification
                    img = aug.augment_single(img)
                    img = img.astpye(np.float32) / rescale

                else: #Mask
                    img, lbl = aug.augment_single_byMask(img, lbl)
                    img = img.astpye(np.float32) / rescale
                    lbl = lbl.astpye(np.float32) / rescale
            
            batch_lbls.append( lbl )
            batch_inputs.append( img )

            if len(  batch_inputs) == batch_size:
                yield np.array(batch_inputs), np.array(batch_lbls)
                batch_inputs, batch_lbls = [] , []










if __name__ == '__main__':
    
    lbls_path = 'severstal-steel-defect-detection/annotations'
    imgs_path = 'severstal-steel-defect-detection/train_images'

    # extractor_func1 = extact_binary()
    # extractor_func2 = extract_class(class_num=4, consider_no_object=False)
    
    # gen = generator( lbls_path, extractor_func1, annonations_name=None, batch_size=32, aug=None)
    # x1,y1 = next(gen)
    # x2,y2 = next(gen)
    # filter_arg={'label_type':["BBOX","MASK"], 'class':[3]}
    # filtered = filter_annonations(annonations_name, path, filter_arg)
    # annontions_names = get_annonations_name(lbls_path)

    # annontions_names_train, annontions_names_val = split_annonations_name(annontions_names)
    # annotations = read_annotations(annontions_names_train,lbls_path)
    # imgs,lbls = get_class_datasets(annotations[:1000],4, consider_no_object=True)

    js = Annotation(os.path.join(lbls_path , '0025bde0c.json' ))
    img = js.get_decoded_masks(cls=0 , considerBackground=True)[1][0]
