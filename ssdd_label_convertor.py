import csv
import os
import numpy as np
import cv2
from os import path
import json
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
def csv_reader( csv_path ):
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
def csv2labelDict( csv_list ):

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


## A function for creating single Json file's string
def create_jsondict( name, elements, pth ):
    # read element
    img = cv2.imread( path.join(pth , name) )

    color_mode = find_colormode(img)
    img_shape = get_img_shape(img)
    label_type = "MASK"
    labels = []
    for idx , cls in enumerate(elements[0]):
        labels_list = elements[1][idx].tolist()
        dic_tmp = {'class': cls , 'mask': labels_list}
        labels.append(dic_tmp)

    return {
        'name': name,
        'path': pth,
        'color_mode': color_mode,
        'size': list(img_shape),
        'label_type': 'MASK',
        'labels' : labels
    }

def save_json( jsn , save_path ):
    json_name = path.splitext(jsn['name'])[0] + ".json"
    with open(path.join(save_path , json_name) , 'w') as jsfile:
        json.dump(jsn , jsfile)

def convert_csv_to_json(images_path , csv_path , save_path):

    csv_file = csv_reader(csv_path)
    dict_lbl = csv2labelDict(csv_file)
    
    print("Conversion Started. Please Wait...")
    counter = 0
    for key, val in dict_lbl.items():
        print(f"Converted Files: {counter} / {len(dict_lbl)}" , end = "\r")
        json_dict = create_jsondict(key , val , images_path)
        save_json(json_dict , save_path)   
        counter += 1

def get_img_shape( img ):
    return img.shape[:2]


def find_colormode( image , threshold = 20 ):
    b_g = np.abs( image[:,:,0] - image[:,:,1] )
    g_r = np.abs( image[:,:,1] - image[:,:,2] )
    b_r = np.abs( image[:,:,0] - image[:,:,2] )

    if np.sum(b_g) < threshold and np.sum(g_r) < threshold and np.sum(b_r) < threshold:
        return "GRAYSCALE"
    
    else:
        return "COLOR"

images_path = r"./severstal-steel-defect-detection/train_images"
csv_path = r"./severstal-steel-defect-detection/train.csv"
save_path = r"./severstal-steel-defect-detection/jsons"

convert_csv_to_json(images_path , csv_path , save_path)