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


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def _create_jsondict( name, elements, pth ):
    # read element
    img = cv2.imread( path.join(pth , name) )

    color_mode = _find_colormode(img)
    img_shape = _get_img_shape(img)
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
        'included_object': 'YES',
        'label_type': 'MASK',
        'labels' : labels
    }

def _save_json( jsn , save_path , singleLine = False):
    json_name = path.splitext(jsn['name'])[0] + ".json"
    with open(path.join(save_path , json_name) , 'w') as jsfile:
        if singleLine:
            json.dump(jsn , jsfile)
        else:   
            json.dump(jsn , jsfile , indent= 4)

def convert_csv_to_json(images_path , csv_path , save_path, singleLine = False):

    csv_file = csv_reader(csv_path)
    dict_lbl = csv2labelDict(csv_file)

    print("Conversion Started. Please Wait...")
    counter = 0
    total = len(dict_lbl.items())
    for key, val in dict_lbl.items():
        printProgressBar (counter, total, prefix = 'Converting CSV to JSON', suffix = 'Completed', decimals = 1, length = 100, fill = '█', printEnd = "\r")
        json_dict = _create_jsondict(key , val , images_path)
        _save_json(json_dict , save_path, singleLine)   
        counter += 1


def create_json_for_unlabeled_image(image_path , json_path , singleLine = False):

    json_path_list = os.listdir(json_path)
    image_path_list = os.listdir(image_path)

    def reduce_path(filename , pth):
        if path.isfile(path.join(pth,filename)):
            fn_extended = path.splitext(filename)

            return fn_extended[:-1][0] , fn_extended[-1] 
    
    json_path_list_reduced = list(
        map(lambda inp: reduce_path(inp , json_path)[0] , json_path_list)
    )

    image_path_list_reduced = list(
        map(lambda inp: reduce_path(inp , image_path) , image_path_list)
    )

    # print(json_path_list_reduced)
    total = len(image_path_list_reduced)
    iteration = 1

    print("\nCreating label for unlabled images. Please Wait...")
    for i_fname , i_fname_extension in image_path_list_reduced:
        printProgressBar (iteration, total, prefix = 'Creating JSONS', suffix = 'Completed', decimals = 1, length = 100, fill = '█', printEnd = "\r")
        if  not i_fname in json_path_list_reduced:

            img = cv2.imread( path.join(image_path , i_fname + i_fname_extension) )
            color_mode = _find_colormode(img)
            img_shape = _get_img_shape(img)

            temp_dic = {
                'name': i_fname + i_fname_extension,
                'path': image_path,
                'color_mode': color_mode,
                'size': list(img_shape),
                'included_object': 'NO',
                'label_type': '',
                'labels' : ''
            }

            _save_json(temp_dic , json_path , singleLine = singleLine)
            iteration += 1


def _get_img_shape( img ):
    return img.shape[:2]

def _print_progress_bar(max = 100 , cur = 0):
    print("")

def _find_colormode( image , threshold = 20 ):
    b_g = np.abs( image[:,:,0] - image[:,:,1] )
    g_r = np.abs( image[:,:,1] - image[:,:,2] )
    b_r = np.abs( image[:,:,0] - image[:,:,2] )

    if np.sum(b_g) < threshold and np.sum(g_r) < threshold and np.sum(b_r) < threshold:
        return "GRAYS"
    
    else:
        return "COLOR"

# def _pretify_lists(json_path):
#     print(json_path)
#     print(path.exists(json_path))

#     with open(json_path , "+r") as file:
#         str_file = file.read()
#         split_str = str_file.split('"mask"')
#         split_str[1] = split_str[1].replace("\n" , "").replace(" " , "")
#         file.seek(0)
#         file.write(split_str[0]+split_str[1])

def create_all_json(image_path , csv_path , json_save_path):

    os.system('clear')
    convert_csv_to_json(image_path , csv_path , json_save_path , singleLine = False)
    create_json_for_unlabeled_image(image_path , json_save_path , singleLine = False )
    print("Conversion Finished Successfully!")

def main():
    # _pretify_lists(r'.\severstal-steel-defect-detection\annotations\000a4bcdd.json')

    images_path = r"./severstal-steel-defect-detection/train_images"
    csv_path = r"./severstal-steel-defect-detection/train.csv"
    json_save_path = r"./severstal-steel-defect-detection/annotations"
    create_all_json(images_path , csv_path , json_save_path)
if __name__ == "__main__":
    main()