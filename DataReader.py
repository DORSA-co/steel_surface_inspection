import csv
import numpy as np

csv_path = 'severstal-steel-defect-detection/train.csv'

def csv_reader( csv_path):
    with open( csv_path, newline='') as csvfile :
            csv_iter = csv.reader( csvfile)
            csv_file = list(csv_iter)
            return csv_file[1:]  



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




csv_list = csv_reader( csv_path)
dict_lbl = csv2labelDict(csv_list)


    




    
m=0