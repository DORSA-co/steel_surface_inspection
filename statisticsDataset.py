import DataReader
import numpy as np
from matplotlib import pyplot as plt
from DataReader import Annotation
import os


#______________________________________________________________________________________________________________________________________________
#explain:
#   draw histogram of binary labels 
#atribiut:
#   annonation_path: path of annonations folder( jason files)
#   annonations_name: list of annonations name that we want calculate their histogram. if be None, histogram calculated for all annonations in directory
#______________________________________________________________________________________________________________________________________________

def binary_hist(annonation_path, annonations_name=None):

    if annonations_name is None:
        annonations_name = os.listdir( annonation_path )
    annonations_name = list( filter( lambda x:x[-5:]=='.json' , annonations_name))

    count_free  = 0
    count_object = 0

    for name in annonations_name:
        path = os.path.join( annonation_path, name) 
        anonation = Annotation(path)
        if anonation.have_object():
            count_object += 1
        else :
            count_free += 1

    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.1, 0.6, 0.8])
    Classes = ['Defects', 'Free']
    counts = [count_object, count_free]
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Binary Labels')
    bar = ax.bar(Classes,counts, width=0.3, color=['darkred','g'])
    ax.bar_label(bar, label_type='center', color='w')
    plt.show()


#______________________________________________________________________________________________________________________________________________
#explain:
#   draw histogram of pixels of maske for each class 
#atribiut:
#   annonation_path: path of annonations folder( jason files)
#   annonations_name: list of annonations name that we want calculate their histogram. if be None, histogram calculated for all annonations in directory
#   userPercent:????????????????/
#______________________________________________________________________________________________________________________________________________
def defect_pixels_hist(annonation_path , annonations_name=None, data_show_stat = 1 ):

    if annonations_name is None:
        annonations_name = os.listdir( annonation_path )
    annonations_name = list( filter( lambda x:x[-5:]=='.json' , annonations_name))
    

    def add_to_array(inp_dic, inp_dic_count , cls , count):
        if cls in inp_dic.keys():
            inp_dic[cls] += count
            inp_dic_count[cls] += 1
        
        elif cls not in inp_dic.keys():
            inp_dic[cls] = count
            inp_dic_count[cls] = 1

    buckets = {}
    class_count = {}

    for anot_name in annonations_name:
        annotation = Annotation(os.path.join(annonation_path , anot_name))
        if annotation.have_object() :
            mask_list = annotation.get_masks()
            for mask in mask_list:
                count = np.sum( mask.__coded_mask__[:, 1] )
                add_to_array(buckets, class_count , mask.class_id , count) 
    
    fig = plt.figure(edgecolor='k' , )
    ax = fig.add_axes([0.2, 0.15, 0.6, 0.7])

    if data_show_stat == 0:
        ax.set_ylabel('Pix Number Percent')
        total = sum(buckets.values())
        #print(total)
        for key , elm in buckets.items():
            buckets[key] /= (total * 0.01)

    elif data_show_stat == 1:
        ax.set_ylabel('Pix Number')
    
    elif data_show_stat == 2:
        ax.set_ylabel('Mask/Tootal Percent')
        total = sum(buckets.values())
        #print(total)
        for key , elm in buckets.items():
            buckets[key] /= (256*1600*class_count[key])* 0.01
        
    ax.set_xlabel('Class')
    ax.set_title('Class-Pixel Bar Chart')
    bar  = ax.bar(buckets.keys() , buckets.values())
    ax.bar_label(bar, color='k')

    plt.show()


#______________________________________________________________________________________________________________________________________________
#explain:
#   draw histogram of class labels 
#atribiut:
#   annonation_path: path of annonations folder( jason files)
#   annonations_name: list of annonations name that we want calculate their histogram. if be None, histogram calculated for all annonations in directory
#______________________________________________________________________________________________________________________________________________
def class_hist( annonation_path , annonations_name=None ):
    if annonations_name is None:
        annonations_name = os.listdir( annonation_path )
    all_classes = []
    for name in annonations_name:
        path = os.path.join( annonation_path, name) 
        anonation = Annotation(path)
        if anonation.have_object():
            all_classes.extend( list( anonation.get_classes() )) 
    all_classes = np.array( all_classes)
    classes_id , counts = np.unique( all_classes, return_counts=True)
    classes_id = list(classes_id)
    classes_id = list( map( lambda x: 'Class_' + str(x), classes_id))
    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.1, 0.6, 0.8])
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Class Labels')
    bar = ax.bar(classes_id,counts, width=0.3)
    ax.bar_label(bar, label_type='center', color='w')
    plt.show()


if __name__ == '__main__':
    csv_path = 'severstal-steel-defect-detection/train.csv'
    img_path = 'severstal-steel-defect-detection/train_images'
    annonation_path = 'severstal-steel-defect-detection/annotations'
    # class_hist(annonation_path=annonation_path, annonations_name= None)
    # binary_hist(annonation_path=annonation_path, annonations_name= None)
    defect_pixels_hist(annonation_path , data_show_stat=2)


