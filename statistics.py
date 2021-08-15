import DataReader
import numpy as np
from matplotlib import pyplot as plt
from DataReader import Annotation
import os

def binary_hist(annonation_path, annonations_name=None):

    if annonations_name is None:
        annonations_name = os.listdir( annonation_path )

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


def defect_pixels_hist(annonation_path , userPercent = False , img_size = (256 * 1600)):

    annonation_names = os.listdir(annonation_path)

    def add_to_array(inp_dic , cls , count):
        if cls in inp_dic.keys():
            inp_dic[cls] += count
        
        elif cls not in inp_dic.keys():
            inp_dic[cls] = count

    buckets = {}

    for anot_name in annonation_names:
        # print(os.path.join(annonation_path , anot_name))
        annotation = Annotation(os.path.join(annonation_path , anot_name))
        if annotation.have_object() :
            mask_list = annotation.get_encoded_mask()
            for mask in mask_list:
                count = np.sum( mask.codedMask_[:, 1] )
                add_to_array(buckets , mask.class_ , count) 

    print(buckets)
    if userPercent:
        total = sum(buckets.values())
        print(total)
        for key , elm in buckets.items():
            buckets[key] /= (total * 0.01)


    plt.bar(buckets.keys() , buckets.values())

    plt.xlabel('Class')
    plt.ylabel('Pix Number')
    plt.title('Class-Pixel Bar Chart')
    # plt.show()



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
    ax.set_title('Histogram of Binary Labels')
    bar = ax.bar(classes_id,counts, width=0.3)
    ax.bar_label(bar, label_type='center', color='w')
    # plt.show()


csv_path = 'severstal-steel-defect-detection/train.csv'
img_path = 'severstal-steel-defect-detection/train_images'
annonation_path = 'severstal-steel-defect-detection/annotations'
# class_hist(annonation_path=annonation_path, annonations_name= None)
# binary_hist(annonation_path=annonation_path, annonations_name= None)
defect_pixels_hist(annonation_path , userPercent=True)