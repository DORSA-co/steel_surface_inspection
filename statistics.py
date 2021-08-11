from typing import NewType
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

    






csv_path = 'severstal-steel-defect-detection/train.csv'
img_path = 'severstal-steel-defect-detection/train_images'
annonation_path = 'severstal-steel-defect-detection/annotations'
binary_hist(annonation_path=annonation_path, annonations_name= None)


stop