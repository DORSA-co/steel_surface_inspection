import DataReader
import numpy as np
from matplotlib import pyplot as plt

def binary_hist(csv_path, img_path):
    csv = DataReader.csv_reader( csv_path )
    imgs_list,_ = DataReader.get_imgs_list(img_path,split=0)
    dict_lbl = DataReader.csv2labelDict(csv)
    binary_lbl,_ = DataReader.get_binary_labels( dict_lbl, imgs_list)

    count_all = len(binary_lbl)
    count_defects = np.sum(binary_lbl) #count 1
    count_free = count_all - count_defects

    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.1, 0.6, 0.8])
    Classes = ['Defects', 'Free']
    counts = [count_defects, count_free]
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Binary Labels')
    bar = ax.bar(Classes,counts, width=0.3, color=['darkred','g'])
    ax.bar_label(bar, label_type='center', color='w')
    plt.show()

    






csv_path = 'severstal-steel-defect-detection/train.csv'
img_path = 'severstal-steel-defect-detection/train_images'

binary_hist(csv_path, img_path)


stop