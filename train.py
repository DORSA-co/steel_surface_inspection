import cv2
import numpy as np
from tensorflow import keras
import json




class Trainer():



    def __init__(self, path):
        json_file = self.__read__(path)
        self.epochs = int(json_file['epochs'])
        self.batch_size = int(json_file['batch-size'])
        self.imgs_path = json_file['train-data-path']
        self.lbls_path = json_file['train-label-path']
        self.lr = float(json_file['learning-rate'])
        self.validations_split = float(json_file['validation-split'] )
        self.model_config_path = json_file['model-conig-path']
        self.checkpoint_path = json_file['checkpoint-path']
        self.out_path = json_file['save-weights-path']

        

    def __read__(self, path):
        with open(path , 'r') as file:
            json_file = json.load(file)
        return json_file








if __name__ == "__main__":
    path = 'train.json'
    train = Trainer(path)
    

    pass