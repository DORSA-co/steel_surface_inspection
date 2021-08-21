import cv2
import numpy as np
from tensorflow import keras
import json
import ModelDeveloper
import DataReader
import augmention
import DataViewr
import Features


__loss_dict__ = { ModelDeveloper.BINARY: keras.losses.binary_crossentropy,
            ModelDeveloper.CLASSIFICATION: keras.losses.categorical_crossentropy,
            ModelDeveloper.REGRESSION: keras.losses.mse,
            ModelDeveloper.POSETIVE_REGRESSION: keras.losses.mse,
            ModelDeveloper.NORMAL_REGRESSION: keras.losses.mse

        }







class trainConfig():

    def __init__(self, path):
        self.config = self.__read__(path)

    def __read__(self, path):
        with open(path , 'r') as file:
            json_file = json.load(file)
        return json_file


    def get_epochs(self):
        assert 'epochs' in self.config.keys(), "epochs not exist in train config file"
        return int(self.config['epochs'])

    def get_batch_size(self):
        assert 'batch-size' in self.config.keys(), "batch-size not exist in train config file"
        return int(self.config['batch-size'])
    
    def get_imgs_path(self):
        assert 'train-data-path' in self.config.keys(), "train-data-path not exist in train config file"
        return self.config['train-data-path']
    
    def get_lbls_path(self):
        assert 'train-label-path' in self.config.keys(), "train-label-path not exist in train config file"
        return self.config['train-label-path']
    
    def get_learning_rate(self):
        assert 'learning-rate' in self.config.keys(), "learning-rate not exist in train config file"
        return float(self.config['learning-rate'])

    def get_validation_split(self):
        assert 'validation-split' in self.config.keys(), "validation-split not exist in train config file"
        return float(self.config['validation-split'])
    
    def get_model_config_path(self):
        assert 'model-conig-path' in self.config.keys(), "model-conig-path not exist in train config file"
        return self.config['model-conig-path']


    def get_checkpoint_path(self):
        assert 'checkpoint-path' in self.config.keys(), "checkpoint-path not exist in train config file"
        return self.config['checkpoint-path']
    
    def get_out_path(self):
        assert 'save-weights-path' in self.config.keys(), "save-weights-path not exist in train config file"
        return self.config['save-weights-path']

    def get_class_num(self):
        assert 'class-num' in self.config.keys(), "class-num not exist in train config file"
        return int(self.config['class-num'])
    
    
    

    











if __name__ == "__main__":
    path = 'train.json'
    train_config = trainConfig(path)


    model_developer = ModelDeveloper.ModelBuilder( train_config.get_model_config_path() )
    model = model_developer.build()
    model.compile( keras.optimizers.Adam( learning_rate= train_config.get_learning_rate ), loss= __loss_dict__[ model_developer.output_type ] )


    
    
    aug = augmention.augmention()
    if model_developer.output_type == ModelDeveloper.BINARY:
        extractor_func = DataReader.extact_binary()

    elif model_developer.output_type == ModelDeveloper.CLASSIFICATION:
        extractor_func = DataReader.extract_class(train_config.get_class_num(), False)

    
    featurs_extractor=[Features.get_hog(bin_n=25,split=2), Features.get_hoc(bin_n=25, split=2)]

    gen = DataReader.generator( train_config.get_lbls_path(), extractor_func, annonations_name=None, batch_size=train_config.get_batch_size(), aug=aug, rescale=255, resize=(128,800), featurs_extractor=featurs_extractor)
    DataViewr.Viewer(gen)
    pass




    