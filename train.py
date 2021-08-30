import cv2
import numpy as np
from tensorflow import keras
import json
import ModelDeveloper
import DataReader
import augmention
import DataViewr
import Features
import  tensorflow as tf
import os
import statisticsDataset
import numpy as np
from deep_utils import metrics
from deep_utils import callbacks
np.seterr(divide='ignore', invalid='ignore')

#______________________________________________________________________________________________________________________________________
#
#______________________________________________________________________________________________________________________________________
cpu = tf.config.experimental.list_physical_devices('CPU')[0]
gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu)>0:
    gpu = gpu[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    #tf.config.experimental.set_visible_devices(cpu)
    print("GPU known")
else :
    print("GPU unknown")




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
    
    
    
class Preperations():
    

    def __init__(self, train_config: trainConfig):
        self.train_config = train_config

        self.model_developer = ModelDeveloper.ModelBuilder( train_config.get_model_config_path() )

        self.__loss_dict__ = { 
                'bin': keras.losses.binary_crossentropy,
                'cls': keras.losses.categorical_crossentropy,
                'reg': keras.losses.mse,
            }
        
    def prepareModel(
                    self,
                    optimizer : keras.optimizers.Optimizer,
                    print_summary = True
                    ):

        self.model = self.model_developer.build()

        if self.model_developer.output_type == 'cls':
            metric = self.__getClsMetrics()

        elif self.model_developer.output_type == 'bin':
            metric = self.__getBinMetrics()

        self.model.compile(
            optimizer(learning_rate= train_config.get_learning_rate),
            loss = self.__loss_dict__[ self.model_developer.output_type ],
            metrics = metric
        )

        if print_summary:
            self.model.summary()
            print(self.model.input_shape)




    def prepareData(self, augmentation : augmention , featurs_extractor: list, filter_args=None):
        
        if self.model_developer.output_type == 'bin' and ( self.model_developer.model_type in ['c2d', 'd2d']):
            extractor_func = DataReader.extact_binary()

        elif self.model_developer.output_type == 'cls' and ( self.model_developer.model_type in ['c2d', 'd2d']) :
            extractor_func = DataReader.extract_class(self.train_config.get_class_num(), False)
        
        elif self.model_developer.model_type in ['c2c']:
            extractor_func = DataReader.extract_mask( self.train_config.get_class_num(), mask_size=self.model.output_shape[1:-1], consider_no_object=False, class_id=None )

        
        if filter_args is None:
            annonations_name = DataReader.get_annonations_name(self.train_config.get_lbls_path())
        else:
            annonations_name = DataReader.filter_annonations(self.train_config.get_lbls_path(), filter_args )

        trains_list, val_list = DataReader.split_annonations_name(annonations_name, split=self.train_config.get_validation_split())

        print('training on {} data and validation on {} data'.format(len(trains_list), len(val_list)))
        

        #statisticsDataset.binary_hist(train_config.get_lbls_path(), trains_list)
        #statisticsDataset.binary_hist(train_config.get_lbls_path(), val_list)

        self.train_gen = DataReader.generator( 
                                    self.train_config.get_lbls_path(),
                                    extractor_func,
                                    annonations_name=trains_list,
                                    batch_size=self.train_config.get_batch_size(),
                                    aug = augmentation,
                                    rescale=255, resize=(128,800),
                                    featurs_extractor=featurs_extractor
                                    )
        
        self.val_gen = DataReader.generator( 
                                    self.train_config.get_lbls_path(),
                                    extractor_func,
                                    annonations_name=val_list,
                                    batch_size=self.train_config.get_batch_size(),
                                    aug=None,
                                    rescale=255, resize=(128,800),
                                    featurs_extractor=featurs_extractor
                                    )

        self.meta = {'trainsize': len(trains_list) , 'valsize': len(val_list)}



    def startFitting(self, load_weights = False, model_name = None):
        
        if model_name is not None:
            model_path = os.path.join( self.train_config.get_out_path(), model_name)
        
        else:
            model_path = os.path.join( 
                self.train_config.get_out_path(),
                f'Model_{self.model_developer.model_type}_{self.model_developer.output_type}.h5'
                )

        if load_weights:
                self.model.load_weights(model_path)


        callback = callbacks.CustomCallback( model_path )
        self.model.fit(  
            self.train_gen,
            validation_data = self.val_gen,
            batch_size= self.train_config.get_batch_size(),
            epochs = self.train_config.get_epochs(),
            steps_per_epoch = self.meta['trainsize'] // self.train_config.get_batch_size(),
            validation_steps = self.meta['valsize'] // self.train_config.get_batch_size(),
            callbacks = [callback]
            )


    def __getBinMetrics(self):
        return [
            'acc',
            metrics.BIN_Metrics().specificity,
            metrics.BIN_Metrics().recall,
            metrics.BIN_Metrics().precision,
        ]

    def __getClsMetrics(self):
        return [
            'This is where you put the metrics'
        ]











if __name__ == "__main__":
    path = 'train.json'
    train_config = trainConfig(path)

    aug = augmention.augmention(shift_range=(-100, 100),
                    rotation_range=(-10,10),
                    zoom_range=(0.9,1.1),
                    shear_range=(-0.05,0.05),
                    hflip=True, 
                    wflip = True, 
                    color_filter=True,
                    chance= 0.5 )
    
    featurs_extractor = [ 
        Features.get_hog(bin_n=25, split_h=2, split_w=4),
        Features.get_hoc(bin_n=25, split_h=2, split_w=4)
        ]# Features.get_hog(bin_n=25,split_h=1,split_w=4) ]

    filter_args = {
                "included_object": ["YES"]
                }

    prep = Preperations(train_config)

    prep.prepareModel( optimizer = keras.optimizers.RMSprop, print_summary = True )

    prep.prepareData(augmentation = None , featurs_extractor = None, filter_args=filter_args)

    viewer = DataViewr.Viewer(prep.train_gen)
    
    prep.startFitting(load_weights=False)

    

    '''
    train_gen = DataReader.generator( train_config.get_lbls_path(),
                                extractor_func,
                                annonations_name=trains_list,
                                batch_size=train_config.get_batch_size(),
                                aug=None,
                                rescale=255, resize=(128,800),
                                featurs_extractor=None
                                )

    DataViewr.ViewerByModel(train_gen, model, th=0.5)
    '''