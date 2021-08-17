import cv2
import tensorflow
from tensorflow import keras
import json
import os
import numpy as np

### Rename any method name as you fit

class ModelBuilder():
    
    def __read__(self , path):
        assert os.path.isfile(path) , f"The following path is not available! \n {path} "
        assert os.path.splitext(path)[-1] == '.json', f"The following file is not JSON!"

        with open(path , 'r') as file:
            json_file = json.load(file)

        return json_file


    def __init__(self , path):
        self.__json = self.__read__(path)

    def builder(self):


        ## Assigning a dictionary name (using the names listed in configsample.json) to each function
        generator_dict = {
            'c2c' : self._cnn2cnn,
            'c2d' : self._cnn2dense,
            'd2d' : self._dense2dense
        }

        output_type_list = ['reg' , 'cls' , 'bin']

        assert self.__json['model-type'] in generator_dict.keys() , 'Model Type Not Valid!'
        model_type = self.__json['model-type']
        generator = generator_dict[model_type]


        assert self.__json['output-type'] in output_type_list , 'Model\'s output type is Not Valid!'
        output_type = self.__json['output-type']
        
        input_shape = np.array(
            self.__json['input-dimension']
        )

        output_neuron = int( self.__json['output-neuron_count'] )

        return generator(input_shape , output_neuron , output_type)


    
    def _dense2dense(self ,  input_shape , output_neuron, output_type ):
        print('dense2dense: OK' , input_shape , output_neuron , output_type)

    def _cnn2dense(self, input_shape , output_neuron , output_type ):
        print('cnn2dense: OK', input_shape , output_neuron , output_type)
        

    def _cnn2cnn( self, input_shape , output_neuron , output_type ):
        print('cnn2cnn: OK', input_shape , output_neuron , output_type)
        

model_generator = ModelBuilder(r"model_config\config-test.json")
model_generator.builder()