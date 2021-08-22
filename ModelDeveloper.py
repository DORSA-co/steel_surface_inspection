import cv2
import tensorflow
from tensorflow import keras
import json
import os
import numpy as np
from tensorflow.python.keras.engine import keras_tensor

BINARY = 'sigmoid'
CLASSIFICATION = 'softmax'
POSETIVE_REGRESSION = 'relu'
REGRESSION = None
NORMAL_REGRESSION = 'tanh'

### Rename any method name as you fit

class ModelBuilder():
    
    def __read__(self , path):
        assert os.path.isfile(path) , f"The following path is not available! \n {path} "
        assert os.path.splitext(path)[-1] == '.json', f"The following file is not JSON!"

        with open(path , 'r') as file:
            json_file = json.load(file)

        return json_file

    ## This function will later utlize to grab the json data directly from LabView or any other program
    def __grab__(self):
        pass


    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   initializer. If a path is passed, the object will automaticaly tries to read the json the file from the path.
    #   if not, the # build function is not available # and user can use the model_generator functions directly.
    #
    #arg:
    #   path
    #       path: a valid path to config.json file
    #return:
    # 
    #______________________________________________________________________________________________________________________________________________    
    def __init__(self , path = None):
        if path is not None:
            self.__json = self.__read__(path)

    
    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   using the json data passed to the object, this method will call model_generator methods and return the coresponding model.
    #   current available model_generator functions are:
    #       simple_cnn2cnn
    #       simple_cnn2dense
    #       simple_dense2dense
    #
    #arg:
    #
    #return:
    #   model
    #______________________________________________________________________________________________________________________________________________    

    def build(self):

        ## Assigning a dictionary name (using the names listed in configsample.json) to each function
        ## the values will later be used directly to call model_generator functions
        generator_dict = {
            'c2c' : self.simple_cnn2cnn,
            'c2d' : self.simple_cnn2dense,
            'd2d' : self.simple_dense2dense
        }

        ## Assigning a dictionary name (using the variables listed in configsample.json) to each activation type
        output_type_dict = {
            'reg': REGRESSION,
            'cls': CLASSIFICATION,
            'bin': BINARY
        }

        assert self.__json['model-type'] in generator_dict.keys() , 'Model Type Not Valid!'
        model_type = self.__json['model-type']
        generator = generator_dict[model_type]


        assert self.__json['output-type'] in output_type_dict.keys() , 'Model\'s output type is Not Valid!'
        output_type_val = self.__json['output-type']
        self.output_type = output_type_dict[output_type_val]
        
        input_shape = np.array(
            self.__json['input-dimension']
        )

        output_neuron = int( self.__json['output-neuron_count'] )

        return generator(input_shape , output_neuron , self.output_type)


    
    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   build a model coresponds to args
    #arg:
    #   input_shape, output_neuron, output_type
    #   input_shape: shape of inputs in tuple format. the shape should be (h,w,channle)
    #   output_neuron: numbrer of channel in last layer. it show numner of output for each pixel in output array
    #   output_type: type of outbut that could be BINARY, CLASSIFICATION
    #
    #return:
    #   model
    #______________________________________________________________________________________________________________________________________________
    def simple_dense2dense(self ,  input_shape , output_neuron, output_type ):
        model = keras.Sequential()
        model.add( keras.layers.Input(shape = input_shape) )
        model.add( keras.layers.Dense(256 , activation='relu')) #or 64 and 64 as units
        model.add( keras.layers.Dense(128 , activation='relu'))
        model.add( keras.layers.Dense(output_neuron , activation=output_type))

        return model
        

    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   build a model coresponds to args
    #arg:
    #   input_shape, output_neuron, output_type
    #   input_shape: shape of inputs in tuple format. the shape should be (h,w,channle)
    #   output_neuron: numbrer of channel in last layer. it show numner of output for each pixel in output array
    #   output_type: type of outbut that could be BINARY, CLASSIFICATION, REGRESSION and so on
    #
    #return:
    #   model
    #______________________________________________________________________________________________________________________________________________
    def simple_cnn2cnn( self, input_shape , output_neuron , output_type ):
        model = keras.Sequential()
        model.add( keras.layers.Input(shape=input_shape))
        model.add( keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))

        model.add( keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.MaxPooling2D( pool_size=(2,2)))
        model.add( keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.MaxPooling2D( pool_size=(2,2)))
        model.add( keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.MaxPooling2D( pool_size=(2,2)))
        model.add( keras.layers.Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add( keras.layers.Conv2D(output_neuron, (3,3), padding='same', activation=output_type))

        return model


    #______________________________________________________________________________________________________________________________________________
    #explain:
    #   build a model coresponds to args
    #arg:
    #   input_shape, output_neuron, output_type
    #   input_shape: shape of inputs in tuple format. the shape should be (h,w,channle)
    #   output_neuron: numbrer of dense neuron in last layer. it show numner of output
    #   output_type: type of outbut that could be BINARY, CLASSIFICATION, REGRESSION and so on
    #
    #return:
    #   model
    #______________________________________________________________________________________________________________________________________________
    def simple_cnn2dense(self, input_shape , output_neuron , output_type ):
        model = keras.Sequential()
        model.add( keras.layers.Input(shape=input_shape))
        model.add( keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))

        model.add( keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.MaxPooling2D( pool_size=(2,2)))
        model.add( keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.MaxPooling2D( pool_size=(2,2)))
        model.add( keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.MaxPooling2D( pool_size=(2,2)))
        model.add( keras.layers.Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add( keras.layers.GlobalAvgPool2D())
        model.add( keras.layers.Dense(output_neuron, activation=output_type))
        return model


if __name__ == '__main__':
    print('start')
    # model_builder = ModelBuilder()
    # model = model_builder.simple_cnn2dense( (300,300,3), 30, BINARY )
    model_builder = ModelBuilder(r'model_config\config-test.json')
    model = model_builder.build()
    model.summary()