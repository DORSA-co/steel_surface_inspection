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
        self.model_type = self.__json['model-type']
        generator = generator_dict[self.model_type]


        assert self.__json['output-type'] in output_type_dict.keys() , 'Model\'s output type is Not Valid!'
        self.output_type = self.__json['output-type']
        self.__output_type = output_type_dict[self.output_type]
        
        self.input_shape = np.array(
            self.__json['input-dimension']
        )

        self.output_neuron = int( self.__json['output-neuron_count'] )

        return generator(self.input_shape , self.output_neuron , self.__output_type)


    
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

        model.add( keras.layers.Dense(256 , activation=None)) #or 64 and 64 as units
        model.add( keras.layers.BatchNormalization())
        model.add( keras.layers.ReLU())

        model.add( keras.layers.Dense(512 , activation=None)) #or 64 and 64 as units
        model.add( keras.layers.BatchNormalization())
        model.add( keras.layers.ReLU())

        model.add( keras.layers.Dense(128 , activation=None))
        model.add( keras.layers.BatchNormalization())
        model.add( keras.layers.ReLU())

        model.add( keras.layers.Dense(64 , activation=None))
        model.add( keras.layers.BatchNormalization())
        model.add( keras.layers.ReLU())

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
        '''
        inputs = keras.layers.Input(input_shape)
        conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = keras.layers.Dropout(0.5)(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = keras.layers.Dropout(0.5)(conv5)
        up6 = keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5))
        merge6 = keras.layers.concatenate([drop4,up6], axis = 3)
        conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        up7 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6))
        merge7 = keras.layers.concatenate([conv3,up7], axis = 3)
        conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        up8 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))
        merge8 = keras.layers.concatenate([conv2,up8], axis = 3)
        conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        up9 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))
        merge9 = keras.layers.concatenate([conv1,up9], axis = 3)
        conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
        model = keras.Model(inputs,  conv10)
        '''

        inputs = keras.layers.Input(input_shape)
        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = keras.layers.Dropout(0.5)(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

        out = keras.layers.Conv2D(output_neuron, 3, activation = output_type, padding = 'same', kernel_initializer = 'he_normal')(conv5)
        model = keras.Model(inputs,  out)
        return model
        
        conv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = keras.layers.Dropout(0.5)(conv5)

        up6 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5))
        merge6 = keras.layers.concatenate([drop4,up6], axis = 3)
        conv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6))
        merge7 = keras.layers.concatenate([conv3,up7], axis = 3)
        conv7 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        out = keras.layers.Conv2D(output_neuron, 3, activation = output_type, padding = 'same', kernel_initializer = 'he_normal')(conv8)

        model = keras.Model(inputs,  out)
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

        model.add( keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None))
        model.add( keras.layers.BatchNormalization())
        model.add( keras.layers.ReLU())
        model.add( keras.layers.AveragePooling2D( pool_size=(2,2)))

        model.add( keras.layers.Conv2D(64, kernel_size=(7,7), strides=(7,7), padding='valid', activation=None))
        model.add( keras.layers.BatchNormalization())
        model.add( keras.layers.ReLU())

        model.add( keras.layers.Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None))
        model.add( keras.layers.BatchNormalization())
        model.add( keras.layers.ReLU())
        model.add( keras.layers.MaxPooling2D( pool_size=(2,2)))

        model.add( keras.layers.GlobalAvgPool2D())
        model.add( keras.layers.Dense(128 , activation='relu'))
        model.add( keras.layers.Dense(64 , activation='relu'))
        model.add( keras.layers.Dense(output_neuron, activation=output_type))
        


        return model


if __name__ == '__main__':
    print('start')
    # model_builder = ModelBuilder()
    # model = model_builder.simple_cnn2dense( (300,300,3), 30, BINARY )
    model_builder = ModelBuilder(r'configs\config-model.json')
    model = model_builder.build()
    model.summary()