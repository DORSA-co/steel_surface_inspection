import cv2
import tensorflow
from tensorflow import keras
from tensorflow.python.keras.engine import keras_tensor

### Rename any method name as you fit
BINARY = 'sigmoid'
CLASSIFICATION = 'softmax'
POSETIVE_REGRESSION = 'relu'
REGRESSION = None
NORMAL_REGRESSION = 'tanh'

class ModelInitializer():
    
    def __read__(self , path):
        pass

    def __init__(self):
        pass
    def __parse_conf(self , path):
        pass

    def creator(self):
        pass
    
    def simple_dense2dense(self ,  input_shape , output_neuron, output_type ):
        pass

    
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





if __name__ == '__main__':
    print('start')
    modelinit = ModelInitializer()
    model = modelinit.cnn2dense( (300,300,3), 30, BINARY )
    model.summary()
    