## Initializing Metrics.py
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.metrics as metrics
import numpy as np


class BIN_Metrics():

    def __init__(self, threshold = 0.8):
        self.__threshold = threshold

    def BIN_TruePos(self , y_true, y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(true , pred)

        metered = tf.cast(metered , tf.int32)
        return tf.reduce_sum(metered)


    def BIN_TrueNeg(self , y_true, y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(~true , ~pred)

        metered = tf.cast(metered , tf.int32)
        return tf.reduce_sum(metered)


    def BIN_FalsePos(self , y_true, y_pred ):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(~true , pred)

        metered = tf.cast(metered , tf.int32)
        return tf.reduce_sum(metered)
        

    def BIN_FalseNeg(self , y_true, y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(true , ~pred)
        metered = tf.cast(metered , tf.int32)

        return tf.reduce_sum(metered)


# yt = [[0, 0, 0, 0],[1,1,0,0],[0,  0,0,  0],[1,   0,0,1]]
# yp = [[1, 1, 0, 1],[0,1,0,0],[0.9,0,0.2,0],[0.88,0,0,0]]
# yt = tf.constant(yt)
# yp = tf.constant(yp)

# binm = BIN_Metrics()
# print(binm.BIN_FalseNeg( yt , yp))
# print(binm.BIN_FalsePos(yt , yp))
# print(binm.BIN_TrueNeg(yt , yp))
# print(binm.BIN_TruePos(yt , yp))







# class BIN_TruePos(Metric):

#     def __init__(self, name = 'Binary_true_positives' ,  **kwargs):

#         super(BIN_TruePos , self).__init__(name = name , **kwargs)