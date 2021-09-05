## Initializing Metrics.py
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.metrics as metrics
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class BIN_Metrics():

    def __init__(self, threshold = 0.5):
        self.__threshold = threshold

    def True_Pos(self , y_true, y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(true , pred)
        metered = tf.cast(metered , tf.float16)

        all_ = tf.cast( 
            tf.reduce_sum(y_true) + tf.reduce_sum(1 - y_true),
            dtype=tf.float16
        )

        true_pos_count = tf.reduce_sum(metered)

        return tf.math.divide_no_nan(true_pos_count , all_)

                





    def True_Neg(self , y_true, y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(~true , ~pred)
        metered = tf.cast(metered , tf.float16)

        all_ = tf.cast( 
            tf.reduce_sum(y_true) + tf.reduce_sum(1 - y_true),
            dtype=tf.float16
        )

        true_neg_count = tf.reduce_sum(metered)

        return tf.math.divide_no_nan(true_neg_count , all_)




    def False_Pos(self , y_true, y_pred ):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(~true , pred)
        metered = tf.cast(metered , tf.float16)

        all_ = tf.cast( 
            tf.reduce_sum(y_true) + tf.reduce_sum(1 - y_true),
            dtype=tf.float16
        )

        false_pos_count = tf.reduce_sum(metered)

        return tf.math.divide_no_nan(false_pos_count , all_)


        

    def False_Neg(self , y_true, y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(true , ~pred)
        metered = tf.cast(metered , tf.float16)
        
        all_ = tf.cast( 
            tf.reduce_sum(y_true) + tf.reduce_sum(1 - y_true),
            dtype=tf.float16
        )

        false_neg_count = tf.reduce_sum(metered)
       

        return tf.math.divide_no_nan(false_neg_count , all_)
        

    def recall(self , y_true , y_pred):
        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        tp = self.True_Pos(y_true , y_pred)
        fn = self.False_Neg(y_true , y_pred)

        return tf.math.divide_no_nan(tp , (tp + fn) )



    def precision(self , y_true , y_pred):
        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        tp = self.True_Pos(y_true , y_pred)
        fp = self.False_Pos(y_true , y_pred)

        return tf.math.divide_no_nan(tp, (tp + fp) )

    
    def specificity(self, y_true , y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - self.__threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        metered = tf.logical_and(~true , ~pred)
        metered = tf.cast(metered , tf.float16)

        all_ = tf.cast( 
            tf.reduce_sum(1 - y_true),
            dtype=tf.float16
        )

        true_neg_count = tf.reduce_sum(metered)

        return tf.math.divide_no_nan(true_neg_count , all_)       








def iou_metric(class_id , threshold=0.5):
    def __iou__(y_true, y_pred):

        pred = tf.math.floor(
            (tf.math.sign( y_pred - threshold ) + 1) / 2
        )

        true = tf.cast(y_true , tf.bool)
        pred = tf.cast(pred , tf.bool)

        unity = tf.logical_and(true , pred)
        union = tf.logical_or(true , pred)

        unity = tf.cast(unity, dtype=tf.float32)
        union = tf.cast(union, dtype=tf.float32)

        unity_area = tf.reduce_sum(unity, axis=[-2, -3])
        union_area = tf.reduce_sum(union, axis=[-2, -3])

        iou = tf.math.divide_no_nan(unity_area , union_area)
        return tf.reduce_mean(iou, axis=0)[class_id ]
    return __iou__

                






'''
data_ = np.random.rand( 100, 8 )
print(data_)
label_ = np.sum(data_ , axis= 1)
print(label_)
model = Sequential()
model.add(Dense(20 , input_dim=8 , activation='relu'))
model.add(Dense(20 , activation='relu'))
model.add(Dense(20 , activation='relu'))
model.add(Dense(1 , activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[
        'accuracy' ,
        BIN_Metrics(0.7).False_Neg ,
        BIN_Metrics(0.7).False_Pos ,
        BIN_Metrics(0.7).True_Neg ,
        BIN_Metrics(0.7).True_Pos ,
        ])

model.fit(data_, label_, epochs=10, batch_size=1)
'''


# yt = [[0, 0, 0, 0],[1,1,0,0],[0,  0,0,  0],[1,   0,0,1]]
# yp = [[1, 1, 0, 1],[0,1,0,0],[0.9,0,0.2,0],[0.88,0,0,0]]
# yt = tf.constant(yt)
# yp = tf.constant(yp)

# binm = BIN_Metrics()
# print(binm.False_Neg( yt , yp))
# print(binm.False_Pos(yt , yp))
# print(binm.True_Neg(yt , yp))
# print(binm.True_Pos(yt , yp))







# class BIN_TruePos(Metric):

#     def __init__(self, name = 'Binary_true_positives' ,  **kwargs):

#         super(BIN_TruePos , self).__init__(name = name , **kwargs)