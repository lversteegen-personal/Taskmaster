import keras
import tensorflow as tf

     
class UnitRegularizer(keras.regularizers.Regularizer):

    def __init__(self, row_strength=0.01, column_strength=0.01):
        self.row_strength = row_strength
        self.column_strength = column_strength

    def __call__(self, x):
        return self.row_strength * tf.reduce_mean(tf.square(tf.reduce_sum(tf.abs(x), axis=-1))) + \
            self.column_strength * tf.reduce_mean(tf.square(tf.reduce_sum(tf.abs(x), axis=-2)))

    def get_config(self):
        return {'row_strength': self.row_strength,
                'column_strength': self.column_strength
                }
