import tensorflow as tf
import keras
import numpy as np

@keras.saving.register_keras_serializable(package="MyLayers")
class IPLU(tf.keras.layers.Layer):
  def __init__(self, power=1.0, origin=True, **kwargs):
    super(IPLU, self).__init__(**kwargs)
    self.power = power
    self.factor = 1/power
    self.origin = origin
    if origin:
      self.left_offset = -self.factor
      self.right_offset = 0
    else:
      self.left_offset = 0
      self.right_offset = self.factor

  def call(self, inputs:tf.Tensor):
    y = tf.where(inputs < 0, self.factor*tf.pow(tf.abs(1-inputs)+0.001,-self.power)+self.left_offset,inputs+self.right_offset)
    return y

  def get_config(self):
        base_config = super().get_config()
        config = {
            "power": self.power,
            "origin": self.origin
        }
        return {**base_config, **config}