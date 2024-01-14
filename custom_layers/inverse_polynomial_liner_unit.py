import tensorflow as tf
import keras
import numpy as np

@keras.saving.register_keras_serializable(package="MyLayers")
class IPLU(tf.keras.layers.Layer):
    def origin_identity(power:float=0.5, **kwargs)-> "IPLU":

        return IPLU(power, 1.0, 0.0, 1.0, 0.0, **kwargs)

    def positive(power: float = 0.5, slope: float = 1.0, **kwargs)-> "IPLU":

        intercept = slope /power
        return IPLU(power, 1.0, 0.0, slope, intercept, **kwargs)

    def intercept_positive(power: float = 0.5, intercept: float = 0.5, **kwargs)-> "IPLU":

        slope = intercept * power
        return IPLU(power, 1.0, 0.0, slope, intercept, **kwargs)

    def __init__(self, power: float, apex: float, switch: float, slope: float, intercept: float, **kwargs):

        super(IPLU, self).__init__(**kwargs)

        self.power = power
        self.slope = slope
        self.apex = apex
        self.intercept = intercept
        self.switch = switch

        self.left_offset = slope*switch + \
            intercept - slope*(apex - switch)/power
        self.left_factor = (apex-switch)**(power+1)*slope/power

        self.numerical_gap = apex-switch

    def call(self, inputs: tf.Tensor):
        y = tf.where(inputs < self.switch,
                     self.left_factor *
                     tf.pow(tf.maximum(self.apex-inputs,
                            self.numerical_gap), -self.power)+self.left_offset,
                     self.slope*inputs+self.intercept)
        return y

    def get_config(self):
        base_config = super().get_config()
        config = {
            "power": self.power,
            "apex": self.apex,
            "switch": self.switch,
            "slope": self.slope,
            "intercept": self.intercept
        }
        return {**base_config, **config}
