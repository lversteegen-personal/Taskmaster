import tensorflow as tf
import keras

class TorusConv3D(keras.layers.Layer):

    def __init__(self, filters:int, kernel_size, **kwargs):

        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_x = kernel_size[0]
        self.kernel_y = kernel_size[1]
        self.kernel_z = kernel_size[2]

        self.output_shape = self.kernel_size+self.filters

        self.kernel_initializer = keras.initializers.get("glorot_uniform")
        self.bias_initializer = keras.initializers.get("zeros")

    def build(self:"TorusConv3D", input_shape):

        channels=input_shape[-1]
        
        self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_size+(self.filters, channels),
            initializer=self.kernel_initializer
        )
        self.bias = self.add_weight(
            name="bias",
            shape=self.filters,
            initializer=self.bias_initializer
        )

        self.built = True

    def call(self, inputs:tf.Tensor):

        input_shape = inputs.shape
        result = tf.zeros(shape=((input_shape[0],)+self.output_shape))

        for x in range(self.kernel_x):
            for y in range(self.kernel_y):
                for z in range(self.kernel_z):
                    result += tf.tensordot(inputs,self.kernel[x,y,z,:],axes=[[4],[1]])
                    inputs = tf.roll(inputs,shift=1,axis=3)
                inputs = tf.roll(inputs,shift=1,axis=2)
            inputs = tf.roll(inputs,shift=1,axis=1)

        result += self.bias

        return result
