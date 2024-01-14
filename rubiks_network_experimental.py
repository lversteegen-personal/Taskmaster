import numpy as np

import keras.layers as kl
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
import tensorflow as tf
import keras

from custom_layers.inverse_polynomial_liner_unit import IPLU
from keras.layers import ELU
from keras.layers import LeakyReLU

from weighted_model import WeightedModel

from task_tree import task_tree_node
import small_rubiks as rubiks

from utils import dotdict, copy_network
import pickle

# tf.debugging.enable_check_numerics()


@tf.function
def reward_loss(y_true: tf.Tensor, y_pred: tf.Tensor):

    log = tf.math.multiply(tf.abs(tf.math.log(y_pred)), y_true)
    square = tf.square(y_pred-y_true)
    return tf.reduce_mean(square*(1+log), axis=-1)


class student_network:

    def __init__(self, params):

        self.state_size = params.state_size
        self.action_codes = params.action_codes
        self.params = params

    def create(params):

        combined = student_network(params)
        combined.build_core(params.core_params)
        combined.build_value_network(params.value_network_params)
        combined.build_state_network(params.state_network_params)

        return combined

    def load(path):

        with open(path+"_params.obj", 'rb') as file:
            params = pickle.load(file)

        network = student_network.create(params)
        network.state_network.load_weights(path+"_state_network.ckpt")
        network.value_network.load_weights(path+"_value_network.ckpt")

        return network

    def save(self, path):

        with open(path+"_params.obj", 'wb') as file:
            pickle.dump(self.params, file)

        self.value_network.save(path+"_value_network.ckpt")
        self.state_network.save(path+"_state_network.ckpt")

        print(f"Network saved under {path}.")

    def build_core(self, params):

        self.state_input = kl.Input(shape=self.state_size, name="state_input")

        self.core = self.state_input

    def build_value_network(self, params):

        x = self.core
        x = kl.GaussianNoise(0.03)(x)

        x = kl.Dense(16*16*6,kernel_regularizer=regularizers.l1(1e-5))(x)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.Reshape(target_shape=(16, 16, 6))(x)
        
        x = kl.Conv2D(filters=64, kernel_size=2, padding="same")(x)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.BatchNormalization()(x)

        y: tf.Tensor

        for _ in range(8):
            y = kl.Conv2D(filters=64,kernel_size=2,padding="same")(x)
            y = kl.LeakyReLU(0.05)(y)
            y = kl.BatchNormalization()(y)
            x = kl.Add()([x,y])

        x = kl.Conv2D(filters=6,kernel_size=2,padding="same")(y)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Flatten()(x)

        value = kl.Dense(1)(x)
        value = kl.Activation('softplus', name='value_output')(value)

        reward = kl.Dense(self.action_codes)(x)
        reward = kl.Activation('softplus', name='reward_output')(reward)

        reward_confidence = kl.Dense(self.action_codes)(x)
        reward_confidence = kl.Activation('softplus', name='reward_confidence_output')(reward_confidence)

        model = WeightedModel(inputs=self.state_input, outputs=[
                              value, reward, reward_confidence])

        opt = Adam(params.learning_rate, clipvalue=1)

        model.compile(optimizer=opt, run_eagerly=False)

        self.value_network = model

    def build_state_network(self, params):

        core_input = self.core
        action_input = kl.Input(self.action_codes, name="action_input")
        x: tf.Tensor = kl.Concatenate()([core_input, action_input])

        x = kl.Dense(16*16*6,kernel_regularizer=regularizers.l1(1e-5))(x)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.Reshape(target_shape=(16, 16, 6))(x)
        
        x = kl.Conv2D(filters=32, kernel_size=2, padding="same")(x)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.BatchNormalization()(x)

        y: tf.Tensor

        for _ in range(3):
            y = kl.Conv2D(filters=32,kernel_size=2,padding="same")(x)
            y = kl.LeakyReLU(0.05)(y)
            y = kl.BatchNormalization()(y)
            x = kl.Add()([x,y])

        x = kl.Conv2D(filters=6,kernel_size=2,padding="same")(y)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Flatten()(x)

        next_state = kl.Dense(self.state_size,kernel_regularizer=regularizers.l1(1e-5))(x)
        next_state = kl.Reshape(target_shape=(54,6))(next_state)
        next_state = kl.Softmax(axis=-1)(next_state)
        next_state = kl.Flatten(name='state_output')(next_state)
        #next_state = kl.Activation('sigmoid',name='state_output')(next_state)

        model = Model(inputs=[self.state_input,action_input],outputs = next_state)
        opt = Adam(learning_rate=params.learning_rate)
        model.compile(optimizer=opt,loss="binary_crossentropy")

        self.state_network = model

    def make_residual_layer(self, x:tf.Tensor,params):

        y = kl.Dense(params.residual_units,
            kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        y = kl.ELU()(y)
        y = kl.BatchNormalization()(y)
        y = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(y)
        y = kl.ELU()(y)
        y = kl.BatchNormalization()(y)
        z = kl.Concatenate()([x,y])
        return z

    def predict_value(self, input_state):

        value, reward, reward_confidence = self.value_network(input_state)
        value, reward, reward_confidence = (value.numpy(),reward.numpy(),reward_confidence.numpy())

        return value, reward, reward_confidence

    def fit_value(self, input_state, value, reward, reward_confidence, reward_weights, epochs=1):

        self.value_network.fit(x=input_state,y=[value, reward, reward_confidence], sample_weight=reward_weights,batch_size=64, epochs=epochs,shuffle =True)

    def predict_state(self,  states, actions):

        next_states = self.state_network.predict(x=[states,actions])
        return next_states

    def fit_state(self, states, actions, next_states, epochs=1):

        self.state_network.fit(x=[states,actions],y=next_states,batch_size=64, epochs=epochs,shuffle =True,validation_split=1/16)