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
from custom_layers.unit_regularizer import UnitRegularizer

from weighted_model import WeightedModel

from utils import dotdict, copy_network
import pickle

from task import task

# tf.debugging.enable_check_numerics()


@tf.function
def reward_loss(y_true: tf.Tensor, y_pred: tf.Tensor):

    log = tf.math.multiply(tf.abs(tf.math.log(y_pred)), y_true)
    square = tf.square(y_pred-y_true)
    return tf.reduce_mean(square*(1+log), axis=-1)


@tf.function
def crossentropy_loss(y_true: tf.Tensor, y_pred: tf.Tensor):

    return -tf.reduce_mean(tf.reduce_sum(y_true*tf.math.log(y_pred+1e-5), axis=-1), axis=-1)


class student_network:

    def __init__(self, params):

        task: "task" = params.task
        self.state_shape = task.state_shape
        self.one_hot_categories = task.one_hot_categories
        self.state_size = np.prod(self.state_shape)
        self.action_codes = task.n_actions

        self.params = params

        self.width = 32

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

        self.state_input = kl.Input(
            shape=self.state_shape+(self.one_hot_categories,), name="state_input")

        x = kl.GaussianNoise(0.03)(self.state_input)

        x = kl.Conv1D(filters=self.width*self.width, kernel_size=1, data_format="channels_first",
                      kernel_regularizer=UnitRegularizer(0, 0.01))(x)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.Reshape(target_shape=(self.width, self.width, self.one_hot_categories))(x)

        for _ in range(params.residual_layers):
            y = kl.Conv2D(filters=self.one_hot_categories, kernel_size=4, padding="same")(x)
            y = kl.LeakyReLU(0.05)(y)
            #y = kl.BatchNormalization()(y)
            y = kl.Conv2D(filters=self.one_hot_categories, kernel_size=4, padding="same")(y)
            y = kl.LeakyReLU(0.05)(y)
            #y = kl.BatchNormalization()(y)
            x = kl.Add()([x, y])

        self.core = x

    def build_value_network(self, params):

        x = self.core

        x = kl.Conv2D(filters=params.residual_filters, kernel_size=4, padding="same")(x)
        x = kl.LeakyReLU(0.05)(x)

        #x = kl.BatchNormalization()(x)

        y: tf.Tensor

        for _ in range(params.residual_layers):
            y = kl.Conv2D(filters=params.residual_filters, kernel_size=4, padding="same")(x)
            y = kl.LeakyReLU(0.05)(y)
            #y = kl.BatchNormalization()(y)
            y = kl.Conv2D(filters=params.residual_filters, kernel_size=4, padding="same")(y)
            y = kl.LeakyReLU(0.05)(y)
            #y = kl.BatchNormalization()(y)
            x = kl.Add()([x, y])

        x = kl.Conv2D(filters=6, kernel_size=2, padding="same")(y)
        x = kl.LeakyReLU(0.05)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Flatten()(x)

        value = kl.Dense(1)(x)
        value = kl.Activation('softplus', name='value_output')(value)

        reward = kl.Dense(self.action_codes, name='reward_output')(x)

        reward_confidence = kl.Dense(self.action_codes)(x)
        reward_confidence = kl.Activation(
            'softplus', name='reward_confidence_output')(reward_confidence)

        model = WeightedModel(inputs=self.state_input, outputs=[
                              value, reward, reward_confidence])

        opt = keras.optimizers.Adam(params.learning_rate, clipvalue=1)

        model.compile(optimizer=opt, run_eagerly=False)

        self.value_network = model

    def build_state_network(self, params):

        core_input = self.core
        x: tf.Tensor = core_input

        action_input = kl.Input(self.action_codes, name="action_input")

        y: tf.Tensor = kl.Dense(self.width*self.width*18,kernel_regularizer=UnitRegularizer(0,0.01))(action_input)
        y = kl.LeakyReLU(0.05)(y)
        y = kl.Reshape(target_shape=(self.width, self.width, 18))(y)

        x = kl.Concatenate(axis=-1)([x, y])

        x = kl.Conv2D(filters=16, kernel_size=4, padding="same")(x)
        x = kl.LeakyReLU(0.05)(x)

        #x = kl.BatchNormalization()(x)

        y: tf.Tensor

        for _ in range(params.residual_layers):
            y = kl.Conv2D(filters=16, kernel_size=4, padding="same")(x)
            y = kl.LeakyReLU(0.05)(y)
            #y = kl.BatchNormalization()(y)
            y = kl.Conv2D(filters=16, kernel_size=4, padding="same")(y)
            y = kl.LeakyReLU(0.05)(y)
            #y = kl.BatchNormalization()(y)
            x = kl.Add()([x, y])

        x = kl.Conv2D(filters=self.one_hot_categories,
                      kernel_size=1, padding="same")(x)
        x = kl.LeakyReLU(0.05)(x)
        #x = kl.BatchNormalization()(x)
        x = kl.Reshape(target_shape=(self.width*self.width, self.one_hot_categories))(x)
        next_state = kl.Conv1D(filters=self.state_size, kernel_size=1,
                               data_format="channels_first", kernel_regularizer=UnitRegularizer(1e-3,1e-3))(x)

        next_state = kl.Reshape(
            target_shape=self.state_shape+(self.one_hot_categories,))(next_state)
        next_state = kl.Softmax(name='state_output', axis=-1)(next_state)

        model = Model(inputs=[self.state_input,
                      action_input], outputs=next_state)
        opt = Adam(learning_rate=params.learning_rate, clipvalue=1)
        model.compile(optimizer=opt, loss=crossentropy_loss,
                      metrics=["accuracy"])

        self.state_network = model

    def make_residual_layer(self, x: tf.Tensor, params):

        y = kl.Dense(params.residual_units,
                     kernel_regularizer=params.residual_weights_reg, bias_regularizer=params.residual_bias_reg)(x)
        y = kl.ELU()(y)
        y = kl.BatchNormalization()(y)
        y = kl.Dense(params.residual_units, kernel_regularizer=params.residual_weights_reg,
                     bias_regularizer=params.residual_bias_reg)(y)
        y = kl.ELU()(y)
        y = kl.BatchNormalization()(y)
        z = kl.Concatenate()([x, y])
        return z

    def predict_value(self, input_state):

        value, reward, reward_confidence = self.value_network(input_state)
        value, reward, reward_confidence = (
            value.numpy(), reward.numpy(), reward_confidence.numpy())

        return value, reward, reward_confidence

    def fit_value(self, input_state, value, reward, reward_confidence, reward_weights, epochs=1):

        self.value_network.fit(x=input_state, y=[value, reward, reward_confidence],
                               sample_weight=reward_weights, batch_size=64, epochs=epochs, shuffle=True)

    def predict_state(self,  states, actions):

        next_states = self.state_network.predict(x=[states, actions])
        return next_states

    def fit_state(self, states, actions, next_states, epochs=1):

        self.state_network.fit(x=[states, actions], y=next_states, batch_size=64,
                               epochs=epochs, shuffle=True, validation_split=1/16)
