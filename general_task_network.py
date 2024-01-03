import numpy as np

import keras.layers as kl
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import keras

from task_tree import task_tree_node
import small_rubiks as rubiks

from utils import dotdict

class student_network:

    def __init__(self):

        self.state_size = 24*6
        self.action_codes = 12

    def create(params):

        combined = student_network()
        combined.build_core(params.core_params)
        combined.build_value_network(params.value_network_params)
        combined.build_state_network(params.state_network_params)

        return combined

    def build_core(self,params):
        
        self.state_input = kl.Input(shape=self.state_size,name="state_input")
        
        x = self.state_input
        x = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        x = kl.LeakyReLU(alpha=params.relu_leak)(x)
        x = kl.BatchNormalization()(x)

        for _ in range(params.residual_layers):

            x = self.make_residual_layer(x, params)

        self.core = x

    def build_value_network(self, params):

        x = self.core

        x = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        x = kl.LeakyReLU(alpha=params.relu_leak)(x)
        x = kl.BatchNormalization()(x)

        for _ in range(params.post_core_residual_layers):

            x = self.make_residual_layer(x, params)

        y=x
        for _ in range(params.value_fork_layers):

            y = self.make_residual_layer(y, params)

        y = kl.Dense(1,kernel_regularizer=params.residual_weights_reg)(y)
        value = kl.Activation('sigmoid',name='value_output')(y)

        y=x
        for _ in range(params.reward_fork_layers):

            y = self.make_residual_layer(y, params)

        reward = kl.Dense(self.action_codes)(y)
        reward = kl.Activation('sigmoid',name='reward_output')(reward)

        reward_confidence = kl.Dense(self.action_codes)(y)
        reward_confidence = kl.Activation('sigmoid',name='reward_confidence_output')(reward_confidence)

        model = Model(inputs= self.state_input,outputs = [value,reward,reward_confidence])

        opt = Adam(params.learning_rate)
        losses={'value_output':'mean_squared_error','reward_output':'mean_squared_error','reward_confidence_output':'mean_squared_error'}

        model.compile(optimizer=opt, loss=losses)

        self.value_network = model

    def build_state_network(self, params):

        core_input = self.core
        action_input = kl.Input(self.action_codes,name="action_input")
        x=kl.Concatenate()([core_input,action_input])

        x = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        x = kl.LeakyReLU(alpha=params.relu_leak)(x)
        x = kl.BatchNormalization()(x)

        for _ in range(params.post_core_residual_layers):

            x = self.make_residual_layer(x, params)

        next_state = kl.Dense(self.state_size)(x)
        next_state = kl.Activation('sigmoid',name='state_output')(next_state)

        model = Model(inputs=[self.state_input,action_input],outputs = next_state)
        opt = Adam(learning_rate=params.learning_rate)
        model.compile(optimizer=opt,loss="binary_crossentropy")

        self.state_network = model

    def make_residual_layer(self, x:tf.Tensor,params):

        y = kl.Dense(params.residual_units,
            kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        y = kl.LeakyReLU(alpha=params.relu_leak)(y)
        y = kl.BatchNormalization()(y)
        y = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(y)
        y = kl.LeakyReLU(alpha=params.relu_leak)(y)
        y = kl.BatchNormalization()(y)
        z = kl.Add()([x,y])
        return z

    def predict_value(self, input_state):

        value, reward, reward_confidence = self.value_network(input_state)
        value, reward, reward_confidence = (value.numpy(),reward.numpy(),reward_confidence.numpy())

        return value, reward, reward_confidence

    def fit_value(self, input_state, value, reward, reward_confidence, epochs=1):

        self.value_network.fit(x=input_state,y=[value, reward, reward_confidence],batch_size=64, epochs=epochs,shuffle =True)

    def predict_state(self,  states, actions):

        next_states = self.state_network.predict(x=[states,actions])
        return next_states

    def fit_state(self, states, actions, next_states, epochs=1):

        self.state_network.fit(x=[states,actions],y=next_states,batch_size=64, epochs=epochs,shuffle =True,validation_split=1/16)