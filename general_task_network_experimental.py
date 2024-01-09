import numpy as np

import keras.layers as kl
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
import tensorflow as tf
import keras
from inverse_polynomial_linear_unit import IPLU
from keras.layers import ELU

from task_tree import task_tree_node
import small_rubiks as rubiks

from utils import dotdict, copy_network
import pickle 

#tf.debugging.enable_check_numerics()

#@tf.function
def reward_loss(y_true:tf.Tensor, y_pred:tf.Tensor):

    #e=0.01
    #y_true = tf.minimum(tf.maximum(y_true,e),1-e)
    #y_pred = y_pred * (1-e) + 0.5*e
    #mean = tf.reduce_mean((tf.math.log(1-y_pred)-tf.math.log(1-y_true))*(tf.math.log(y_true)-tf.math.log(y_pred)),axis=-1)

    y_true = tf.maximum(0.001,y_true)
    log = tf.math.multiply(tf.abs(tf.math.log(y_pred)), y_true)
    square = tf.square(y_pred-y_true)
    return tf.reduce_mean(square*(1+log),axis=-1)

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
            pickle.dump(self.params,file)

        self.value_network.save(path+"_value_network.ckpt")
        self.state_network.save(path+"_state_network.ckpt")

        print(f"Network saved under {path}.")

    def build_core(self,params):
        
        self.state_input = kl.Input(shape=self.state_size,name="state_input")

        self.core = self.state_input

    def build_value_network(self, params):

        x = self.core

        x = kl.Dense(2048,kernel_regularizer=regularizers.l2(0))(x)
        x = ELU()(x)
        x = kl.Dense(2048,kernel_regularizer=regularizers.l2(0))(x)
        x = ELU()(x)
        x = kl.Dense(1024,kernel_regularizer=regularizers.l2(0))(x)
        x = ELU()(x)

        y = kl.Dense(512,kernel_regularizer=regularizers.l2(0))(x)
        y = ELU()(y)

        value = kl.Dense(1,kernel_regularizer=regularizers.l2(0),bias_regularizer=regularizers.l2(0.01))(y)
        #value = kl.Activation('softplus',name='value_output')(value)
        value = IPLU(origin=False,name="value_output")(value)

        y = kl.Dense(512,kernel_regularizer=regularizers.l2(0))(x)
        y = ELU()(y)

        reward = kl.Dense(self.action_codes,kernel_regularizer=regularizers.l2(0),bias_regularizer=regularizers.l2(0.01))(y)
        #reward = kl.Activation('softplus',name='reward_output')(reward)
        reward = IPLU(origin=False,name='reward_output')(reward)

        reward_confidence = kl.Dense(self.action_codes)(y)
        #reward_confidence = kl.Activation('softplus',name='reward_confidence_output')(reward_confidence)
        reward_confidence = IPLU(origin=False,name='reward_confidence_output')(reward_confidence)

        model = Model(inputs= self.state_input,outputs = [value,reward,reward_confidence])

        opt = Adam(params.learning_rate)
        losses={'value_output':mean_squared_error,'reward_output':mean_squared_error,'reward_confidence_output':mean_squared_error}

        model.compile(optimizer=opt, loss=losses)

        self.value_network = model

    def build_state_network(self, params):

        core_input = self.core
        action_input = kl.Input(self.action_codes,name="action_input")
        x=kl.Concatenate()([core_input,action_input])

        x = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        x = kl.ELU()(x)
        x = kl.BatchNormalization()(x)

        for _ in range(params.post_core_residual_layers):

            x = self.make_residual_layer(x, params)

        next_state = kl.Dense(self.state_size)(x)
        next_state = kl.Activation('sigmoid',name='state_output')(next_state)

        model = Model(inputs=[self.state_input,action_input],outputs = next_state)
        opt = Adam(learning_rate=params.learning_rate,clipnorm=1)
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

    def fit_value(self, input_state, value, reward, reward_confidence, epochs=1):

        copy = copy_network(self.value_network)
        self.value_network.fit(x=input_state,y=[value, reward, reward_confidence],batch_size=64, epochs=epochs,shuffle =True)

        for w in self.value_network.trainable_weights:
            v = w.numpy()
            if np.any(np.abs(v)>1000) or np.any(np.isnan(v)):
                raise Exception("This shouldn't happen")

    def predict_state(self,  states, actions):

        next_states = self.state_network.predict(x=[states,actions])
        return next_states

    def fit_state(self, states, actions, next_states, epochs=1):

        self.state_network.fit(x=[states,actions],y=next_states,batch_size=64, epochs=epochs,shuffle =True,validation_split=1/16)