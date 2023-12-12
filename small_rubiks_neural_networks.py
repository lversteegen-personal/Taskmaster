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

    def create(residual_layers):

        network = student_network()
        network.build_value_network(residual_layers)
        return network

    def save(self, path):

        self.value_network.save(path)
        print(f"Network saved under {path}.")

    def load_value_network(path):

        network = student_network()
        network.value_network = keras.models.load_model(path)
        return network

    def build_value_network(self, residual_layers):

        params = dotdict({
            "residual_weights_reg" : regularizers.l2(l2=0.001),
            "residual_bias_reg" : regularizers.l2(0.001),
            "relu_leak" : 0.1,
            "residual_units" : 100
        })

        input = kl.Input(shape=self.state_size)
        
        x = kl.BatchNormalization()(input)
        x = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        x = kl.LeakyReLU(alpha=params.relu_leak)(x)
        x = kl.BatchNormalization()(x)

        for i in range(residual_layers):

            x = self.make_residual_layer(x, params)

        eval = kl.Dense(1,kernel_regularizer=params.residual_weights_reg)(x)
        eval = kl.Activation('sigmoid',name='eval_output')(eval)

        policy = kl.Dense(12,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01))(x)
        policy = kl.Activation('sigmoid',name='policy_output')(policy)

        model = Model(inputs= input,outputs = [eval,policy])

        opt = Adam(learning_rate=0.002)
        losses={'eval_output':'mean_squared_error','policy_output':'mean_squared_error'}

        model.compile(optimizer=opt, loss=losses,loss_weights=[3,1])

        self.value_network = model

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

    def predict_value(self, task_nodes):

        #Each row has the form [state] (add context functionality later)
        nn_input = np.empty((len(task_nodes),self.state_size),dtype=float)

        t : task_tree_node
        for i, t in enumerate(task_nodes):
            
            nn_input[i] = rubiks.make_neural_input(t.state)

        eval, policy = self.value_network(nn_input)
        eval, policy = (eval.numpy(),policy.numpy())

        return eval, policy

    def fit_value(self, states, evals, policies, epochs=1):

        inputs = rubiks.make_neural_input(states)
        self.value_network.fit(x=inputs,y=[evals, policies],batch_size=16, epochs=epochs,shuffle =True)

    def predict_state(self,  states, actions):

        actions = (np.arange(12) == actions[...,None]).astype(float)
        states = rubiks.make_neural_input(states)
        next_states = self.state_network.predict(x=[states,actions])
        return next_states

    def fit_state(self, states, actions, next_states, epochs=1):

        actions = (np.arange(12) == actions[...,None]).astype(float)
        states = rubiks.make_neural_input(states)
        next_states = rubiks.make_neural_input(next_states)
        self.state_network.fit(x=[states,actions],y=next_states,batch_size=32, epochs=epochs,shuffle =True,validation_split=1/16)

    def build_state_network(self, residual_layers):

        params = dotdict({
            "residual_weights_reg" : regularizers.l2(l2=0.001),
            "residual_bias_reg" : regularizers.l2(0.001),
            "relu_leak" : 0.1,
            "residual_units" : 300,
            "learning_rate" : 0.001
        })

        state_input = kl.Input(self.state_size, name="state")
        action_input = kl.Input(12,name="action")

        input = kl.Concatenate()([state_input,action_input])
        
        x=input
        #x = kl.BatchNormalization()(x)
        x = kl.Dense(params.residual_units,kernel_regularizer=params.residual_weights_reg,bias_regularizer=params.residual_bias_reg)(x)
        x = kl.LeakyReLU(alpha=params.relu_leak)(x)
        #x = kl.BatchNormalization()(x)

        for i in range(residual_layers):

            x = self.make_residual_layer(x, params)

        next_state = kl.Dense(self.state_size)(x)
        next_state = kl.Activation('sigmoid',name='policy_output')(next_state)

        model = Model(inputs=[state_input,action_input],outputs = next_state)
        opt = Adam(learning_rate=params.learning_rate)
        model.compile(optimizer=opt,loss="binary_crossentropy")

        self.state_network = model

