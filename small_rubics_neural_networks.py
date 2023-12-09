import numpy as np

import keras.layers as kl
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

from task_tree import task_tree_node
import small_rubics as rubics

class student_network:

    def __init__(self):

        self.input_size = 24*6
        self.epochs=1
        self.build_network()

    def build_network(self):

        self.neural_network = None

        self.residual_weights_reg = regularizers.l2(l2=0.0001)
        self.residual_bias_reg = regularizers.l2(0.001)
        self.relu_leak = 0.1
        self.residual_units = 100
        self.residual_layers = 0

        input = kl.Input(shape=self.input_size)
        
        #   kernel_regularizer=self.residual_weights_reg,bias_regularizer=self.residual_bias_reg
        x = kl.BatchNormalization()(input)
        x = kl.Dense(self.residual_units,kernel_regularizer=self.residual_weights_reg,bias_regularizer=self.residual_bias_reg)(x)
        x = kl.LeakyReLU(alpha=self.relu_leak)(x)
        x = kl.BatchNormalization()(x)

        for i in range(self.residual_layers):

            x = self.make_residual_layer(x)

        eval = kl.Dense(1,kernel_regularizer=self.residual_weights_reg)(x)
        eval = kl.Activation('sigmoid',name='eval_output')(eval)

        policy = kl.Dense(12,kernel_regularizer=self.residual_weights_reg,bias_regularizer=self.residual_bias_reg)(x)
        policy = kl.Activation('sigmoid',name='policy_output')(policy)

        model = Model(inputs= input,outputs = [eval,policy])

        opt = Adam(learning_rate=0.01)
        losses={'eval_output':'mean_squared_error','policy_output':'mean_squared_error'}

        model.compile(optimizer=opt, loss=losses,loss_weights=[3,1])

        self.neural_network = model

    def make_residual_layer(self, x:tf.Tensor):

        y = kl.Dense(self.residual_units,
            kernel_regularizer=self.residual_weights_reg,bias_regularizer=self.residual_bias_reg)(x)
        y = kl.LeakyReLU(alpha=self.relu_leak)(y)
        y = kl.BatchNormalization()(y)
        y = kl.Dense(self.residual_units,kernel_regularizer=self.residual_weights_reg,bias_regularizer=self.residual_bias_reg)(y)
        y = kl.BatchNormalization()(y)
        z = kl.Add()([x,y])
        z = kl.LeakyReLU(alpha=self.relu_leak)(z)
        return z

    def predict(self, task_nodes):

        #Each row has the form [state] (add context functionality later)
        nn_input = np.empty((len(task_nodes),self.input_size),dtype=float)

        t : task_tree_node
        for i, t in enumerate(task_nodes):
            
            nn_input[i] = rubics.make_neural_input(t.state)

        eval, policy = self.neural_network(nn_input)
        eval, policy = (eval.numpy(),policy.numpy())

        return nn_input, eval, policy

    def fit(self, inputs, evals, policies):

        self.neural_network.fit(x=inputs,y=[evals, policies],batch_size=16, epochs=self.epochs)
