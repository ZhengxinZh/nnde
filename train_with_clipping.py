#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm

import tensorflow as tf
from tensorflow import keras



# clip model weights to a given hypercube
class ClipConstraint(keras.constraints.Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return keras.backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}








# # Input aligned as rows
# # sigmoid layer
# class Sig(keras.layers.Layer):
#     """y = phi(w.x + b)"""

#     def __init__(self, units=32, input_dim=32):
#         super(Sig, self).__init__()
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=(input_dim, units), dtype="float32"),
#             trainable=True,
#         )
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
#         )


#         beta_init = tf.random_normal_initializer()
#         self.beta = tf.Variable(
#             initial_value=beta_init(shape=(units,1), dtype="float32"),
#             trainable=True,
#         )
#         b0_init = tf.zeros_initializer()
#         self.b0 = tf.Variable(
#             initial_value=b0_init(shape=(1,), dtype="float32"), trainable=True
#         )



#     def call(self, inputs):
#         return tf.matmul( tf.math.sigmoid( tf.matmul(inputs, self.w) + self.b ) , self.beta ) + self.b0



# Instantiate our layer.
# sigmoid_layer = Sig(units=num, input_dim=dim)
# The layer can be treated as a function.
# Here we call it on some data.
# y = sigmoid_layer(tf.ones((2, 2)))
# assert y.shape == (2, 1)
# print(y)
# print(sigmoid_layer.weights)
# assert sigmoid_layer.weights == [sigmoid_layer.w, sigmoid_layer.b, sigmoid_layer.beta, sigmoid_layer.b0]




# class ActivityRegularization(keras.layers.Layer):
#     """Layer that creates an activity sparsity regularization loss."""

#     def __init__(self, rate=1e-2):
#         super(ActivityRegularization, self).__init__()
#         self.rate = rate

#     def call(self, inputs):
#         # We use `add_loss` to create a regularization loss
#         # that depends on the inputs.
#         self.add_loss(self.rate * tf.reduce_sum(inputs))
#         return inputs




# prepare dataset
dataset = np.load('1e6_2d_gaussian.npy')
# dataset = np.ones((10,1))


# dimension, number of neurons, diameter of weights
dim = np.shape(dataset)[1]
num = 10
diameter = 1

# constraint
constr = ClipConstraint(diameter)
# constr = keras.constraints.max_norm(diameter)

inputs = keras.Input(shape=(dim,))
x = keras.layers.Dense(num, activation='sigmoid', kernel_constraint=constr )(inputs)
outputs = keras.layers.Dense(1, kernel_constraint=constr)(x)
model = keras.Model(inputs, outputs)

model.add_loss( tf.abs(outputs) )
optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
# keras.optimizers.SGD(learning_rate=1e-3)
model.compile(optimizer)

# training
inputs = dataset
model.fit(dataset, batch_size=100, epochs=1)

print(dataset[1] )
print(model.predict( dataset[1:2] ))
print( tf.get_static_value( tf.reduce_mean(model.predict(dataset)) ) )




