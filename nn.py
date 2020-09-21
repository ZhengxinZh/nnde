#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm

import tensorflow as tf
from tensorflow import keras







# Input aligned as rows
class Sig(keras.layers.Layer):
    """y = phi(w.x + b)"""

    def __init__(self, units=32, input_dim=32):
        super(Sig, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )


        beta_init = tf.random_normal_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=(units,1), dtype="float32"),
            trainable=True,
        )
        b0_init = tf.zeros_initializer()
        self.b0 = tf.Variable(
            initial_value=b0_init(shape=(1,), dtype="float32"), trainable=True
        )



    def call(self, inputs):
        return tf.matmul( tf.math.sigmoid( tf.matmul(inputs, self.w) + self.b ) , self.beta ) + self.b0




# Instantiate our layer.
sigmoid_layer = Sig(units=4, input_dim=2)

# The layer can be treated as a function.
# Here we call it on some data.
y = sigmoid_layer(tf.ones((2, 2)))
assert y.shape == (2, 1)

print(y)
print(sigmoid_layer.weights)

assert sigmoid_layer.weights == [sigmoid_layer.w, sigmoid_layer.b, sigmoid_layer.beta, sigmoid_layer.b0]










# def f(x):
#     return (x*np.cosh(x)-np.sinh(x));

# def leastsquare(v1,v2,size):
#     A = 0;
#     B1 = 0;
#     B2 = 0;
#     C = 0;
#     D = 0;
#     for x in v1:
#         for y in v2:
#             A = A + x*y;
#             B1 = B1 + x;
#             B2 = B2 + y;
#             C = C + x*x;
#             D = D + x;
#     return ((A - B1*B2/size)/(C - D*D/size));
    
# def L2norm2(coor,num,h,M):
#     L=float(coor*M*coor.T) + 1/3 -2/(np.e*np.sinh(1))+(np.sinh(2)-2)/(4*np.sinh(1)*np.sinh(1));
#     for i in range(1,num):
#         L=L-2*coor[0,i-1]*(i*h*h+(i-1)*(np.cosh(i*h)-np.cosh((i-1)*h))/np.sinh(1)-(i+1)*(np.cosh((i+1)*h)-np.cosh(i*h))/np.sinh(1)+(f((i+1)*h)+f((i-1)*h)-2*f(i*h))/(h*np.sinh(1)));
#     return L;
# def H1norm2(coor,num,h,M,K):
#     H1=float(coor*K*coor.T) + (2+np.sinh(2))/(4*np.sinh(1)*np.sinh(1))-1;
#     for i in range(1,num):
#         H1=H1-2*coor[0,i-1]*(np.sinh((i+1)*h)+np.sinh((i-1)*h)-2*np.sinh(i*h))/(h*np.sinh(1));
#     return (H1+L2norm2(coor,num,h,M));



# eL2 = np.zeros(8);
# eH1 = np.zeros(8);
# dn = np.linspace(-2*np.log(2),-9*np.log(2),8);

# for n in range(2,10):
#     num = 2**n;
#     h = 1/num;
#     u = np.zeros(num-1);
#     v = np.zeros(num+1);
#     dt = np.linspace(0,1,num+1)
#     F = np.matrix(np.zeros(num-1));
#     K = np.matrix(np.zeros((num-1,num-1)));
#     M = np.matrix(np.zeros((num-1,num-1)));
#     u1=np.zeros(num+1);
#     for i in range(num+1):
#         u1[i] = i*h-np.sinh(i*h)/np.sinh(1);

#     K[0,0]=2;
#     K[0,1]=-1;
#     K[num-2,num-3]=-1;
#     K[num-2,num-2]=2;
#     for i in range(1,num-2):
#         K[i,i-1] = -1;
#         K[i,i+1] = -1;
#         K[i,i] = 2;
#     K = K/h;
#     M[0,0]=2/3;
#     M[0,1]=1/6;
#     M[num-2,num-3]=1/6;
#     M[num-2,num-2]=2/3;
#     for i in range(1,num-2):
#         M[i,i-1] = 1/6;
#         M[i,i+1] = 1/6;
#         M[i,i] = 2/3;
#     M = M*h;

#     for i in range(num-1):
#         F[0,i] = (i+1)*h*h;
#     u = np.array(((K+M).I * F.T).T)[0];
#     v[1:num]=u;
#     eL2[n-2] = np.log(np.sqrt(L2norm2(np.matrix(u),num,h,M)));
#     eH1[n-2] = np.log(np.sqrt(H1norm2(np.matrix(u),num,h,M,K)));


# plt.plot(dn, eL2, label = 'L2 error')
# plt.plot(dn, eH1, label = 'H1 error')
# plt.legend(loc=0)
# pic = plt.gcf()
# pic.savefig('convergence order.png')
# plt.show()

# kL2 = leastsquare(dn,eL2,5)
# print('The L2 convergence order is', kL2)

# kH1 = leastsquare(dn,eH1,5)
# print('The H1 convergence order is', kH1)

#for x in eL2:
#    print(x)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#X1, X2 = np.meshgrid(dx1, dx2)
#surf = ax.plot_surface(X1, X2, u1, label = 'n = 64',linewidth=0, antialiased=False)
#pic = plt.gcf()
#pic.savefig('4a.png')
#plt.show()