#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import tensorflow as tf
from tensorflow import keras

# Truncated Gaussian
# from scipy.stats import truncnorm
# fig, ax = plt.subplots(1, 1)
# a, b = 0.1, 2
# mean, var, skew, kurt = truncnorm.stats(a, b, moments='mvsk')
# x = np.linspace(truncnorm.ppf(0.01, a, b), truncnorm.ppf(0.99, a, b), 100)
# ax.plot(x, truncnorm.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='truncnorm pdf')
# plt.show()
# rv = truncnorm(a, b)
# r = truncnorm.rvs(a, b, size=10)
# print(r)

# Multivariate Normal
mean=(0,0)
cov=[[1,0],[0,3]]
r= np.random.multivariate_normal(mean,cov,1000000)
np.save('1e6_2d_gaussian_00',r)

# # print(r)
# x,y = r.T
# plt.scatter(x,y)
# plt.axis('equal')
# plt.show()

# s=np.load('1e6_2d_gaussian.npy')
# print(np.shape(s)[1])
# x,y = s.T
# plt.scatter(x,y)
# plt.axis('equal')
# plt.show()







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