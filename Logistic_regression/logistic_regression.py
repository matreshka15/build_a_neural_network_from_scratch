#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


''' 
In order to implement logistic regression, we need to initialize parameters w and b, note that when using logistic regression, 
we have only one computing unit in the neural network. Therefore parameters can be initialized to all zeros. 
However, if we are using multiple computing units(e.g. sigmoid or relu), it is required to initialize the parameter w randomly, while
b can be set to all zeros.
'''
def logistic_parameter_initialize(x_dim):
    # x_dim indicates the dimensions of input feature,a bias unit b is defaultly set.
    w = np.zeros((1,x_dim))
    b = 0
    
    parameter = {'w':w,'b':b}
    return parameter

#implementation of sigmoid function
def sigmoid(z):
    value = 1/(1+np.exp(-z))
    return value

#plot Sigmoid function 
plt.figure(figsize=(4,3))
plt.title('Sigmoid function\'s shape')
z = np.linspace(-10,10)
plt.plot(z,sigmoid(z))

#note the dimensions of vectors:
# w - (1,x_dims)
# b - (1,x_dims) *broadcasted*
# X - (x_dims,m) *m=number of samples


# Forward propagation step: compute the predicted y's label
def forward_prop(w,b,X):
    z = np.dot(w,X)+b
    a = sigmoid(z)
    return z,a

# Compute cost function: used to check convergence
def compute_cost(a,y):
    m = a.shape[1]
    cost = -np.sum(y*np.log(a)+(1-y)*np.log(1-a))/m
    return cost

# Back propagation step: compute partial derivatives of each parameter respectively
def back_prop(X,a,y):
    m = a.shape[1]
    dz = a - y
    dw = np.dot(X,dz.T).T/m
    db= np.sum(dz)/m
    # Note: dw should have the same dimension as w have.Therefore back_prop return dw.T
    return dw,db

# The overall implementation of trainning a logistic regression
def train_logistic_regression(X,y,number_of_iteration = 1000,learning_rate = 0.03,print_cost = True,plot_cost = True):
    # Dimension convert: make sure all vectors are in proper shapes.
    y = y.reshape(1,-1)   # y is a row vector
    m = y.shape[1]  #  m = total number of trainning examples
    X = X.reshape(-1,m)
    x_dim = X.shape[0]
    
    params = logistic_parameter_initialize(x_dim)
    w = params['w']
    b = params['b']
    
    if(plot_cost == True):
        i_curve = []
        cost_curve = []
        plt.figure(figsize=(5,5))
        plt.title('Cross entrophy of regression')
    
    for i in range(1,number_of_iteration+1):
        z,a = forward_prop(w,b,X)
        dw,db = back_prop(X,a,y)
        w = w - learning_rate*dw
        b = b - learning_rate*db
        cost = compute_cost(a,y)
        # Visualize the process of regression
        if(i%100 == 0 and print_cost == True):
            print('number of iterations:{}, cost = {}'.format(i,cost))
        if(i%100 == 0 and plot_cost == True):
            i_curve.append(i)
            cost_curve.append(cost)
    if(plot_cost==True):        
        i_curve = np.reshape(i_curve,(1,-1))
        cost_curve = np.reshape(cost_curve,(1,-1))
        plt.scatter(i_curve,cost_curve)
    
    return w,b




