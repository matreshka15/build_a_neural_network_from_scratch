#!/usr/bin/env python
# coding: utf-8

#Note: This is a visualized version of neural networks with multiple layers lib.
import numpy as np
import matplotlib.pyplot as plt

''' 
In order to implement neural networks, which turns out that we are using multiple computing units(e.g. sigmoid or relu), 
it is required to initialize the parameter w randomly, whileb can be set to all zeros.
'''

# nn_structure is a tuple, indicating the number of layers and units in each layer.
# for example, nn_structure = [4,4,3,1] indicates a neural network with 1 input layer, 2 hidden layers and one output unit.
def nn_parameter_initialize(nn_structure):
    # x_dim indicates the dimensions of input feature,a bias unit b is defaultly set.
    parameter = {}
    # Note: for ReLu unit, we use He initialization to faster convergence; for sigmoid in the output layer, do Xavier initialization. 
    for i in range(1,len(nn_structure)-1):
        parameter['w'+str(i)] = np.random.randn(nn_structure[i],nn_structure[i-1])*np.sqrt(2./nn_structure[i-1])
        parameter['b'+str(i)] = np.zeros((nn_structure[i],1))
    L = len(nn_structure)-1
    parameter['w'+str(L)] = np.random.randn(nn_structure[L],nn_structure[L-1])*np.sqrt(1./nn_structure[L-1])
    parameter['b'+str(L)] = np.zeros((nn_structure[L],1))    
    return parameter


# In[388]:


#implementation of sigmoid function
def sigmoid(z):
    value = 1/(1+np.exp(-z))
    return value

def relu(z):
    value = np.maximum(0,z)
    return value

def derivative_of_activation(a,activation='sigmoid'):
    derivative = 0
    shape = a.shape
    if(activation == 'sigmoid'):
        derivative = sigmoid(a)*(1-sigmoid(a))
    elif(activation == 'relu'):
        derivative = a>0
        derivative = derivative.astype(np.int32)
    return derivative.reshape(shape)


# note the dimensions of vectors:
# w - (number_of_units_in_current_layer,number_of_units_in_previous_layer)
# b - (number_of_units_in_current_layer,1) *broadcasted*
# X - (x_dims,m)                           *m=number of samples

'''
Note that in neural networks with multiple layers, when doing forward prop, we need to cache some values for back prop step.
For example:
    when doing forward prop, we calculate z = np.dot(w,X)+b ,and then output a = g(z), when function g can be relu or sigmoid.
    when doing back prop, we calculate dz = da*g'(z), where g'(z) is the derivative of g(z)
    Therefore we can just cache z to save some time for calculating z multiple times.
    
In a nutshell, what we can cache in forward prop are: z,a,w
'''

# Forward propagation step: compute the predicted y's label
def linear_forward_prop(w,b,X):
    z = np.dot(w,X) + b
    linear_cache = w
    #print('Linear_forward_z=',z)
    return z,linear_cache

def single_layer_forward_prop(z,activation='relu'):
    if(activation == 'relu'):
        a = relu(z)
    elif(activation == 'sigmoid'):
        a = sigmoid(z)
    activation_cache = a,z
    return a,activation_cache

def L_layer_forward_prop(X,parameters):
    A = {0:X}
    cache = {}  
    L = len(parameters) // 2
    for i in range(L-1):
        w = parameters['w'+str(i+1)]
        b = parameters['b'+str(i+1)]
        z,linear_cache = linear_forward_prop(w,b,A[i])
        a,activation_cache = single_layer_forward_prop(z,activation='relu')
        A[i+1] = a
        cache['layer_'+str(i+1)] = linear_cache,activation_cache
    w = parameters['w'+str(L)]
    b = parameters['b'+str(L)]    
    z,linear_cache = linear_forward_prop(w,b,A[L-1])
    yhat,activation_cache = single_layer_forward_prop(z,activation='sigmoid')
    cache['layer_'+str(L)] = linear_cache,activation_cache
    return yhat,cache


# Compute cost function: used to check convergence
def compute_cost(yhat,y):
    m = yhat.shape[1]
    cost = -np.sum(y*np.log(yhat)+(1-y)*np.log(1-yhat))/m
    return cost

# Back propagation step: compute partial derivatives of each parameter respectively
def linear_back_prop(w,a_previous,dz_l):
    m = a_previous.shape[1]
    dw_l = np.dot(dz_l,a_previous.T)/m
    assert(dw_l.shape == w.shape)
    
    db = np.sum(dz_l,axis=1,keepdims=True)/m
    da_previous = np.dot(w.T,dz_l)
    assert(da_previous.shape == a_previous.shape)
    return dw_l,db,da_previous
 # Note: dw should have the same dimension as w have.Therefore back_prop returns dw.T
    
def single_layer_back_prop(da_l,z_l,activation):
    derivative = derivative_of_activation(z_l,activation)
    dz_l = da_l * derivative
    assert(dz_l.shape == z_l.shape)
    return dz_l

def L_layer_back_prop(X,y,cache_from_forward,testAL = np.zeros(1)):
    dW = {}
    db = {}
    dA = {}
    W = {}
    A = {0:X}
    Z = {}
    m = y.shape[1]
    L=len(cache_from_forward)
    for layer in range(1,L+1):
        linear_cache_l,activation_cache_l = cache_from_forward['layer_'+str(layer)]
        W[layer] = linear_cache_l
        A[layer],Z[layer] = activation_cache_l
    #Initialize the output layer
    if((testAL == np.zeros(1))):
        yhat = A[L]
    else:
        yhat = testAL
    dA[L] = -np.divide(y,yhat)+np.divide((1-y),(1-yhat))
    dz_L = single_layer_back_prop(dA[L],Z[L],activation='sigmoid')
    dW[L],db[L],dA[L-1] = linear_back_prop(W[L],A[L-1],dz_L)
    
    for i in reversed(range(1,L)):
        dz_l = single_layer_back_prop(dA[i],Z[i],activation='relu')
        dW[i],db[i],dA[i-1] = linear_back_prop(W[i],A[i-1],dz_l)
        assert(dW[i].shape == W[i].shape)
    return dW,db

def update_parameters(params,learning_rate,dW,dB):
        L = len(params) // 2
        for j in range(1,L+1):
            params['w'+str(j)] = params['w'+str(j)] - learning_rate*dW[j]
            params['b'+str(j)] = params['b'+str(j)] - learning_rate*dB[j]
        return params

def dimension_convert(X,y):
    # Dimension convert: make sure all vectors are in proper shapes.
    y = y.reshape(1,-1)# y is a row vector
    M = y.shape[1]  #  m = total number of trainning examples
    if(X.shape[1] != M):
        X=X.T       #=====> Note that array.reshape and array.T are different!
    return X,y


# The overall implementation of training a logistic regression
# Note: net_structure indicates the shape of hidden layers and output layers. No input layer should be included.

def train_neural_network(X,y,net_structure,number_of_iteration = 1000,learning_rate = 0.03,lambd = 0.7,print_cost = True,plot_cost = True):
    # Dimension convert: make sure all vectors are in proper shapes.
    X,y = dimension_convert(X,y)
    print('*******Dimension Check*******')
    print('Input feature\'s dimension: ',X.shape)
    print('Output\'s dimension: ',y.shape)
    print('*****************************')
    
    #Normalize input feature X
    miu = np.sum(X)/(X.shape[0]*X.shape[1])
    X = X - miu
    sigma2 = np.sum(X**2)/(X.shape[0]*X.shape[1])
    sigma = np.sqrt(sigma2)
    X = X / sigma
    
    x_dim = X.shape[0]
    L = len(net_structure) # number of layers
    # Initialize parameters
    nn_structure = [x_dim]+net_structure
    params = nn_parameter_initialize(nn_structure)
    print('Training {} layers neural network...'.format(len(params)//2))
    if(plot_cost == True):
        i_curve = []
        cost_curve = []
        plt.figure(figsize=(5,5))
        plt.title('Cross entropy')
    
    cache={}
    for i in range(1,number_of_iteration+1):
            # Steps:
                # 1:Forward propagation
                # 2:Compute cost
                # 3:Backward Propagation
                # 4:Update parameters
                
        yhat,cache = L_layer_forward_prop(X,params)
        assert(yhat.shape == y.shape)
        cost = compute_cost(yhat,y)
        dW,dB = L_layer_back_prop(X,y,cache)
            
        #Gradient decent
        params = update_parameters(params,learning_rate,dW,dB)
            
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
    return params,miu,sigma



#After training the neural network, we can now use it to make predictions.
def nn_predict(parameters,miu,sigma,X,y=0,evaluate = True):
    L = len(parameters) // 2
    w1 = parameters['w1'] 
    num_of_features = w1.shape[1]
    if(X.shape[0] != num_of_features):
        X=X.T       #=====> Note that array.reshape and array.T are different!
    X = X - miu
    X = X / sigma
    yhat,cache = L_layer_forward_prop(X,parameters)
    yhat = yhat>0.5
    #Codes below is used to evaluate the performance 
    #You can just ignore this part
    if(evaluate == True):
        y=y.reshape(1,-1)
        train_accuracy = np.sum(yhat==y)/y.shape[1]
        print('accuracy = %.2f\n'%train_accuracy)
    return yhat





