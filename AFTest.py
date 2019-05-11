#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:18:37 2019

@author: rohrdr
"""

import numpy as np
from ActivationFunctions import Sigmoid, TanH, ReLU, leaky_ReLU, Softplus
from Tools import eval_err, grad_num
npoints = 4
nsamples = 3

###
#   some text snippets for the error messages
###
err1 = "Error in the get_activation"
err2 = " Function"


def test_suite():
    """
    Runs all the tests available:
        - SigmoidTest
        - TanHTest
        - ReLUTest
        - leaky_ReLUTest
        - SoftplusTest
    """
    
    x = np.random.randn(npoints, nsamples) * 0.01
    
    ret = sigmoid_test(x)
    
    print("The results of the SigmoidTest are")
    print(ret)
    
    ret = tanh_test(x)
    
    print("The results of the TanHTest are")
    print(ret)
    
    ret = relu_test(x)
    
    print("The results of the ReLUTest are")
    print(ret)
    
    ret = leaky_relu_test(x)
    
    print("The results of the leaky_ReLUTest are")
    print(ret)
    
    ret = softplus_test(x)
    
    print("The results of the Softplus are")
    print(ret)
    
    return


def sigmoid_test(x):

    def test_activation(x, sig):

        y = sig.get_activation(x)
        z = np.exp(x) / (1.0 + np.exp(x))
        
        errmsg = err1 + " function of the Sigmoid" + err2
    
        res = eval_err(z, y, errmsg)
        
        return res

    def test_derivative(x, sig):

        y = sig.get_activation_der(x)
        z = np.exp(x) / np.power(1.0 + np.exp(x), 2)
    
        errmsg = err1 + "_der function of the Sigmoid" + err2

        res = eval_err(z, y, errmsg)
    
        return res

    result = []
    
    sig = Sigmoid()
    
    result.append(test_activation(x, sig))
    result.append(test_derivative(x, sig))
    result.append(num_derivative(x, sig))
        
    return result 


def tanh_test(x):

    def test_activation(x, tan):
        
        y = tan.get_activation(x)
        z = np.sinh(x) / np.cosh(x)
        
        errmsg = err1 + " function of the TanH" + err2
    
        res = eval_err(z, y, errmsg)
        
        return res

    def test_derivative(x, tan):

        y = tan.get_activation_der(x)
        z = 1.0 / np.power(np.cosh(x), 2)
    
        errmsg = err1 + "_der function of the TanH" + err2

        res = eval_err(z, y, errmsg)
    
        return res

    result = []
    
    tan = TanH()
    
    result.append(test_activation(x, tan))
    result.append(test_derivative(x, tan))
    result.append(num_derivative(x, tan))
        
    return result 


def relu_test(x):
    
    def test_activation(x, re_l):
        
        y = re_l.get_activation(x)
        z = np.where(x <= 0.0, 0.0, x)
        
        errmsg = err1 + " function of the ReLU" + err2
    
        res = eval_err(z, y, errmsg)
        
        return res

    def test_derivative(x, re_l):

        y = re_l.get_activation_der(x)
        z = np.where(x <= 0.0, 0.0, 1.0)
    
        errmsg = err1 + "_der function of the ReLU" + err2

        res = eval_err(z, y, errmsg)
    
        return res

    result = []
    re_l = ReLU()
    
    result.append(test_activation(x, re_l))
    result.append(test_derivative(x, re_l))
    result.append(num_derivative(x, re_l))
        
    return result 


def leaky_relu_test(x):
    
    def test_activation(x, l_re_l):
        
        y = l_re_l.get_activation(x)
        z = np.where(x <= 0.0, 0.01 * x, x)
        
        errmsg = err1 + " function of the leaky_ReLU" + err2
    
        res = eval_err(z, y, errmsg)
        
        return res

    def test_derivative(x, l_re_l):

        y = l_re_l.get_activation_der(x)
        z = np.where(x <= 0.0, 0.01, 1.0)
    
        errmsg = err1 + "_der function of the leaky_ReLU" + err2

        res = eval_err(z, y, errmsg)
    
        return res
    
    result = []
    l_re_l = leaky_ReLU()
    
    result.append(test_activation(x, l_re_l))
    result.append(test_derivative(x, l_re_l))
    result.append(num_derivative(x, l_re_l))
        
    return result 


def softplus_test(x):
    
    def test_activation(x, l_re_l):
        
        y = l_re_l.get_activation(x)
        z = np.log(1.0) + np.log(1.0 + np.exp(x))
        
        errmsg = err1 + " function of the SoftPlus" + err2
    
        res = eval_err(z, y, errmsg)
        
        return res

    def test_derivative(x, l_re_l):

        y = l_re_l.get_activation_der(x)
        z = np.exp(x) / (1.0 + np.exp(x))
    
        errmsg = err1 + "_der function of the Softplus" + err2

        res = eval_err(z, y, errmsg)
    
        return res
    
    result = []
    sop = Softplus()
    
    result.append(test_activation(x, sop))
    result.append(test_derivative(x, sop))
    result.append(num_derivative(x, sop))
        
    return result    


def num_derivative(x, actfunc):
    
    y = actfunc.get_activation_der(x)
    
    nx = x.shape[0]
    ny = x.shape[1]
    
    new_x = x.reshape((nx * ny, 1))
    
    z = grad_num(new_x, actfunc.get_activation)
    
    new_z = z.reshape((nx, ny))
    
    errmsg = "error in the numerical gradient of" + str(actfunc)
    
    res = eval_err(new_z, y, errmsg)
    
    return res
    

if __name__ == '__main__':
    test_suite()
