#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 09:16:42 2019

@author: rohrdr
"""
from numpy.linalg import norm
import numpy as np

threshold = 1e-08
eps = 1e-08

def eval_err(Z, Y, errmsg):
    
    result = True
    
    error = norm(Z-Y)
    error = np.squeeze(error)
    
    if error > threshold:
        
        print (errmsg)
        print ("error = " + str(error))
        result = False
    
    return result


def grad_num(X, func, *args, **kwargs):
    
    assert(isinstance(X, np.ndarray))
    assert(X.shape[1] == 1)
    n = X.shape[0]
    
    Y = func(X, *args, **kwargs)
    grad = np.zeros((n,1))
    
    for i in range(n):
        
        X[i] += eps
        Y2 = func(X, *args, **kwargs)
        grad[i] += np.sum(Y2-Y).T/ eps
        
        X[i] -= 2*eps
        Y2 = func(X, *args, **kwargs)
        grad[i] += np.sum(Y-Y2).T/ eps
        
        X[i] += eps
        
    return grad / 2
        