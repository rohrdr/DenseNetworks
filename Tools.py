#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 09:16:42 2019

@author: rohrdr
"""
from numpy.linalg import norm
import numpy as np

threshold = 1e-07
eps = 1e-08


def eval_err(z, y, errmsg):
    
    result = True
    
    error = norm(z - y)
    error = np.squeeze(error) / np.linalg.norm(z)
    
    if error > threshold:
        
        print(errmsg)
        print("error = " + str(error))
        result = False
    
    return result


def grad_num(x, func, *args, **kwargs):
    
    assert(isinstance(x, np.ndarray))
    assert(x.shape[1] == 1)
    n = x.shape[0]
    
    y = func(x, *args, **kwargs)
    grad = np.zeros((n, 1))
    
    for i in range(n):
        
        x[i] += eps
        y2 = func(x, *args, **kwargs)
        grad[i] += np.sum(y2 - y).T / eps
        
        x[i] -= 2 * eps
        y2 = func(x, *args, **kwargs)
        grad[i] += np.sum(y - y2).T / eps
        
        x[i] += eps
        
    return grad / 2
