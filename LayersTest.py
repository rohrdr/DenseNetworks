#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:55:22 2019

@author: rohrdr
"""

import numpy as np
import ActivationFunctions as afuncs
from Layers import Layer
from Tools import grad_num, eval_err


def test_suite():
    """
    Runs all the tests available:
        - checks the derivative dA
        - checks the derivative dW
        - checks the derivative db
        ...
        all of the above for the Activation Functions
        - Sigmoid
        - TanH
        - ReLU
        - LeakyRelu
        - Softplus
    """
    n_x = 7
    n_y = 4
    samples = 3
    
    afs = [afuncs.Sigmoid(), afuncs.TanH(), afuncs.ReLU(), afuncs.LeakyRelu(), afuncs.Softplus()]
    
    for af in afs:

        res = True

        lay = Layer(n_x, n_y, af)
     
        x = np.random.randn(n_x, samples)
        y = lay.get_y(x)

        d_x = np.ones((n_y, samples))
        d_a, d_w, d_b = lay.get_grad(d_x)
        
        new_x = x.reshape(n_x*samples, 1)
        num = grad_num(new_x, test_x, lay)
        err = eval_err(num, d_a.reshape(n_x*samples, 1), "error in X")
        
        if not err:
            res = False
            print("dA:     " + str(d_a.shape))
            print(d_a)
            print("num dA: " + str(num.reshape(n_x, samples).shape))
            print(num.reshape(n_x, samples))
    
        new_w = lay.W.reshape(n_x*n_y, 1)
        num = grad_num(new_w, testd_w, lay) / samples
        err = eval_err(num, d_w.reshape(n_x*n_y, 1), "error in W")

        if not err:
            res = False
            print("dW:     " + str(d_w.shape))
            print(d_w)
            print("num dW: " + str(num.reshape(n_y, n_x).shape))
            print(num.reshape(n_y, n_x))

        new_b = lay.b
        num = grad_num(new_b, testd_b, lay) / samples
        err = eval_err(num, d_b, "error in b")

        if not err:
            res = False
            print("db:     " + str(d_b.shape))
            print(d_b)
            print("num dA: " + str(num.shape))
            print(num)

    if res: print('All tests ran successfully')

    return res


def testd_w(w, lay):
    
    new_w = w.reshape(lay.ny, lay.nx)
    
    lay.W = new_w
    
    return lay.get_y(lay.X)


def testd_b(b, lay):
        
    lay.b = b
    
    return lay.get_y(lay.X)


def test_x(x, lay):
    
    new_x = x.reshape(lay.nx, int(x.shape[0] / lay.nx))
    
    return lay.get_y(new_x)


if __name__ == '__main__':
    test_suite()
