#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:55:22 2019

@author: rohrdr
"""

import numpy as np
import ActivationFunctions as AF
from Layers import Layer
from Tools import grad_num, eval_err
npoints = 4

def TestSuite():
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
    
    AFs = [AF.Sigmoid(), AF.TanH(), AF.ReLU(), AF.LeakyRelu(), AF.Softplus()]
    AFs = [AF.Sigmoid()]
    
    for af in AFs:
    
        Lay = Layer(n_x,n_y,af)
     
        X = np.random.randn(n_x,samples)
        Y = Lay.get_Y(X)

        dX = np.ones((n_y,samples))
        dA, dW, db = Lay.get_grad(dX)
        
        newX = X.reshape(n_x*samples, 1)
        num = grad_num(newX, TestX, Lay)
        err = eval_err(num, dA.reshape(n_x*samples,1), "error in X")
        
        if not err:
            print ("dA:     " + str(dA.shape))
            print (dA)
            print ("num dA: " + str(num.reshape(n_x,samples).shape))
            print (num.reshape(n_x,samples))
    
        newW = Lay.W.reshape(n_x*n_y,1)
        num = grad_num(newW, TestdW, Lay) / samples
        err = eval_err(num, dW.reshape(n_x*n_y,1), "error in W")

        if not err:
            print ("dW:     " + str(dW.shape))
            print (dW)
            print ("num dW: " + str(num.reshape(n_y,n_x).shape))
            print (num.reshape(n_y,n_x))

        newb = Lay.b
        num = grad_num(newb, Testdb, Lay) / samples
        err = eval_err(num, db, "error in b")

        if not err:
            print ("db:     " + str(db.shape))
            print (db)
            print ("num dA: " + str(num.shape))
            print (num)
    
    return

def TestdW(W, Lay):
    
    newW = W.reshape(Lay.ny, Lay.nx)
    
    Lay.W = newW
    
    return Lay.get_Y(Lay.X)

def Testdb(b, Lay):
        
    Lay.b = b
    
    return Lay.get_Y(Lay.X)

def TestX(X, Lay):
    
    newX = X.reshape(Lay.nx, int(X.shape[0] / Lay.nx))
    
    return Lay.get_Y(newX)