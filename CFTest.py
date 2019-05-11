#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:10:39 2019

@author: rohrdr
"""

import numpy as np
import CostFunctions as CF
from Tools import grad_num, eval_err

def TestSuite():
    """
    Runs all the tests available:
        - Cross Entropy (cost and derivative)
        
    """
    
    N = 5
    m = 3
    
    costfunction = CF.CrossEntropy()
    
    Y = np.abs(np.random.randn(N,m)) * 0.1
    Yhat = np.abs(np.random.randn(N,m)) * 0.1
    
    cost = costfunction.get_loss(Yhat, Y)

    ders = []    
    for i in range(m):
        newYhat = Yhat[:,i].reshape(N,1)
        newY = Y[:,i].reshape(N,1)
        der = costfunction.get_loss_der(newYhat, newY)
        num = grad_num(newYhat, costfunction.get_loss, newY)
        err = eval_err(num, der, "error in the " + str(i) + "-th column of Y")
        if not err:
            print ("iteration " + str(i))
            print ("analytical derivative")
            print (der)
            print ("numerical derivative")
            print (num)
        
        ders.append(der)


    errmsg = "error between derivative at once and one by one"
    der = costfunction.get_loss_der(Yhat, Y)
    ders = np.squeeze(ders).T
        
    err = eval_err(ders, der, errmsg)
            
    if not err:
        print ("all at once")
        print (der)
        print ("one by one")
        print (ders)
    
    return
