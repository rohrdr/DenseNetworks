#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:10:39 2019

@author: rohrdr
"""

import numpy as np
import CostFunctions as CF
from Tools import grad_num, eval_err


def test_suite():
    """
    Runs all the tests available:
        - Cross Entropy (cost and derivative)
        
    """
    
    n = 5
    m = 3
    
    costfunction = CF.CrossEntropy()
    
    y = np.abs(np.random.randn(n, m)) * 0.1
    yhat = np.abs(np.random.randn(n, m)) * 0.1
    
    cost = costfunction.get_loss(yhat, y)

    ders = []    
    for i in range(m):
        new_yhat = yhat[:, i].reshape(n, 1)
        new_y = y[:, i].reshape(n, 1)
        der = costfunction.get_loss_der(new_yhat, new_y)
        num = grad_num(new_yhat, costfunction.get_loss, new_y)
        err = eval_err(num, der, "error in the " + str(i) + "-th column of Y")
        if not err:
            print("iteration " + str(i))
            print("analytical derivative")
            print(der)
            print("numerical derivative")
            print(num)
        
        ders.append(der)

    errmsg = "error between derivative at once and one by one"
    der = costfunction.get_loss_der(yhat, y)
    ders = np.squeeze(ders).T
        
    err = eval_err(ders, der, errmsg)
            
    if not err:
        print("all at once")
        print(der)
        print("one by one")
        print(ders)
    
    return
