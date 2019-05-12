#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:32:58 2019

@author: rohrdr
"""

import numpy as np
import DenseNetworks as dn
import ActivationFunctions as af
import CostFunctions as cf
from Tools import eval_err


def test_suite():

    np.random.seed(1)

    layers = list()
    firstlayer = list()
    firstlayer.append(4)
    firstlayer.append(3)
    firstlayer.append(af.ReLU())
    
    secondlayer = list()
    secondlayer.append(3)
    secondlayer.append(1)
    secondlayer.append(af.Sigmoid())

    layers.append(firstlayer)
    layers.append(secondlayer)
    
    my_dn = dn.DenseNN(layers, cf.CrossEntropy())
    
    x = np.random.randn(4, 2)
    
    y = np.array([1.0, 0.0]).reshape(1, 2)
    
    print("X")
    print(x)
    print("Y")
    print(y)
    
    loss = my_dn.get_loss(x, y)
    
    print("loss: " + str(loss))
    
    my_dn.train_dn(x, y, maxiter=1000, print_frequency=100)

    y_target = np.array([[0.98678318, 0.16073323]])
    yhat = my_dn.forward_propagation(x)

    res = eval_err(y_target, yhat, errmsg='error in train_dn')

    if res: print('All tests ran successfully')
    
    return


if __name__ == '__main__':
    test_suite()
