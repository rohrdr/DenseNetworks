#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:32:58 2019

@author: rohrdr
"""

import numpy as np
import DenseNetworks as DN
import ActivationFunctions as AF
import CostFunctions as CF

def TestSuite():
    
    layers = []
    firstlayer = []
    firstlayer.append(4)
    firstlayer.append(3)
    firstlayer.append(AF.ReLU())
    
    secondlayer = []
    secondlayer.append(3)
    secondlayer.append(1)
    secondlayer.append(AF.Sigmoid())


    layers.append(firstlayer)
    layers.append(secondlayer)
    
    myDN = DN.denseDN(layers, CF.CrossEntropy())
    
    X = np.random.randn(4,2)
    
    Y = np.array([1.0,0.0]).reshape(1,2)
    
    print ("X")
    print (X)
    print ("Y")
    print (Y)
    
    loss = myDN.get_loss(X,Y)
    
    print ("loss: " + str(loss))
    
    myDN.trainDN(X, Y, maxiter = 1000, print_frequency = 100)
    
    return