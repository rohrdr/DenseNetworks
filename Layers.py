#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:55:08 2019

@author: rohrdr
"""

import numpy as np
from ActivationFunctions import ActivationFunction

class Layer:
    """
    a dense layer for neural networks
    
    """
    
    def __init__(self, nx, ny, activation_function):
        """
        initialize layer with activation_function, feature dimension nx and output dimension ny
        
        """
        
        assert( isinstance(nx, np.int))
        assert( isinstance(ny, np.int))
        assert( isinstance(activation_function, ActivationFunction))
        
        self.nx = nx
        self.ny = ny
        self.W = np.random.randn(ny, nx) / np.sqrt(nx) # * 0.01
        self.b = np.zeros((ny, 1))
        self.AF = activation_function
        
        self.X = None
        self.Y = None
        self.Z = None
        
        return
    
    def get_Y(self, A):
        """
        calculate output vector Y given feature vector A
        
        """
        
        assert(isinstance(A,np.ndarray))
        assert(A.shape[0] == self.nx)
        m = A.shape[1]
        self.X = A
        
        Z = np.dot(self.W, self.X) + self.b
        assert (Z.shape == (self.ny, m))
        self.Z = Z
        Y = self.AF.get_activation(self.Z)
        assert (Y.shape == (self.ny, m))
        self.Y = Y
        
        return self.Y
    
    def get_grad(self, dA):
        """
        calculate dZ, dW and db given dA
        (compare notation of Andrew Ng in deeplearning.ai course on Coursera)
        
        """
        
        assert(isinstance(dA,np.ndarray))
        m = self.Z.shape[1]
        assert(dA.shape == self.Z.shape)
        
        dZ = dA * self.AF.get_activation_der(self.Z)
        dW = np.dot(dZ,self.X.T) / m
        db = np.sum(dZ, axis = 1, keepdims = True) / m
        
        return np.dot(self.W.T, dZ), dW, db
    
    def update_Wb(self, dW, db):
        """
        updates W und b
        
        """
        
        self.W += dW
        self.b += db
        
        return