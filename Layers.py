#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:55:08 2019

@author: rohrdr
"""

import numpy as np
from ActivationFunctions import ActivationFunction
from ActivationFunctions import invertableActivationFunction


class Layer:
    """
    a dense layer for neural networks
    
    """
    
    def __init__(self, nx, ny, activation_function):
        """
        initialize layer with activation_function, feature dimension nx and output dimension ny
        
        """
        
        assert(isinstance(nx, np.int))
        assert(isinstance(ny, np.int))
        assert(isinstance(activation_function, ActivationFunction))
        
        self.nx = nx
        self.ny = ny
        self.W = np.random.randn(ny, nx) / np.sqrt(nx)  # * 0.01
        self.b = np.zeros((ny, 1))
        self.AF = activation_function
        
        self.X = None
        self.Y = None
        self.Z = None
        
        return
    
    def get_y(self, a):
        """
        calculate output vector Y given feature vector a
        
        """
        
        assert(isinstance(a, np.ndarray))
        assert(a.shape[0] == self.nx)
        m = a.shape[1]
        self.X = a
        
        z = np.dot(self.W, self.X) + self.b
        assert (z.shape == (self.ny, m))
        self.Z = z
        y = self.AF.get_activation(self.Z)
        assert (y.shape == (self.ny, m))
        self.Y = y
        
        return self.Y
    
    def get_grad(self, d_a):
        """
        calculate dZ, dW and db given d_a
        (compare notation of Andrew Ng in deeplearning.ai course on Coursera)
        
        """
        
        assert(isinstance(d_a, np.ndarray))
        m = self.Z.shape[1]
        assert(d_a.shape == self.Z.shape)
        
        d_z = d_a * self.AF.get_activation_der(self.Z)
        d_w = np.dot(d_z,self.X.T) / m
        d_b = np.sum(d_z, axis = 1, keepdims = True) / m
        
        return np.dot(self.W.T, d_z), d_w, d_b
    
    def update_wb(self, d_w, d_b):
        """
        updates W und b
        
        """
        
        self.W += d_w
        self.b += d_b
        
        return


class invertableLayer(Layer):

    def __init__(self, nx, ny, activation_function):

        assert(isinstance(activation_function, invertableActivationFunction))

        super().__init__(self, nx, ny, activation_function)

        return