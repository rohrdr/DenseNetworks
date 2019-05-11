#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:05:56 2019

@author: rohrdr
"""
import numpy as np
import abc

class ActivationFunction(abc.ABC):
    """
    Abstract Class for ActivationFunctions
    """
    
    @abc.abstractmethod
    def __init__(self):
        
        return
    
    @abc.abstractmethod
    def _get_activation(self, X ):
        
        
        return
    
    @abc.abstractmethod
    def _get_activation_der(self, X):
        
        return
    
    def checkArray(self, X):
        
        assert( isinstance(X, np.ndarray))
        assert( np.asarray(X).dtype.kind == "f")
        
        return
    
class Sigmoid(ActivationFunction):
    """
    The Sigmoid Activation Class (child of Abstract Class Activation Class)
    
    g(X) = 1 / ( 1 + exp[ -X ])
    g'(X) = g(X) * (1 - g(X))
    
    """
    
    def __init__(self):
        
        super().__init__()
        
        return
    
    def get_activation(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation(X)
        
        return Y
    
    def _get_activation(self, X):
        
        super()._get_activation(X)
        
        return 1.0 / (1.0 + np.exp(-X))
    
    def get_activation_der(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation_der(X)
        
        return Y
    
    def _get_activation_der(self, X):
        
        super()._get_activation_der(X)
        
        value = self.get_activation(X)
        
        return value * (1 - value)
    
    
class TanH(ActivationFunction):
    """
    The TanH Activation Class (child of Abstract Class Activation Class)
    
    g(X) = tanh[X]
    g'(X) = 1 - tanh^2[X]
    
    """
    
    def __init__(self):
        
        super().__init__()
        
        return
    
    def get_activation(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation(X)
        
        return Y
    
    def _get_activation(self, X):
        
        super()._get_activation(X)
        
        return np.tanh(X)
    
    def get_activation_der(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation_der(X)
        
        return Y
    
    def _get_activation_der(self, X):
        
        super()._get_activation_der(X)
        
        return 1.0 - np.power(self.get_activation(X), 2)
    
 
class ReLU(ActivationFunction):
    """
    The ReLU Activation Class (child of Abstract Class Activation Class)
    
    g(X) = max(0,X)
    
    g'(X) = 1 if X > 0
            0 if X <= 0
    
    """
    
    def __init__(self):
        
        super().__init__()
        
        return
    
    def get_activation(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation(X)
        
        return Y
    
    def _get_activation(self, X):
        
        super()._get_activation(X)
        
        return np.where(X > 0.0, X, 0.0)
    
    def get_activation_der(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation_der(X)
        
        return Y
    
    def _get_activation_der(self, X):
        
        super()._get_activation_der(X)
        
        return np.where(X > 0.0, 1.0 , 0.0)
    

class leaky_ReLU(ActivationFunction):
    """
    The leaky_ReLU Activation Class (child of Abstract Class Activation Class)
    
    g(X) = X, if X >0
           0.01 * X, if X <= 0
           
    g'(X) = 1, if X > 0
            0.01m if X <= 0
    
    """
    
    def __init__(self):
        
        super().__init__()
        
        return
    
    def get_activation(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation(X)
        
        return Y
    
    def _get_activation(self, X):
        
        super()._get_activation(X)
        
        return np.where(X > 0.0, X, 0.01 * X)
    
    def get_activation_der(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation_der(X)
        
        return Y
    
    def _get_activation_der(self, X):
        
        super()._get_activation_der(X)
        
        return np.where(X > 0.0, 1.0 , 0.01)
    
    
    
class Softplus(ActivationFunction):
    """
    The Softplus Activation Class (child of Abstract Class Activation Class)
    
    g(X) = log[ 1 + exp[X] ]
    
    g'(X) = 1 / (1 + exp[-X])
    
    """
    
    def __init__(self):
        
        super().__init__()
        
        return
    
    def get_activation(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation(X)
        
        return Y
    
    def _get_activation(self, X):
        
        super()._get_activation(X)
        
        return np.log(1.0 + np.exp(X))
    
    def get_activation_der(self, X):
        
        self.checkArray(X)
        
        Y = self._get_activation_der(X)
        
        return Y
    
    def _get_activation_der(self, X):
        
        super()._get_activation_der(X)
        
        return 1.0 / (1.0 + np.exp(-X))