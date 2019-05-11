#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:37:05 2019

@author: rohrdr
"""

import numpy as np
import abc

class LossFunction(abc.ABC):
    """
    Abstract Class for LossFunctions
    """
    
    @abc.abstractmethod
    def __init__(self):
        
        return
    
    @abc.abstractmethod
    def _get_loss(self, Yhat, Y):
        
        
        return
    
    @abc.abstractmethod
    def _get_loss_der(self, Yhat, Y):
        
        return
    
    def check_arrays(self, Yhat, Y):
    
        assert( isinstance(Y, np.ndarray) )
        assert( (np.asarray(Y).dtype.kind == "f") or (np.asarray(Y).dtype.kind == "i") )
        
        assert( isinstance(Yhat, np.ndarray))
        assert( np.asarray(Yhat).dtype.kind == "f" )
        
        assert( Y.shape == Yhat.shape )
        assert( len(Y.shape) == 2)
        
        assert( np.all(Yhat >= 0.0) )
        assert( np.all(Yhat <= 1.0) )
    
        return
    
    
class CrossEntropy(LossFunction):
    """
    LossFunction CrossEntropy
    
    Z( y^\hat ) = - 1/N sum_i [y_i log[y^\hat_i] + (1 - y_i) log[1 - y^\hat_i]]

    where y^\hat_i is the prediction of the ith example and y_i is corresponding true value
    
    Z'( y^\hat ) = - ( y / y^\hat  - (1 - y) / (1 - y^\hat))
    
    where y^\hat is the prediction vector with all samples and y is the corresponding true values
    """
    
    def __init__(self):
        
        super().__init__()
        
        return
    
    def get_loss(self, Yhat, Y):
        
        self.check_arrays(Yhat, Y)
        
        N = Yhat.shape[1]
        Z = self._get_loss(Yhat, Y, N)
        
        return np.squeeze(Z)
    
    def _get_loss(self, Yhat, Y, N):
        
        Z = super()._get_loss(Yhat, Y)
        
        Z = - ( Y * np.log(Yhat) + (1.0 - Y) * np.log(1.0 - Yhat) ) / N
        
        return Z.sum(axis = 1, keepdims = True)
    
    def get_loss_der(self, Yhat, Y):
        
        self.check_arrays(Yhat, Y)
        
        Z = self._get_loss_der(Yhat, Y)
        
        return Z
    
    def _get_loss_der(self, Yhat, Y):
        
        Z = super()._get_loss(Yhat, Y)
        
        Z = - ( (Y / Yhat) - (1.0 - Y)/(1.0 - Yhat) )
        
        return Z
    
    
