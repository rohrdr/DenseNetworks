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
    def _get_loss(self, yhat, y):

        return
    
    @abc.abstractmethod
    def _get_loss_der(self, yhat, y):
        
        return

    @staticmethod
    def check_arrays(self, yhat, y):
    
        assert(isinstance(y, np.ndarray))
        assert((np.asarray(y).dtype.kind == "f") or (np.asarray(y).dtype.kind == "i"))
        
        assert(isinstance(yhat, np.ndarray))
        assert(np.asarray(yhat).dtype.kind == "f")
        
        assert(y.shape == yhat.shape)
        assert(len(y.shape) == 2)
        
        assert(np.all(yhat >= 0.0))
        assert(np.all(yhat <= 1.0))
    
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
    
    def get_loss(self, yhat, y):
        
        self.check_arrays(yhat, y)
        
        n = yhat.shape[1]
        z = self._get_loss(yhat, y, n)
        
        return np.squeeze(z)
    
    def _get_loss(self, yhat, y, n):
        
        z = super()._get_loss(yhat, y)
        
        z = - (y * np.log(yhat) + (1.0 - y) * np.log(1.0 - yhat)) / n
        
        return z.sum(axis=1, keepdims=True)
    
    def get_loss_der(self, yhat, y):
        
        self.check_arrays(yhat, y)
        
        z = self._get_loss_der(yhat, y)
        
        return z
    
    def _get_loss_der(self, yhat, y):
        
        z = super()._get_loss(yhat, y)
        
        z = - ((y / yhat) - (1.0 - y) / (1.0 - yhat))
        
        return z

