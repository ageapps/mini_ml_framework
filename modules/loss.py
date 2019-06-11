
# coding: utf-8

# # Mini deep-learning framework

# In[4]:


import numpy as np
import math
from . import LogManager

class LossMSE(object):
    def __init__(self, debug=False):
        self.y_t = np.array([])
        self.y_p = np.array([])
        self.L = np.array([])
        self.dL_dy = np.array([])
        self.m = 1
        self.logger = LogManager.getLogger(__name__, debug)

    def cost(self, predicted, target):
        """
        input is y_p and y_t
        output is L = MSE = 1/2m*(y_t-y_p)^2
        """ 
        
        self.m = target.shape[1]
        self.y_t = target
        self.y_p = predicted
        self.L = (1/(2*self.m))*np.sum(np.square(self.y_p-self.y_t))
        self.logger.debug('Cost | L:', self.L)
        return self.L
    
      
    def backward(self):
        """
        output is dL/dy = 2(y_p-y_t) = dL/da
        """ 
        
        self.dL_dy = (1/self.m)*(self.y_p-self.y_t)
        self.logger.debug('Backward | dL_dy:', self.dL_dy)
        return self.dL_dy
    
    
    def param(self):
        '''
        output is [L, dL/dw]
        '''
        
        return [(self.L,self.dL_dy)]

