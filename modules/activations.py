
# coding: utf-8

# # Mini deep-learning framework

# In[4]:


import numpy as np
import math

class ReLu(object):
    def __init__(self):
        self.a = np.array([])
        self.dL_ds = np.array([])
           
    def forward (self, s):  
        """
        input should be s = wx+b
        output is a = f(s) = f(wx+b)
        """
        
        self.a = np.maximum(s, 0)
        return self.a
                   
    def backward (self, dL_da):
        """
        input should be dL/da=dL/dy
        output is dL/ds = df(s)/dw
        """
    
        #self.dL_ds = self.dL_ds * dL_da
        self.dL_ds = np.maximum(dL_da, 0)
        return self.dL_ds
    
    def update(self, params):
        pass
    
    def param(self):
        return [()]


# In[8]:


class Tanh(object):
    def __init__(self):
        self.a = np.array([])
        self.dL_ds = np.array([])
        
    def forward (self, s):
        """
        input should be s = wx+b
        output is a = f(s) = f(wx+b)
        """ 

        self.a = np.tanh(s)
        return self.a
    
    def backward (self, dL_da) :   
        """
        input should be dL/da=dL/dy
        output is dL/ds = df(s)/dw
        """
        
        self.dL_ds = 1.0 - np.square(np.tanh(dL_da))
        self.dL_ds = np.multiply(self.dL_ds, dL_da) 
        return self.dL_ds
    
    def update(self, params):
        pass
    
    def param(self):
        return [()]

