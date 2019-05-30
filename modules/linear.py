
# coding: utf-8

# # Mini deep-learning framework

# In[4]:


import numpy as np
import math

# ## Modules

# In[5]:

class Linear(object):

    def __init__(self, in_size, out_size, bias=True):
        self.in_size = in_size
        self.out_size = out_size
        
        self.x_b = np.array([])
        self.bias = bias
        weight_size = self.in_size
        if bias:
            weight_size += 1
        
        # Initial values of the weights
        var_w = 2/(weight_size + self.out_size)
        self.w = np.random.randn(weight_size, self.out_size) * math.sqrt(var_w)
        self.dL_dw = np.zeros(self.w.shape)
        #print("W ", self.w.shape)
        #print("dw ", self.dL_dw.shape)

        
    
    def forward (self, X):
        """
        input is x
        output is s = w*x
        """
        
        assert (X.shape[1] == self.in_size),("Input size is not correct", "Should be:",self.in_size, "found:", X.shape[1])
           
        # stack the bias to the input
        if self.bias:
            b_column = np.ones((len(X), 1))
            self.x_b = np.c_[b_column, X]
        else:
            self.x_b = X
        return np.dot(self.x_b,self.w)
            
         
    def backward (self, dL_ds):
        '''
        input is dL/ds
        output dL/dy = d(wx+b)/dw = x 
        '''
        
        self.dL_dw = np.dot(self.x_b.T, dL_ds)
        assert self.dL_dw.shape == self.w.shape,("Dimension is not correct","dL_dw:", self.dL_dw.shape,"x_b:", self.x_b.shape, "should be W:", self.w.shape)
        
        dL_x0 = np.dot(dL_ds, self.w.T)
        assert dL_x0.shape == self.x_b.shape,("Dimension is not correct","w:", self.w.shape,"dL_ds:", dL_ds.shape, "should be W:", self.x_b.shape)
        
        if self.bias:
            # return the x without the 1st column (bias)
            dL_x0 = dL_x0[:,1:]
        
        return dL_x0
    
    
    def update (self, params):
        '''
        input should be new w
        '''
        
        new_w = params[0]
        assert new_w.shape == self.w.shape, ("Params size is not correct","Params ", new_w.shape,"W ", self.w.shape)
        self.w = new_w
    
    def param(self):
        '''
        output is w, dw
        '''
        
        return [(self.w, self.dL_dw)]
