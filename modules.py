
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


# In[6]:


class LossMSE(object):
    def __init__(self):
        self.y_t = np.array([])
        self.y_p = np.array([])
        self.L = np.array([])
        self.dL_dy = np.array([])
        self.m = 1

    def cost(self, predicted, target):
        """
        input is y_p and y_t
        output is L = MSE = 1/2m*(y_t-y_p)^2
        """ 
        
        self.m = target.shape[1]
        self.y_t = target
        self.y_p = predicted
        self.L = (1/(2*self.m))*np.sum(np.square(self.y_p-self.y_t))
        return self.L
    
      
    def backward(self):
        """
        output is dL/dy = 2(y_p-y_t) = dL/da
        """ 
        
        self.dL_dy = (1/self.m)*(self.y_p-self.y_t)
        return self.dL_dy
    
    
    def param(self):
        '''
        output is [L, dL/dw]
        '''
        
        return [(self.L,self.dL_dy)]


# In[7]:


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


# In[9]:


class Sequential(object):
    def __init__(self, *args):
        self.modules = {}
        for idx, module in enumerate(args):
            self.modules.update({idx: module})
        #print("Model with:", len(self.modules), "layers")
            
    def forward (self, x):
        """
        input is x
        output is y
        """ 
        
        xi=x
        for module in self.modules.values():
            xi = module.forward(xi)
        return xi
    
   
    def backward (self, dL_dy) :   
        """
        input is dL/da=dL/dy
        output is dL/dw 
        """ 
        
        yi=dL_dy
        module_list = [self.modules[k] for k in self.modules]
        module_list.reverse()
        for module in module_list:
            yi = module.backward(yi)
        return yi

    
    def update(self, params):
        for i,_ in enumerate(self.modules.values()):
            param = params[i]
            module = self.modules[i]
            if len(param) > 1:
                # only update if module has params
                module.update([param])
                
        #print("W:",self.fc1.param()[0][0])
        #print("dW:",self.fc1.param()[0][1])
    
    def param(self):
        '''
        output is [(Z, dZ/dw)]
        '''
        
        return [module.param()[0] for module in self.modules.values()]
