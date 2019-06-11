
# coding: utf-8

# # Mini deep-learning framework

# In[2]:

import sys
import os
sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import math
from modules import Linear, LossMSE, ReLu, Tanh, Sequential, Trainer, LogManager
from modules.helpers import standardize, plotCostAndData

logger = LogManager.getLogger(__name__, False)

# ## Single Layer

# In[3]:


X = 2 * np.random.rand(100,1)
Y = 4 +3*X+np.random.randn(100,1)
X = standardize(X)
Y = standardize(Y)
model = Linear(X.shape[1],Y.shape[1])
optim = LossMSE()
trainer = Trainer(model, optim)
eta = 0.001
iterations = 100


# In[4]:


cost = trainer.trainBatchGD(X,Y,iterations, eta=eta)


# In[5]:


plotCostAndData(model,X,Y,cost)


# ## MultiLayer

# ### Linear + Activation

# In[8]:


class NN(object):
    def __init__(self, in_layer_size, out_layer_size):
        self.fc1 = Linear(in_layer_size,out_layer_size, bias=False)
        self.ac1 = Tanh()
        
    def forward(self, x):
        s1 = self.fc1.forward(x)
        a1 = self.ac1.forward(s1)
        return a1
    
    def update(self, params):
        #print("W:", params[0].shape)
        self.fc1.update([params[0]])
        if len(params)>1:
            #print("R:", len([params[1]]))
            self.ac1.update(params[1])
        #print("W:",self.fc1.param()[0][0])
        #print("dW:",self.fc1.param()[0][1])

    def backward(self, dL_dy):
        '''
        output dy/dw2 = d(f(wx+b))/dw = x
        output dy/dw1 = d(f(wx+b))/dw = x 
        '''
        #print(dL_dy)
        dL_ds = self.ac1.backward(dL_dy)
        dL_dy0 = self.fc1.backward(dL_ds)
        #print(dL_dy0)
        return dL_dy0
    
    
    def param(self):
        return [self.fc1.param()[0],self.ac1.param()[0]]


# In[9]:


nn = NN(X.shape[1],Y.shape[1])
optim_nn = LossMSE()
trainer = Trainer(nn, optim_nn)

eta = 0.001
iterations = 100


# In[10]:


cost_nn = trainer.trainBatchGD(X,Y,iterations, eta=eta)


# In[11]:


plotCostAndData(nn,X,Y,cost_nn)


# ### MLP (Lin+ Relu + Lin)

# In[13]:


N, D_in, H1,H2, D_out = 200, 1, 50, 50, 1

X = 3 * np.random.rand(N,D_in)
Y = 4*np.sin(X) + 0.5*np.random.randn(N,D_out)
X = standardize(X)
Y = standardize(Y)

logger.info("X: "+str(X.shape))
logger.info("Y: "+str(Y.shape))


# In[14]:


class NN2(object):
    def __init__(self, in_layer_size, hidden_layer_size, out_layer_size):
        self.fc1 = Linear(in_layer_size,hidden_layer_size)
        self.ac1 = ReLu()
        self.fc2 = Linear(hidden_layer_size,out_layer_size)

        
    def forward(self, x):
        s1 = self.fc1.forward(x)
        a1 = self.ac1.forward(s1)
        a2 = self.fc2.forward(a1)
        return a2
    
    def update(self, params):
        self.fc1.update([params[0]])
        self.fc2.update([params[1]])
        
    def backward(self, dL_dy2):
        '''
        output dy/dw2 = d(f(wx+b))/dw = x
        output dy/dw1 = d(f(wx+b))/dw = x 
        '''

        #dL_ds2 = self.ac2.backward(dL_dy2)
        dL_dy1 = self.fc2.backward(dL_dy2)
        dL_ds1 = self.ac1.backward(dL_dy1)
        dL_dy0 = self.fc1.backward(dL_ds1)
        
        return dL_dy0
    
    
    def param(self):
        return [self.fc1.param()[0],self.fc2.param()[0]]


# In[15]:


nn = NN2(X.shape[1],30,Y.shape[1])
optim_nn = LossMSE()
trainer = Trainer(nn, optim_nn)
iterations = 200
eta = 0.0001


# In[16]:


cost_nn = trainer.trainBatchGD(X,Y,iterations, eta=eta)
plotCostAndData(nn,X,Y,cost_nn)


# ### Sequential

# In[18]:


nn_seq = Sequential(
    Linear(D_in, H1),
    Tanh(),
    Linear(H1, H2),
    Tanh(),
    Linear(H2, D_out)
)

optim_nn = LossMSE()
trainer = Trainer(nn_seq, optim_nn)
iterations = 300
eta = 0.0008
nn_seq.modules


# In[19]:


cost_nn_seq = trainer.trainBatchGD(X,Y, iterations, eta=eta)
plotCostAndData(nn_seq,X,Y,cost_nn_seq)


# ## Circular input

# In[248]:


def generate_disc_set(nb, radius=1):
    coords = 4*radius*np.random.rand(nb,2)
    center = [2*radius,2*radius]
    #target = np.empty([nb,1])
    target = np.empty([nb,2])
    for index in range(nb):
        x = coords[index][0]
        y = coords[index][1]
        d = np.sqrt(np.square(x-center[0])+np.square(y-center[1]))
        inside = 1.0 if d <= radius else 0
        target[index][0] = inside
        target[index][1] = np.abs(1-inside)
        
    return coords, target


# In[249]:


nb = 1000
inp_train, target_train  = generate_disc_set(nb)
inp_test, target_test  = generate_disc_set(nb)

# Standardizing the data
inp_train = standardize(inp_train);
inp_test = standardize(inp_test);


# In[250]:


plt.title('Target set')
plt.scatter(inp_train[:, 0], inp_train[:, 1], c = target_train[:,0], s = 1)


# In[277]:


D_in, H,D_out = 2, 30, 2
nn_test = Sequential(
    Linear(D_in, H),
    ReLu(),
    Linear(H, H),
    Tanh(),
    Linear(H, H),
    Tanh(),
    Linear(H, H),
    Tanh(),
    Linear(H, D_out),
    Tanh()
)

optim_nn = LossMSE()
trainer = Trainer(nn_test, optim_nn)
iterations = 200
eta = 0.00005
batch_size = 50
cost_nn_seq = trainer.trainBatchGD(inp_train, target_train, iterations, eta=eta, batch_size=batch_size)
plotCostAndData(nn_test,inp_train,target_train,cost_nn_seq)


# In[243]:


# Computing the test error
def compute_nb_errors(model, data_input, data_target):
    mini_batch_size = 10
    nb_data_errors = 0
    for b in range(0, data_input.shape[0], mini_batch_size):
        input_batch = data_input[b:b+mini_batch_size]
        predicted_classes = model.forward(input_batch)
        predicted_classes = predicted_classes > 0.5
        for k in range(mini_batch_size):
            if data_target[b + k, 0] != predicted_classes[k, 0]:
                nb_data_errors += 1

    return nb_data_errors


# In[278]:


nb_test_err = compute_nb_errors(nn_test, inp_test, target_test) 
print('Number of errors on the test set : ', nb_test_err)


# In[291]:


prediction = nn_test.forward(inp_train)
plt.scatter(inp_train[:, 0], inp_test[:, 1], c = prediction[:, 0], s = 1)
plt.title('Prediction on the test set')

