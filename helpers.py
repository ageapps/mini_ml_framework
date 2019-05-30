
import matplotlib.pyplot as plt
import numpy as np


# ## Helpers and Ploting

# In[10]:


def standarize(A):
    return (A - np.mean(A, axis=0)) / np.std(A, axis=0)


# In[14]:


def trainGD(model, optim, X, Y, iterations, v=False, eta=0.01):
    cost_history = []

    if v:
        print("Number of params in model:",len(model.param()))
        print("Params in model")
        print(model.param())

    for i in range(iterations):
        # forward pass, get prediction
        y_pred = model.forward(X)
        # calculate cost
        cost_history.append(optim.cost(y_pred,Y))
        # update weights, backward prop
        updates = []
        dL_dy = optim.backward()
        model.backward(dL_dy)
        for param in model.param():
            if not param:
                # empty param
                updates.append([])
                continue
                
            w = param[0]
            dw = param[1]
            new_w = w - eta*dw
            if v:
                print("w:", w)
                print("dw:", dw)
                print("new_w:", new_w)

            updates.append(new_w)            

        assert len(model.param()) == len(updates),("Lengths are not the same,params",len(model.param()), "updates", len(updates))
        if v:
            print("Iter: {} Cost:{}".format(i,cost_history[i]))
        elif i%10==0:
            print("Iter: {} Cost:{}".format(i,cost_history[i]))
        
        model.update(updates)

    return cost_history


# In[15]:


def trainBatchGD(model, optim, X, Y, iterations, batch_size=5, v=False, eta=0.01):
    cost_history = []

    if v:
        print("Number of params in model:",len(model.param()))
        print("Params in model")
        print(model.param())
        
    for i in range(iterations):
        cost = 0.0
        for b in range(0,X.shape[0],batch_size):            
            X_i = X[b:b+batch_size]
            Y_i = Y[b:b+batch_size]
            # forward pass, get prediction
            y_pred = model.forward(X_i)
            # calculate cost
            cost += optim.cost(y_pred,Y_i)
            # update weights, backward prop
            updates = []
            dL_dy = optim.backward()
            model.backward(dL_dy)
            for param in model.param():
                if not param:
                    # empty param
                    updates.append([])
                    continue

                w = param[0]
                dw = param[1]
                new_w = w - eta*dw
                if v:
                    print("w:", w)
                    print("dw:", dw)
                    print("new_w:", new_w)
                    
                updates.append(new_w)  
                
            assert len(model.param()) == len(updates),("Lengths are not the same,params",len(model.param()), "updates", len(updates))
            model.update(updates)
        
        cost_history.append(cost)
        if v:
            print("Iter: {} Cost:{}".format(i,cost_history[i]))
        elif i%3==0:
            print("Iter: {} Cost:{}".format(i,cost_history[i]))


    return cost_history


# In[16]:


def plotCostAndData(model,X,Y,cost, save=False):
        plotOutput = X.shape[1] == 1 and Y.shape[1] == 1
        
        if plotOutput:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))
            cost_ax = ax[0]
        else:
            fig, cost_ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot Cost
        cost_ax.set_ylabel('Cost')
        cost_ax.set_xlabel('Iterations')
        _ = cost_ax.plot(range(len(cost)), cost, 'b-', label="Cost")
        cost_ax.legend()

        if plotOutput:
            print("Ploting")
            pred = model.forward(X)
            # Plot original data
            _ = ax[1].plot(X, Y, 'b.', label="Input")
            # Plot regression line
            _ = ax[1].plot(X, pred, 'r.', label="Prediction")
            ax[1].set_ylabel('Y')
            ax[1].set_xlabel('X')
            ax[1].legend()
        if save:
            fig.savefig('figure.png')
