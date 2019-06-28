
import matplotlib.pyplot as plt
import numpy as np


# ## Helpers and Ploting

# In[10]:


def standardize(A):
    return (A - np.mean(A, axis=0)) / np.std(A, axis=0)

# In[14]

def plotCostAndData(model,X,Y,cost, fig_name=False, title=False):
        fig_title = 'Learning result'
        if title:
            fig_title = title
        
        plotOutput = X.shape[1] == 1 and Y.shape[1] == 1
        
        if plotOutput:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))
            cost_ax = ax[0]
        else:
            fig, cost_ax = plt.subplots(1, 1, figsize=(10, 8))
        
        fig.suptitle(fig_title)
        # Plot Cost
        cost_ax.set_ylabel('Cost')
        cost_ax.set_xlabel('Iterations')
        _ = cost_ax.plot(range(len(cost)), cost, 'b-', label="Cost")
        cost_ax.legend()

        if plotOutput:
            pred = model.forward(X)
            # Plot original data
            _ = ax[1].plot(X, Y, 'b.', label="Input")
            # Plot regression line
            _ = ax[1].plot(X, pred, 'r.', label="Prediction")
            ax[1].set_ylabel('Y')
            ax[1].set_xlabel('X')
            ax[1].legend()
        if fig_name:
            fig.savefig(fig_name+'.png')
        else:
            plt.show()
