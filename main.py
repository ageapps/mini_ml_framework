# ## Testing

# In[17]:

import torch
from modules import *
torch.set_grad_enabled(False)


class NN(object):
    def __init__(self, in_layer_size, out_layer_size):
        self.fc1 = Linear(in_layer_size, out_layer_size, bias=False)
        self.ac1 = Tanh()

    def forward(self, x):
        s1 = self.fc1.forward(x)
        a1 = self.ac1.forward(s1)
        return a1

    def update(self, params):
        self.fc1.update([params[0]])
        if len(params) > 1:
            #print("R:", len([params[1]]))
            self.ac1.update(params[1])
        # print("W:",self.fc1.param()[0][0])
        # print("dW:",self.fc1.param()[0][1])

    def backward(self, dL_dy):
        '''
        output dy/dw2 = d(f(wx+b))/dw = x
        output dy/dw1 = d(f(wx+b))/dw = x 
        '''

        dL_ds = self.ac1.backward(dL_dy)
        dL_dy0 = self.fc1.backward(dL_ds)
        return dL_dy0

    def param(self):
        return [self.fc1.param()[0], self.ac1.param()[0]]


class NN2(object):
    def __init__(self, in_layer_size, hidden_layer_size, out_layer_size):
        self.fc1 = Linear(in_layer_size, hidden_layer_size)
        self.ac1 = ReLu()
        self.fc2 = Linear(hidden_layer_size, out_layer_size)
        #self.ac2 = Tanh()

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
        return [self.fc1.param()[0], self.fc2.param()[0]]


def testSingleLayer(X, Y, iterations, eta):
    # ### Single Layer
    model = Linear(X.shape[1], Y.shape[1])
    optim = LossMSE()
    eta = 0.001
    iterations = 100

    cost = trainBatchGD(model, optim, X, Y, iterations, eta=eta)
    plotCostAndData(model, X, Y, cost)


def testActivationLayer(X, Y, iterations, eta):
    # #### Linear + Activation
    nn = NN(X.shape[1], Y.shape[1])
    optim_nn = LossMSE()
    cost_nn = trainBatchGD(nn, optim_nn, X, Y, iterations, eta=eta)
    plotCostAndData(nn, X, Y, cost_nn)


def testMLP(X, Y, iterations, eta):
    nn = NN2(X.shape[1], 30, Y.shape[1])
    optim_nn = LossMSE()
    cost_nn = trainBatchGD(nn, optim_nn, X, Y, iterations, eta=eta)
    plotCostAndData(nn, X, Y, cost_nn)


def testSequential(nn_seq, X, Y, iterations, eta):
    optim_nn = LossMSE()
    cost_nn_seq = trainBatchGD(nn_seq, optim_nn, X, Y, iterations, eta=eta)
    plotCostAndData(nn_seq, X, Y, cost_nn_seq)


# In[17]:

def main():

    X = 2 * torch.empty(100, 1).random_()
    Y = 4 + 3 * X+torch.empty(100, 1).random_()
    X = standarize(X)
    Y = standarize(Y)

    eta = 0.001
    iterations = 100
    # testSingleLayer(X, Y, iterations, eta)
    # testActivationLayer(X, Y, iterations, eta)

    N, D_in, H1, H2, D_out = 200, 1, 50, 50, 1
    X = 3 * torch.empty(N, D_in).normal_(0.5, 0.5)
    Y = 4 * torch.sin(X) + 0.5*torch.empty(N, D_out).normal_()
    X = standarize(X)
    Y = standarize(Y)
    iterations = 200
    eta = 0.0001

    # testMLP(X, Y, iterations, eta)

    iterations = 300
    eta = 0.0005
    print("X", X.shape)
    print("Y", Y.shape)
    nn_seq = Sequential(
        Linear(D_in, H1),
        Tanh(),
        Linear(H1, H2),
        Tanh(),
        Linear(H2, D_out)
    )

    testSequential(nn_seq, X, Y, iterations, eta)


if __name__ == '__main__':
    main()
