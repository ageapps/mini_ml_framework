import numpy as np
import time
from . import LogManager

class Trainer(object):
    def __init__(self, model, optimizer, v=False, vv=False):
        self.model = model
        self.optimizer = optimizer
        self.v = v
        self.vv = vv
        self.logger = LogManager.getLogger(__name__, vv)

    def debug(self, content):
        if self.vv:
            self.logger.debug(content)
    
    def info(self, content):
        if self.v:
            self.logger.info(content)

    def trainGD(self, X, Y, iterations, eta=0.01, update_func=None):
        cost_history = []
        y_history= []
        time_history= []

        self.info('Training model with params: ' + str(len(self.model.param())))
        self.debug('Params in model')
        self.debug(self.model.param())
        start_time = time.time()
        for i in range(iterations):
            # forward pass, get prediction
            y_pred = self.model.forward(X)
            # calculate cost
            cost_history.append(self.optimizer.cost(y_pred,Y))
            y_history.append(y_pred)
            time_history.append(time.time() - start_time)
            
            # update weights, backward prop
            updates = []
            dL_dy = self.optimizer.backward()
            self.model.backward(dL_dy)
            for param in self.model.param():
                if not param:
                    # empty param
                    updates.append([])
                    continue
                    
                w = param[0]
                dw = param[1]
                new_w = w - eta*dw
                if self.vv:
                    self.debug('w:' + str(w))
                    self.debug('dw:'+ str(dw))
                    self.debug('new_w:' + str(new_w))

                updates.append(new_w)            

            assert len(self.model.param()) == len(updates),('Lengths are not the same,params',len(self.model.param()), 'updates', len(updates))
            
            if i%10==0:
                self.info('Iter: {} Cost:{}'.format(i,cost_history[i]))
            elif self.vv:
                self.debug('Iter: {} Cost:{}'.format(i,cost_history[i]))

            
            if update_func is not None:
                updates = update_func(updates, i)

            self.model.update(updates)

        return cost_history, y_history, time_history




    def trainBatchGD(self, X, Y, iterations, batch_size=5, eta=0.01, update_func=None):
        cost_history = []
        y_history= []
        time_history= []

        self.info('Training model with params: ' + str(len(self.model.param())))
        self.debug('Params in model')
        self.debug(self.model.param())
        
        start_time = time.time()
        for i in range(iterations):
            cost = 0.0
            for b in range(0,X.shape[0],batch_size):            
                X_i = X[b:b+batch_size]
                Y_i = Y[b:b+batch_size]
                # forward pass, get prediction
                y_pred = self.model.forward(X_i)
                # calculate cost
                cost += self.optimizer.cost(y_pred,Y_i)
                # update weights, backward prop
                updates = []
                dL_dy = self.optimizer.backward()
                self.model.backward(dL_dy)
                for param in self.model.param():
                    if not param:
                        # empty param
                        updates.append([])
                        continue

                    w = param[0]
                    dw = param[1]
                    new_w = w - eta*dw
                    if self.vv:
                        self.debug('w:' + str(w))
                        self.debug('dw:'+ str(dw))
                        self.debug('new_w:' + str(new_w))
                        
                    updates.append(new_w)  
                    
                assert len(self.model.param()) == len(updates),('Lengths are not the same,params',len(self.model.param()), 'updates', len(updates))
                self.model.update(updates)
            
            if update_func is not None:
                updates = update_func(updates, i)
            self.model.update(updates)
            cost_history.append(cost)
            y_history.append(y_pred)
            time_history.append(time.time() - start_time)

            if i%10==0:
                self.info('Iter: {} Cost:{}'.format(i,cost_history[i]))
            elif self.vv:
                self.debug('Iter: {} Cost:{}'.format(i,cost_history[i]))

        return cost_history, y_history, time_history
