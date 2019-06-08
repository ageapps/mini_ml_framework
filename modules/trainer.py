import numpy as np
from .logger import log

def trainGD(model, optim, X, Y, iterations, v=False, eta=0.01, update_func=None):
    cost_history = []

    if v:
        log.info('Number of params in model:',len(model.param()))
        log.info('Params in model')
        log.info(model.param())

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
                print('w:', w)
                print('dw:', dw)
                print('new_w:', new_w)

            updates.append(new_w)            

        assert len(model.param()) == len(updates),('Lengths are not the same,params',len(model.param()), 'updates', len(updates))
        if v:
            log.info('Iter: {} Cost:{}'.format(i,cost_history[i]))
        elif i%10==0:
            log.info('Iter: {} Cost:{}'.format(i,cost_history[i]))
        
        if update_func is not None:
            updates = update_func(updates, i)

        model.update(updates)

    return cost_history




def trainBatchGD(model, optim, X, Y, iterations, batch_size=5, v=False, eta=0.01, update_func=None):
    cost_history = []

    if v:
        log.info('Number of params in model:',len(model.param()))
        log.info('Params in model')
        log.info(model.param())
        
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
                    log.info('w:', w)
                    log.info('dw:', dw)
                    log.info('new_w:', new_w)
                    
                updates.append(new_w)  
                
            assert len(model.param()) == len(updates),('Lengths are not the same,params',len(model.param()), 'updates', len(updates))
            model.update(updates)
        
        if update_func is not None:
            updates = update_func(updates, i)
        model.update(updates)
        cost_history.append(cost)
        if v:
            log.info('Iter: {} Cost:{}'.format(i,cost_history[i]))
        elif i%3==0:
            log.info('Iter: {} Cost:{}'.format(i,cost_history[i]))


    return cost_history
