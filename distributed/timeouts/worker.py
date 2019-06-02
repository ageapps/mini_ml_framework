import time
import sys, os
import math

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from python_sockets.client import Client
from python_sockets.protocol import FragmentProtocol
from python_sockets.protocol import code

from modules import Linear, LossMSE, ReLu, Tanh, Sequential
from modules.trainer import trainBatchGD, trainGD
from modules.helpers import *

HOST = 'localhost'
PORT = 12344
QUEUE_SIZE = 5
UDP_CLIENT = True
HEADER_SIZE = 20

STATE_INITIAL = 'initial'
STATE_LEARNING = 'learning'
STATE_FINISHED = 'finished'

proto = FragmentProtocol()
socket_adapter = Client(PORT, host=HOST, udp=UDP_CLIENT, protocol=proto)

def generate_data(input_size, output_size):
  X = 2 * np.random.rand(input_size,output_size)
  Y = 4 +3*X+np.random.randn(input_size,output_size)
  return X,Y

def get_formated_message( msg, key='', code=0):
    msg = {
        'data': msg,
        'key': key,
        'code': code
        }
    return msg


def on_params_update(params, step):
    print("Updating params")
    for i,param in enumerate(params):
        if len(param) == 0:
            continue
        
        param = param.tolist()
        columns = len(param[0])
        for c in range(columns):
          # get values by column
          w = [row[c] for row in param]
          # send weights
          print('Step {} | Sending weights: {}'.format(step, param))
          agg_msg = {
              'params': w,
              'step': step
          }
          for t in range(10):
            answer = socket_adapter.send_message(get_formated_message(agg_msg, STATE_LEARNING), wait_answer=True)
            print('Got answer: ', answer)
            if answer['code'] != code.CODE_OK:
                print('trying again...')
                time.sleep(0.1)
            else:
              break

          if answer['code'] != code.CODE_OK:
              raise Exception('Error on answer')
          # receive aggregated
          new_param =answer['data']
          if len(new_param) == len(w):
              for j, row in enumerate(param):
                row[c] = new_param[j]  

              params[i] = np.array(param)
          else:
              print('Empty weights')
  
    print('New params are:', params)          
    return params

def main():
  current_state = STATE_INITIAL
  worker_name = 'worker'
  if len(sys.argv) > 1:
    worker_name = sys.argv[1]

  print('Initializing worker ' + worker_name)
  while True:
    answer = socket_adapter.send_message(get_formated_message('setup', current_state), wait_answer=True)
    if answer['key'] == current_state and answer['code'] == code.CODE_OK:
        print("Worker successfully registered")
    else:
        print("Error on setup | message:{}".format(answer))

    learning_parameters = answer['data']
    if learning_parameters and answer['code'] == code.CODE_OK:
      input_size = learning_parameters['input_size']
      output_size = learning_parameters['output_size']
      eta = learning_parameters['eta']
      iterations = learning_parameters['iterations']
      break
    else:
      print("Waiting for setup data")
      time.sleep(2)


  print('Learning parameters are: {}'.format(learning_parameters))
  x,y = generate_data(input_size, output_size)
  X = standarize(x)
  Y = standarize(y)
  model = Linear(X.shape[1],Y.shape[1])
  optim = LossMSE()
  
  while True:
    current_state = STATE_LEARNING
    print("Waiting to start learning")
    answer = socket_adapter.send_message(get_formated_message('',current_state), wait_answer=True)
    if answer['code'] == code.CODE_OK:
      print("Start learning")
      break
    
    time.sleep(2)


  cost = trainGD(model,optim,X,Y,iterations, eta=eta, update_func=on_params_update, v=False)
  plotCostAndData(model,X,Y,cost, fig_name=worker_name)
  # socket_adapter.send_message({ 'name': 'Time', 'time': time.time()})


if __name__ == '__main__':
    main()
