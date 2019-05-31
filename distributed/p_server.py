import sys
from python_sockets.server import UDPRServer
from python_sockets.protocol import FragmentProtocol
from python_sockets.protocol import code

PORT = 12344
HOST = ''
UDP_SERVER = True
HEADER_SIZE = 20

learning_parameters = {
    'iterations': 100,
    'eta': 0.001,
    'input_size': 1000,
    'output_size': 1
}

worker_num = 3

STATE_INITIAL = 'initial'
STATE_LEARNING = 'learning'
STATE_FINISHED = 'finished'


def aggregate_values(params, new_values):
    if len(params) == 0:
        print("Params size is 0", params, new_values)
        return new_values

    assert len(params) == len(
        new_values), "Sizes are not the same: {}/{}".format(len(params), len(new_values))

    print("Aggregating parameters")
    for i, param in enumerate(params):
        params[i] = (param + new_values[i]) / 2

    return params


def send_error(server, destination):
    server.send_message('', destination, code=code.CODE_ERROR)


def send_ack(server, destination):
    server.send_message('', destination, code=code.CODE_OK)


def main():
    proto = FragmentProtocol()
    server = UDPRServer(PORT, protocol=proto)
    server.start()
    current_status = STATE_INITIAL
    workers = []
    current_step = 0
    aggregated_params = []
    cached_params = []

    while True:
        msg, client_address = server.receive_message(send_answer=False)
        key = msg['key']

        if key != current_status:
            print('Received wrong key: {}/{}'.format(key, current_status))
            send_error(server, client_address)
            continue

        if current_status == STATE_INITIAL:
            # setup phase
            if client_address not in workers:
                print('Registering worker: {}'.format(client_address))
                workers.append(client_address)
                print('Sending parameters')
                server.send_message(
                    learning_parameters, client_address, key=current_status, code=code.CODE_OK)
            else:
                print('Worker {} is already registered'.format(client_address))
                send_error(server, client_address)

            if len(workers) == worker_num:
                print('All workers registered')
                current_status = STATE_LEARNING
                workers = []

        elif current_status == STATE_LEARNING:
            data = msg['data']
            if not data:
                send_ack(server, client_address)
                continue

            params = data['params']
            step = data['step']
            if int(step) != current_step:
                print("Wrong step {}/{}".format(step, current_step))
                send_error(server, client_address)
                continue
            else:
                if client_address not in workers:
                    workers.append(client_address)
                    aggregated_params = aggregate_values(
                        aggregated_params, params)
                else:
                    print('Worker {} is already aggregated'.format(client_address))
                    send_error(server, client_address)

            if len(workers) == worker_num:
                print('All workers aggregated in step:', current_step)
                cached_params = aggregated_params
                aggregated_params = []
                print('Next step')
                current_step += 1
                print('Sending parameters to workers')
                for w in workers:
                    server.send_message(
                    cached_params, w, key=current_status, code=code.CODE_OK)
                workers = []
                if current_step == learning_parameters['iterations']:
                    print("Finished")
                    sys.exit()
                    current_status = STATE_FINISHED

        elif current_status == STATE_FINISHED:
            print("Finished")
            sys.exit()
            pass
        else:
            print('Unknown key')
            sys.exit()

    server.stop()


if __name__ == '__main__':
    main()
