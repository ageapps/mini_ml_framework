import logging



FORMAT_CONS = '%(asctime)s %(name)2s %(levelname)2s %(message)s'

def getLogger(name, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=FORMAT_CONS, datefmt='%d-%b-%y %H:%M:%S')
    return logging.getLogger(name)
