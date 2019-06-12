import logging

FORMAT_CONS = '%(asctime)s %(name)2s %(levelname)2s %(message)s'
logging._srcfile = None  # Disable file lookup.
logging.logThreads = 0  # Disable logging thread information
logging.logProcesses = 0  # Disable logging process information
logging.logMultiprocessing = 0  # Disable logging multiprocess information

def getLogger(name, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=FORMAT_CONS, datefmt='%H:%M:%S')
    return logging.getLogger(name)
