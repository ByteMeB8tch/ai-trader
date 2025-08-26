import logging, sys 
 
def get_logger(name="ai_trader"): 
    logger = logging.getLogger(name) 
    if logger.handlers: 
        return logger 
    logger.setLevel(logging.INFO) 
    fmt = logging.Formatter("%%%(asctime)s | %%%(levelname)s | %%%(message)s") 
    sh = logging.StreamHandler(sys.stdout) 
    sh.setFormatter(fmt) 
    logger.addHandler(sh) 
    return logger 
