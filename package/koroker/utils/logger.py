import logging

LOG_FORMAT = '%(asctime)s:%(levelname)s;%(message)s'


# create logger and its file handler
def new_logger(log_name, log_path):
    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # config for log system
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

    # create file handler
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logging.getLogger().addHandler(handler)
    return logger
