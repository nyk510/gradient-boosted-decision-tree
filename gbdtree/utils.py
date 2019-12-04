# coding: utf-8
"""
utility functions
"""

__author__ = "nyk510"

from logging import getLogger, StreamHandler, Formatter

def get_logger(name, level="DEBUG"):
    logger = getLogger(name)

    if logger.handlers:
        logger.handlers = []
    sh = StreamHandler()
    fmter = Formatter('{asctime}\t{name}\t{message}', style='{')
    sh.setFormatter(fmter)
    sh.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(sh)
    return logger
