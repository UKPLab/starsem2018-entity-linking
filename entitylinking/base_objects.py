import abc
import logging


class Loggable(metaclass=abc.ABCMeta):

    def __init__(self, logger=None, **kwargs):
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
