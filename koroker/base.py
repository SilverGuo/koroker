from abc import ABCMeta, abstractmethod

import six
from six.moves import configparser

from .utils.type_check import check_string_list


# base class for sequence labeling
@six.add_metaclass(ABCMeta)
class BaseSeqLabel:

    # sequence labeling for document
    @abstractmethod
    def analysis(self, doc):
        if not check_string_list(doc):
            raise ValueError('doc must be string list')


# base class for data prepare
@six.add_metaclass(ABCMeta)
class BasePrepare:

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def save_data(self):
        pass


# base class for config
@six.add_metaclass(ABCMeta)
class BaseConfig:

    @abstractmethod
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
