from abc import ABCMeta, abstractmethod

import six
from six.moves import configparser

from .utils.type_check import check_string_list


# base class for sequence labeling
@six.with_metaclass(ABCMeta)
class BaseSeqLabel:

    # sequence labeling for document
    @abstractmethod
    def analysis(self, doc):
        if not check_string_list(doc):
            raise ValueError('doc must be string list')


# base class for config
@six.with_metaclass(ABCMeta)
class BaseConfig:

    @abstractmethod
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
