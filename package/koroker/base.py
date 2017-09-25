# std lib
from abc import ABCMeta, abstractmethod

# ext lib
import six


# base class for sequence labeling
@six.with_metaclass(ABCMeta)
class BaseSeqLabel:

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass
