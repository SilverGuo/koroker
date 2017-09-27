# std lib
from abc import ABCMeta, abstractmethod

# ext lib
import six

# package
from .utils.type_check import check_string_list


# base class for sequence labeling
@six.with_metaclass(ABCMeta)
class BaseSeqLabel:

    # sequence labeling for document
    @abstractmethod
    def analysis(self, doc):
        if not check_string_list(doc):
            raise ValueError('doc must be string list')
