import unittest

from koroker.prepare import PrepareNer


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.prepare = PrepareNer('./test/config_example/data_prepare.ini')

    def test_input(self):
        assert len(self.prepare.vocab_dict.keys()) == 2
