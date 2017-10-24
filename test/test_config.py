import unittest

from koroker.config import ConfigPrep, ConfigLstmCrf


class TestConfigPrep(unittest.TestCase):

    def setUp(self):
        self.config = ConfigPrep('./test/config_example/data_prepare.ini')

    def test_param(self):
        assert type(self.config.max_char_vocab) == int
        assert type(self.config.max_word_vocab) == int
        assert type(self.config.lower_word) == bool
