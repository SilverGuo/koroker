import unittest

from koroker.prepare import PrepareNer


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.prepare = PrepareNer('./test/config_name/data_prepare.ini')


