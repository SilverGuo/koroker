from ..base import BasePrepare
from .data_set import DataNer
from ..utils.data_io import read_conll


class PrepareNer(BasePrepare):

    def __init__(self, config_path):
        super(PrepareNer, self).__init__(config_path)

        self.train, self.dev, self.test = self.load_data()

    def load_data(self):
        if self.config.file_format == 'conll':
            read_file = read_conll
        else:
            read_file = read_conll
        return DataNer(self.config.train_path, read_file), \
            DataNer(self.config.train_path, read_file), \
            DataNer(self.config.train_path, read_file)

    def process_data(self):
        pass

    def save_data(self):
        pass
