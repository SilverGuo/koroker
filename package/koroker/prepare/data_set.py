# data set for ner
class DataNer:

    def __init__(self, data_path, file_parse):
        self.sample_word, self.label = file_parse(data_path)
