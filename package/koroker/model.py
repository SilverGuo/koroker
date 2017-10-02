from .base import BaseSeqLabel
from .utils.pipeline import embed_from_npy

from .config import ConfigLstmCrf

# lstm crf model for sequence labeling
class ModelLstmCrf(BaseSeqLabel):

    def __init__(self, config_path):
        # config class
        self.config = ConfigLstmCrf(config_path)

        # embed
        self.word_embed = None
        self.char_embed = None

        # placeholder
        self.seq_len = None
        self.word_id = None
        self.word_len = None
        self.char_id = None
        self.label = None

    # load embed matrix
    def load_embed(self, word_path=None, char_path=None):
        # word embed
        if word_path is not None:
            self.word_embed = embed_from_npy(word_path)

        # char embed
        if char_path is not None:
            self.char_embed = embed_from_npy(word_path)

    # feed dict for model
    def build_feed_dict(self, doc, label=None):
        pass

    # add placeholder

    def analysis(self, doc):
        super(ModelLstmCrf, self).analysis()
