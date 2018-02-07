UNK = 'unk'


# data set for ner
class DataNer:

    def __init__(self, data_path, file_parse):
        self.sample_tok, self.label = file_parse(data_path)
        self.data = None

    def vocab_lookup(self, word_idx, char_idx, label_dict, word_lower=True):
        word_vec = []
        char_vec = []

        # traverse
        for tok_list in self.sample_tok:
            idx_tok_list = []
            char_list = []
            for tok in tok_list:
                # char level
                idx_char_list = []
                for c in tok:
                    idx_char_list.append(char_idx.get(c, char_idx[UNK]))
                char_list.append(idx_char_list)

                # word lower
                if word_lower:
                    tok = tok.lower()
                idx_tok_list.append(word_idx.get(tok, word_idx[UNK]))

            word_vec.append(idx_tok_list)
            char_vec.append(char_list)

        # sample
        sample = dict()
        sample['word_id'] = word_vec
        sample['char_id'] = char_vec

        # label
        label_vec = []
        for sentence in self.label:
            idx_label_list = []
            for l in sentence:
                idx_label_list.append(label_dict[l])
            label_vec.append(idx_label_list)

        # data for training
        self.data = (sample, label_vec)

class OutVoc(DataNer):
    def __init__(self, oov_set):
        self.sample_tok = [o[0] for o in oov_set]
        self.label = [o[1] for o in oov_set]