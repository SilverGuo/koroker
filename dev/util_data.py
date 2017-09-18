import numpy as np

# dataset for ner
class DataNER:

    # norm input
    def __init__(self, data_path, config, norm_func):
        self.lsentence, self.ltag = norm_func(data_path)
        self.config = config

    # create vocab of word and character
    def create_vocab(self):
        # vocab
        word_vocab = set()
        char_vocab = set()

        # traverse
        for ltok in self.lsentence:
            for t in ltok:
                # lower case
                if self.config.word_lower:
                    t = t.lower()
                # pure digit
                if self.config.norm_digit and t.isdigit():
                    word_vocab.add('<NUM>')
                else:
                    word_vocab.add(t)
                # char level
                for c in t:
                    char_vocab.add(c)
        
        # add unknown
        word_vocab.add('<UNK>')
        char_vocab.add('<UNK>')
        
        return word_vocab, char_vocab

    # create entity dict
    def create_entity(self):
        entity = set()
        for label in self.ltag:
            for t in label:
                entity.add(t)
        entity_dict = dict()
        for idx, ent in enumerate(list(entity)):
            entity_dict[ent] = idx
        return entity_dict

    # generate dataset for ner model
    def generate_dataset(self, word_idx, char_idx, entity_dict):
        vec_word = []
        vec_char = []

        # traverse
        for ltok in self.lsentence:
            tok_word = []
            list_char = []
            for t in ltok:
                # lower case
                if self.config.word_lower:
                    t = t.lower()
                # pure digit
                if self.config.norm_digit and t.isdigit():
                    tok_word.append(word_idx['<NUM>'])
                else: 
                    tok_word.append(word_idx.get(t, word_idx['<UNK>']))
                # char level
                tok_char = []
                for c in t:
                    tok_char.append(char_idx.get(c, char_idx['<UNK>']))
                list_char.append(tok_char)
            vec_word.append(tok_word)
            vec_char.append(list_char)
        
        # doc dict
        doc = dict()
        doc['word_id'] = vec_word
        doc['char_id'] = vec_char

        # label
        label = []
        for lent in self.ltag:
            lt = []
            for t in lent:
                lt.append(entity_dict[t])
            label.append(lt)

        # dataset
        self.dataset = (doc, label)

        return



# conll data norm
def conll_norm(data_path):
    lsentence, ltag = [], []
    tok, label = [], []

    # read file
    with open(data_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '' or line.startswith('-DOCSTART-'):
                if tok != [] and label != []: 
                    lsentence.append(tok)
                    ltag.append(label)
                    tok, label = [], []
            else:
                temp = line.split()
                tok.append(temp[0])
                label.append(temp[3])
    
    # list of token seq and tag seq
    return lsentence, ltag 
        

# load embed
def load_embed(embed_path):
    dembed = dict()
    with open(embed_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line != '':
                wvec = line.split()
                dembed[wvec[0]] = [float(v) for v in wvec[1:]]
    return dembed

"""
TO BE DONE
"""
# generate embed
def generate_embed(vocab, embed):
    vocab_embed = []
    for tok in vocab:
        pass
    return


# generate vocab mapping from token list
def generate_vocab(ltok):
    tok_idx, idx_tok = dict(), dict()
    for idx, tok in enumerate(ltok):
        tok_idx[tok] = idx
        idx_tok[idx] = tok
    return tok_idx, idx_tok


# padding for seq list
def pad_lseq(lseq, tok, max_len):
    lseq_padded = []
    lseq_len = []
    for seq in lseq:
        lseq_padded.append(seq[:max_len] + 
                           [tok] * max(max_len - len(seq), 0))
        lseq_len.append(min(len(seq), max_len))

    return lseq_padded, lseq_len


# padding for batch
def pad_batch(batch, tok=0, level='word'):

    # nn unit is word
    if level == 'word':
        max_len = max(map(lambda x: len(x), batch))
        batch_padded, batch_len = pad_lseq(batch, tok, max_len)
    
    # nn unit is char
    elif level == 'char':
        # first pad for char seq
        max_len = max([max(map(lambda x: len(x), doc)) for doc in batch])
        batch_padded = []
        batch_len = []
        for doc in batch:
            doc_padded, doc_len = pad_lseq(doc, tok, max_len)
            batch_padded.append(doc_padded)
            batch_len.append(doc_len)

        # second pad for word seq
        max_doc_len = max(map(lambda x: len(x), batch))
        batch_padded, _ = pad_lseq(batch_padded, [tok]*max_len, max_doc_len)
        batch_len, _ = pad_lseq(batch_len, 0, max_doc_len)
    
    # other
    else:
        batch_padded = []
        batch_len = []
    
    return batch_padded, batch_len


# create batch
def create_batch(data, batch_size):
    # X, y
    X_batch = {'word_id': [], 'char_id': []}
    y_batch = []
    for idx in range(len(data[1])):
        if len(y_batch) == batch_size:
            yield X_batch, y_batch
            X_batch = {'word_id': [], 'char_id': []}
            y_batch = []
        
        X_batch['word_id'].append(data[0]['word_id'][idx])
        X_batch['char_id'].append(data[0]['char_id'][idx])
        y_batch.append(data[1][idx])
    
    # rest
    if len(y_batch) != 0:
        yield X_batch, y_batch
    

# extract chunk
def extract_chunk(label, entity_dict):
    
    idx_other = entity_dict['O']
    idx_dict = {idx: entity for entity, idx in entity_dict.items()}
    
    lchunk = []
    chunk_type, chunk_start = None, None

    for idx, tok in enumerate(label):

        # end of chunk
        if tok == idx_other and chunk_type is not None:
            chunk = (chunk_type, chunk_start, idx)
            # add to list
            lchunk.append(chunk)
            chunk_type, chunk_start = None, None
        
        # next chunk
        elif tok != idx_other:
            tok_class = idx_dict[tok].split('-')[0]
            tok_type = idx_dict[tok].split('-')[1]
            # new chunk
            if chunk_type is None:
                chunk_type, chunk_start = tok_type, idx
            elif tok_type != chunk_type or tok_class == "B":
                chunk = (chunk_type, chunk_start, idx)
                lchunk.append(chunk)
                chunk_type, chunk_start = tok_type, idx
        else:
            pass
    
    # end
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(label))
        lchunk.append(chunk)

    return lchunk

