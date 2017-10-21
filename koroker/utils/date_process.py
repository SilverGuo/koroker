from collections import Counter

import numpy as np

UNK = 'unk'


# load embed from txt
def load_embed(embed_path):
    embed_dict = dict()
    with open(embed_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line != '':
                vec = line.split()
                embed_dict[vec[0]] = [float(v) for v in vec[1:]]
    return embed_dict


# prepare embed
def prepare_embed(vocab, embed):
    vocab_embed = []
    for tok in vocab:
        vocab_embed.append(embed[tok])
    return np.array(vocab_embed, dtype=np.float32)


# create vocab dict from tokenized sentence list
def create_vocab(sample_tok, max_word=25000, max_char=100,
                 word_lower=True):
    # vocab
    word_counter = Counter()
    char_counter = Counter()

    # traverse
    for sentence in sample_tok:
        for tok in sentence:
            for c in tok:
                char_counter[c] += 1
            if word_lower:
                tok = tok.lower()
                word_counter[tok] += 1

    # vocab size limit
    word_vocab = [t[0] for t in word_counter.most_common(max_word-1)]
    word_vocab.append(UNK)
    char_vocab = [t[0] for t in char_counter.most_common(max_char-1)]
    char_vocab.append(UNK)
    return word_vocab, char_vocab


# create label dict
def create_label(label):
    label_vocab = set()
    for sentence in label:
        for l in sentence:
            label_vocab.add(l)
    label_dict = dict()
    for idx, l in enumerate(list(label_vocab)):
        label_dict[l] = idx
    return label_dict


# vocab mapping from token list
def vocab_mapping(tok_list):
    tok_idx, idx_tok = dict(), dict()
    for idx, tok in enumerate(tok_list):
        tok_idx[tok] = idx
        idx_tok[idx] = tok
    return tok_idx, idx_tok


# padding for sequence list
def pad_seq_list(lseq, tok, max_len):
    seq_list_padded = []
    seq_list_len = []
    for seq in lseq:
        seq_list_padded.append(seq[:max_len] +
                               [tok] * max(max_len - len(seq), 0))
        seq_list_len.append(min(len(seq), max_len))

    return seq_list_padded, seq_list_len


# padding for batch
def pad_batch(batch, tok=0, level='word'):
    # list of word id
    if level == 'word':
        max_len = max(map(lambda x: len(x), batch))
        batch_padded, batch_len = pad_seq_list(batch, tok, max_len)

    # list of char id
    elif level == 'char':
        # first pad for char seq
        max_len = max([max(map(lambda x: len(x), doc)) for doc in batch])
        batch_padded = []
        batch_len = []
        for doc in batch:
            doc_padded, doc_len = pad_seq_list(doc, tok, max_len)
            batch_padded.append(doc_padded)
            batch_len.append(doc_len)

        # second pad for word seq
        max_doc_len = max(map(lambda x: len(x), batch))
        batch_padded, _ = pad_seq_list(batch_padded, [tok] * max_len, max_doc_len)
        batch_len, _ = pad_seq_list(batch_len, 0, max_doc_len)

    # other
    else:
        raise ValueError('level must be word or char, got {}'.format(level))

    return batch_padded, batch_len


# create batch
def create_batch(data, batch_size, has_char=True):
    # x, y
    x_batch = {'word_id': [], 'char_id': []}
    y_batch = []
    for idx in range(len(data[1])):
        if len(y_batch) == batch_size:
            yield x_batch, y_batch
            x_batch = {'word_id': [], 'char_id': []}
            y_batch = []

        x_batch['word_id'].append(data[0]['word_id'][idx])
        if has_char:
            x_batch['char_id'].append(data[0]['char_id'][idx])
        y_batch.append(data[1][idx])

    # rest
    if len(y_batch) != 0:
        yield x_batch, y_batch


# get label by chunk
def label_chunk(label, label_dict):
    idx_other = label_dict['O']
    idx_dict = {idx: entity for entity, idx in label_dict.items()}

    seq_chunk = []
    chunk_type, chunk_start = None, None

    for idx, tok in enumerate(label):

        # end of chunk
        if tok == idx_other and chunk_type is not None:
            chunk = (chunk_type, chunk_start, idx)
            # add to list
            seq_chunk.append(chunk)
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
                seq_chunk.append(chunk)
                chunk_type, chunk_start = tok_type, idx
        else:
            pass

    # end
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(label))
        seq_chunk.append(chunk)

    return seq_chunk
