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
