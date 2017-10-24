import numpy as np

from ..model import ModelLstmCrf
from ..utils.data_process import create_batch, label_chunk


# modified lstm crf for ner
class LstmCrfNer(ModelLstmCrf):

    # override for ner
    def run_evaluate(self, sess, data):
        # accuracy sequence
        seq_accuracy = []

        # chunk number
        match_chunk, real_chunk, pred_chunk = 0.0, 0.0, 0.0

        for sample, label in create_batch(data, self.config.batch_size):
            label_pred, seq_len = self.predict_batch(sess, sample)

            for y, y_pred, lseq in zip(label, label_pred, seq_len):
                # mask
                y = y[:lseq]
                y_pred = y_pred[:lseq]

                seq_accuracy += [r == p for (r, p) in zip(y, y_pred)]

                y_chunk = set(label_chunk(y, self.label_dict))
                y_pred_chunk = set(label_chunk(y_pred, self.label_dict))
                match_chunk += len(y_chunk & y_pred_chunk)
                real_chunk += len(y_chunk)
                pred_chunk += len(y_pred_chunk)

        if match_chunk == 0:
            f1 = 0.0
        else:
            p = match_chunk / pred_chunk
            r = match_chunk / real_chunk
            f1 = 2 * p * r / (p + r)

        accuracy = np.mean(seq_accuracy)
        return accuracy, f1
