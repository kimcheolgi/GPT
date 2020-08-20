import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook, trange
import json

""" pretrain 데이터셋 """


class PretrainDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            with tqdm_notebook(total=line_cnt, desc=f"Loading") as pbar:
                for i, line in enumerate(f):
                    instance = json.loads(line)
                    self.sentences.append([vocab.piece_to_id(p) for p in instance["tokens"]])
                    pbar.update(1)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return (torch.tensor(self.sentences[item]), torch.tensor(item))

""" pretrain data collate_fn """
def pretrin_collate_fn(inputs):
    dec_inputs, item = list(zip(*inputs))

    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        dec_inputs,
        torch.stack(item, dim=0),
    ]
    return batch