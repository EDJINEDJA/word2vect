import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)

class Tokenizer():
    def __init__(self) -> None:
        pass

    def get_english_tokenizer(self):
        """
        Documentation:
        https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
        """
        tokenizer = get_tokenizer("basic_english", language="en")

        return tokenizer


class dataloader():
    def __init__(self) -> None:
        pass

    def dataloader():
        pass