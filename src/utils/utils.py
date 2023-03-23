import torch
import os

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from src.utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)

class Functional():
    def __init__(self) -> None:
        pass

    def get_data_iterator(self,ds_name, ds_type, data_dir):
        if ds_name == "WikiText2":
            data_iter = WikiText2(root=data_dir, split=(ds_type))
        elif ds_name == "WikiText103":
            data_iter = WikiText103(root=data_dir, split=(ds_type))
        else:
            raise ValueError("Choose dataset from: WikiText2, WikiText103")
        data_iter = to_map_style_dataset(data_iter)
        return data_iter

    def get_tokenizer(self):
        """
        Documentation:
        https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
        """
        tokenizer = get_tokenizer("basic_english", language="en")

        return tokenizer

    def get_vocab(self , data_iter, tokenizer):
        """Builds vocabulary from iterator"""
        vocab = build_vocab_from_iterator(
            map(tokenizer, data_iter),
            specials=["<unk>"],
            min_freq=1,
        )
        vocab.set_default_index(vocab["<unk>"])

        return vocab

    def collate_cbow(self , batch , text_pipeline):

        """
        Collate_fn for CBOW model to be used with Dataloader.
        `batch` is expected to be list of text paragrahs.

        Context is represented as N=CBOW_N_WORDS past words
        and N=CBOW_N_WORDS future words.

        Long paragraphs will be truncated to contain
        no more that MAX_SEQUENCE_LENGTH tokens.

        Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
        Each element in `batch_output` is a middle word.
        """
        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = text_pipeline(text)

            if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
                continue

            if MAX_SEQUENCE_LENGTH:
                text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

            for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
                token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
                output = token_id_sequence.pop(CBOW_N_WORDS)
                input_ = token_id_sequence
                batch_input.append(input_)
                batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output


    def collate_skipgram(self , batch, text_pipeline):
        """
        Collate_fn for Skip-Gram model to be used with Dataloader.
        `batch` is expected to be list of text paragrahs.

        Context is represented as N=SKIPGRAM_N_WORDS past words
        and N=SKIPGRAM_N_WORDS future words.

        Long paragraphs will be truncated to contain
        no more that MAX_SEQUENCE_LENGTH tokens.

        Each element in `batch_input` is a middle word.
        Each element in `batch_output` is a context word.
        """
        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = text_pipeline(text)

            if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
                continue

            if MAX_SEQUENCE_LENGTH:
                text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

            for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
                token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
                input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
                outputs = token_id_sequence

                for output in outputs:
                    batch_input.append(input_)
                    batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output

    def get_optimizer_class(self, name: str):
        if name == "Adam":
            return optim.Adam
        else:
            raise ValueError("Choose optimizer from: Adam")
            return

    def get_lr_scheduler(self, optimizer, total_epochs: int, verbose: bool = True):
        """
        Scheduler to linearly decrease learning rate,
        so thatlearning rate after the last epoch is 0.
        """
        lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
        return lr_scheduler

    def save_config(config: dict, model_dir: str):
        """Save config file to `model_dir` directory"""
        config_path = os.path.join(model_dir, "config.yaml")
        with open(config_path, "w") as stream:
            yaml.dump(config, stream)


    def save_vocab(self, vocab, model_dir: str):
        """Save vocab file to `model_dir` directory"""
        vocab_path = os.path.join(model_dir, "vocab.pt")
        torch.save(vocab, vocab_path)



class DATALOADER(Functional):

    def __init__(self) -> None:
        super(DATALOADER , self).__init__()


    def DataLoader(self , model_name, ds_name, ds_type, data_dir, batch_size, shuffle , vocab=None):
        """ Do this bellow, if the user don't load their own vocab"""
        data_iter = self.get_data_iterator(ds_name, ds_type, data_dir)
        if not vocab:
            print(f"you don't have you own vocabulary, you are using vocab performed on {model_name} ...")
            tokenizer = self.get_tokenizer()
            vocab = self.get_vocab(data_iter, tokenizer)

        text_pipeline = lambda x: vocab(tokenizer(x))

        if model_name == "cbow":
            collate_fn = self.collate_cbow
        elif model_name == "skipgram":
            collate_fn = self.collate_skipgram
        else:
            raise ValueError("Choose model from: [cbow , skipgram]")

        dataloader = DataLoader(
            data_iter,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
        )
        return dataloader, vocab