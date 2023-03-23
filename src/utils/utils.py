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

class Functional():
    def __init__(self) -> None:
        pass

    def get_data_iterator(ds_name, ds_type, data_dir):
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


class DATALOADER(Functional):

    def __init__(self) -> None:
        super(self, DATALOADER).__init__()


    def DataLoader(self , model_name, ds_name, ds_type, data_dir, batch_size, shuffle , vocab=None):
        """ Do this bellow, if the user don't load their own vocab"""
        if not vocab:
            print(f"you don't have you own, you are using vocab performed on {model_name} ...")
            data_iter = self.get_data_iterator(ds_name, ds_type, data_dir)
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