import torch.nn as nn
import torch.optim as optim

from src.utils.constants import ( EMBED_DIMENSION , EMBED_MAX_NORM)

class WORD2VECTCBOW(nn.Module):
    def __init__(self,vocab_size: int) -> None:
        super(self, WORD2VECTCBOW).__init__()
        self.embddings = nn.Embedding(num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(input_dim = EMBED_DIMENSION, output_dim = vocab_size)


    def forward(self , inputs_):
        x = self.embddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

class WORD2VECTCBOWSKIPGRAM(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(WORD2VECTCBOWSKIPGRAM, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x