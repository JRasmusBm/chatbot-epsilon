"""
Function for using a sentiment pytorch model.
pylint: disable=invalid-name
"""
import random

import numpy as np
import nltk
import spacy
import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# nltk.download("stopwords")
nlp = spacy.load("en")

data_folder = "../../data"
# json_file = f"{data_folder}/amazon_cells_labelled.json"


class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        self.conv_0 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[0], embedding_dim),
        )
        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[1], embedding_dim),
        )
        self.conv_2 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[2], embedding_dim),
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved_0 = nn.functional.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = nn.functional.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = nn.functional.relu(self.conv_2(embedded).squeeze(3))
        pooled_0 = nn.functional.max_pool1d(
            conved_0, conved_0.shape[2]
        ).squeeze(2)
        pooled_1 = nn.functional.max_pool1d(
            conved_1, conved_1.shape[2]
        ).squeeze(2)
        pooled_2 = nn.functional.max_pool1d(
            conved_2, conved_2.shape[2]
        ).squeeze(2)
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        return self.fc(cat)


MAX_VOCAB_SIZE = 25_000

"""
pylint: disable=invalid-name
pylint: disable=missing-docstring
"""


class Sentiment:
    def __init__(self, model_file: str, *args, **kwargs):
        global MAX_VOCAB_SIZE
        self.device = torch.device("cpu")
        self.TEXT = data.Field(tokenize="spacy", batch_first=True,)
        self.LABEL = data.LabelField(dtype=torch.float)
        self.fields = dict(
            text=("text", self.TEXT), label=("label", self.LABEL),
        )
        print("SPLITTING DATA")
        training_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)
        training_data, validation_data = training_data.split(
            random_state=random.seed(SEED)
        )
        print("BUILDING VOCAB")
        self.TEXT.build_vocab(
            training_data,
            vectors="glove.6B.100d",
            unk_init=torch.Tensor.normal_,
            max_size=MAX_VOCAB_SIZE,
        )
        print("DONE BUILDING VOCAB")
        self.LABEL.build_vocab(training_data)
        INPUT_DIM = len(self.TEXT.vocab)
        EMBEDDING_DIM = 100
        N_FILTERS = 100
        FILTER_SIZES = [3, 4, 5]
        OUTPUT_DIM = 1
        DROPOUT = 0.5
        PADDING_INDEX = self.TEXT.vocab.stoi[self.TEXT.pad_token]
        MAX_VOCAB_SIZE = 25_000

        self.model = CNN(
            INPUT_DIM,
            EMBEDDING_DIM,
            N_FILTERS,
            FILTER_SIZES,
            OUTPUT_DIM,
            DROPOUT,
            PADDING_INDEX,
        )
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device)
        )

    def eval(self, sentence, min_len=5):
        self.model.eval()
        if len(sentence.split(" ")) < min_len:
            sentence += " a" * (min_len - len(sentence.split(" ")))
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(self.model(tensor))
        return prediction.item()
