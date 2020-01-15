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
from transformers import BertTokenizer, BertModel

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# nltk.download("stopwords")
nlp = spacy.load("en")

data_folder = "../../data"
# json_file = f"{data_folder}/amazon_cells_labelled.json"


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_input_length = tokenizer.max_model_input_sizes["bert-base-uncased"]
init_token_index = tokenizer.cls_token_id
end_of_string_token_index = tokenizer.sep_token_id
padding_token_index = tokenizer.pad_token_id
unknown_token_index = tokenizer.unk_token_id


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[: max_input_length - 2]
    return tokens


class BERTGRUSentiment(nn.Module):
    def __init__(
        self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout
    ):
        super().__init__()
        self.bert = bert
        embedding_dim = self.bert.config.to_dict()["hidden_size"]
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout,
        )
        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            )
        else:
            hidden = self.dropout(hidden[-1, :, :])
        output = self.out(hidden)
        return output


MAX_VOCAB_SIZE = 25_000

"""
pylint: disable=invalid-name
pylint: disable=missing-docstring
"""


class Sentiment:
    def __init__(self, model_file: str, *args, **kwargs):
        global MAX_VOCAB_SIZE
        self.device = torch.device("cpu")

        self.TEXT = data.Field(
            batch_first=True,
            use_vocab=False,
            preprocessing=tokenizer.convert_tokens_to_ids,
            init_token=init_token_index,
            eos_token=end_of_string_token_index,
            pad_token=padding_token_index,
            unk_token=unknown_token_index,
        )
        self.LABEL = data.LabelField(dtype=torch.float)
        print("SPLITTING DATA")
        training_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)

        training_data, validation_data = training_data.split(
            random_state=random.seed(SEED)
        )
        BATCH_SIZE = 128
        (
            training_iterator,
            validation_iterator,
            test_iterator,
        ) = data.BucketIterator.splits(
            (training_data, validation_data, test_data),
            batch_size=BATCH_SIZE,
            device=self.device,
        )
        print("BUILDING VOCAB")
        self.LABEL.build_vocab(training_data)
        print("DONE BUILDING VOCAB")
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.25

        bert = BertModel.from_pretrained("bert-base-uncased")
        self.model = BERTGRUSentiment(
            bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT,
        )
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device)
        )
        self.model.eval()

    def eval(self, sentence, min_len=5):
        if len(sentence.split(" ")) < min_len:
            sentence += " a" * (min_len - len(sentence.split(" ")))
            self.model.eval()
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[: max_input_length - 2]
        indexed = (
            [init_token_index]
            + tokenizer.convert_tokens_to_ids(tokens)
            + [end_of_string_token_index]
        )
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(self.model(tensor))
        return max(prediction.item() - 0.1, 0.00001)
