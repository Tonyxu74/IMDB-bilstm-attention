from torchtext import data
from utils.model import LSTMAttention
import pandas as pd
from opts import args
from torchtext.vocab import GloVe
import torch


def append_endtag(full_review):
    full_review.append('<end>')
    return full_review


def predict(string, epoch):
    model = LSTMAttention().cuda()
    model.batch_size = 1

    pretrained_dict = torch.load(
        '../models/IMDB_model_{}.pt'.format(epoch)
    )['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    TEXT = torch.load('vocab_field.pt')

    probability, pred_class = model.predict(string, TEXT)
    print(probability)
    print(pred_class)


def build_vocab(path):

    dataframe = pd.read_csv(path, sep="\t", encoding="utf-8")
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=append_endtag)
    dataset = []

    for i in range(dataframe.shape[0]):
        dataset.append({
            'text': text_field.preprocess(dataframe['text'][i]),
            'label': dataframe['label'][i]
        }
    )

    text_field.build_vocab([t['text'] for t in dataset], vectors=GloVe(name='6B', dim=args.embedding_dims))
    torch.save(text_field, 'vocab_field.pt')


if __name__ == "__main__":
    # build_vocab('../sentiment_data/train.csv')
    predict("The actors was brilliant, and the story was quite good.", epoch=25)
