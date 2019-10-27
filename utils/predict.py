from utils.dataset import GenerateIter
from torchtext import data
from utils.model import LSTMAttention
from torch import optim
from torch import nn
import torch


def append_endtag(full_review):
    full_review.append('<end>')
    return full_review


def predict(string, epoch):

    model = LSTMAttention().cuda()
    model.batch_size = 1

    pretrained_dict = torch.load('../models/IMDB_model_{}.pt'.format(epoch))['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=append_endtag)
    LABEL = data.Field(sequential=False)

    train_iter = GenerateIter('../sentiment_data/train.csv', TEXT, LABEL, 200)

    probability, pred_class = model.predict(string, TEXT)
    print(probability)
    print(pred_class)


if __name__ == "__main__":
    predict('This movie is the best.', epoch=15)
