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

    TEXT = torch.load('vocab_field.pt')

    probability, pred_class = model.predict(string, TEXT)
    print(probability)
    print(pred_class)


def build_vocab():
    print("todo")


if __name__ == "__main__":
    predict('This movie is bad.', epoch=10)
