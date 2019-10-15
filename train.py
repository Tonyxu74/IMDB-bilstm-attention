from utils.dataset import GenerateIter
from torchtext import data
from utils.model import LSTMAttention
from torch import optim
from torch import nn
import torch
import tqdm
import numpy as np

'''
Implement embeddings that are trainable, take pretrained weights from GLOVE
https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
implement the predict with a customizable input string
'''


def append_endtag(full_review):
    full_review.append('<end>')
    return full_review


def train():
    continue_train = False

    model = LSTMAttention()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )

    lossfn = nn.CrossEntropyLoss()

    start_epoch = 1

    if continue_train:
        pretrained_dict = torch.load('idunno')['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=append_endtag)
    LABEL = data.Field(sequential=False)

    train_iter = GenerateIter('./sentiment_data/train.csv', TEXT, LABEL, 200)
    val_iter = GenerateIter('./sentiment_data/test.csv', TEXT, LABEL, 200)

    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    for epoch in range(start_epoch, 51):

        pbar = tqdm.tqdm(train_iter)

        pred_classes = []
        ground_truths = []
        n_total = 1
        losses_sum = 0

        # training
        for embedded_review, label in pbar:
            if torch.cuda.is_available():
                embedded_review = embedded_review.cuda()
                label = label.cuda()

            prediction = model(embedded_review)

            pred_class = torch.softmax(prediction, dim=1)
            pred_class = torch.argmax(pred_class, dim=1).cpu().data.numpy().tolist()

            pred_classes.append(pred_class)
            ground_truths.append(label.cpu().data.numpy().tolist())

            loss = lossfn(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_sum += loss
            pbar.set_description('Epoch: {} || Loss: {:.5f} '.format(epoch, losses_sum / n_total))
            n_total += 1

        pred_classes = np.asarray(pred_classes)
        ground_truths = np.asarray(ground_truths)

        accuracy = np.mean((pred_classes == ground_truths)).astype(np.float)

        # validation
        with torch.no_grad():
            model.eval()

            pbar_val = tqdm.tqdm(val_iter)

            pred_classes_val = []
            ground_truths_val = []
            losses_sum_val = 0
            n_total_val = 0

            for embedded_review, label in pbar_val:

                if torch.cuda.is_available():
                    embedded_review = embedded_review.cuda()
                    label = label.cuda()

                prediction = model(embedded_review)

                pred_class = torch.softmax(prediction, dim=1)
                pred_class = torch.argmax(pred_class, dim=1).cpu().data.numpy().tolist()

                pred_classes_val.append(pred_class)
                ground_truths_val.append(label.cpu().data.numpy().tolist())

                loss = lossfn(prediction, label)

                losses_sum_val += loss
                pbar.set_description('Epoch: {} || Val_Loss: {:.5f} '.format(epoch, losses_sum_val / n_total_val))
                n_total_val += 1

            pred_classes_val = np.asarray(pred_classes_val)
            ground_truths_val = np.asarray(ground_truths_val)

            accuracy_val = np.mean((pred_classes_val == ground_truths_val)).astype(np.float)

        print('Epoch: {} || Accuracy: {} || Loss: {} || Val_Acc: {} || Val_Loss: {}'.format(
            epoch, accuracy, losses_sum / n_total, accuracy_val, losses_sum_val / n_total_val
        ))

        if epoch % 5 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './models/IMDB_model_{}.pt'.format(epoch))

        model.train()


if __name__ == '__main__':
    train()





