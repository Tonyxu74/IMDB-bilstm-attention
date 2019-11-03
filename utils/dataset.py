from opts import args
import pandas as pd
from torchtext.vocab import GloVe
from torchtext import data
import torch.utils.data as torchdata
import torch


class IMDB_dataset(data.Dataset):

    def __init__(self, path, text_field, label_field, pad_length, eval):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        self.eval = eval
        self.dataset = []

        dataframe = pd.read_csv(path, sep="\t", encoding="utf-8")

        for i in range(dataframe.shape[0]):
            self.dataset.append({
                'text': text_field.preprocess(dataframe['text'][i]),
                'label': dataframe['label'][i]
            })

        if not self.eval:
            text_field.build_vocab(
                [t['text'] for t in dataset],
                vectors=GloVe(name='6B', dim=args.embedding_dims)
            )
            label_field.build_vocab(t['label'] for t in dataset)

        self.TEXT = text_field
        self.LABEL = label_field
        self.pad_length = pad_length

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]['text']
        if len(text) > self.pad_length:
            text = text[:self.pad_length]
        else:
            text = text + ['<pad>'] * (self.pad_length - len(text))
        embedding = [
            self.TEXT.vocab.vectors[self.TEXT.vocab.stoi[t]] for t in text
        ]
        embedding = torch.stack(embedding)
        label = self.dataset[index]['label']

        return embedding, label


def GenerateIter(
    path,
    text_field,
    label_field,
    pad_length,
    eval=False,
    shuffle=True
):
    params = {
        'batch_size': args.batch_size,
        'shuffle': shuffle,
        'num_workers': 0,
        'pin_memory': False,
        'drop_last': False,
    }

    return torchdata.DataLoader(
        IMDB_dataset(
            path,
            text_field=text_field,
            label_field=label_field,
            pad_length=pad_length,
            eval=eval
        ),
        **params
    )
