import os
import pandas as pd
from codecs import open


def find_file(root_dir, extension):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                all_files.append(os.path.join(path, file))

    return all_files


def load_folder(path, csv_name):

    data = []
    for polarity in ['pos', 'neg']:

        folder_path = '{}/{}'.format(path, polarity)
        file_paths = find_file(folder_path, '.txt')
        for file_path in file_paths:
            data.append({"text": open(file_path, encoding="utf-8").read().strip(), "label": int(polarity == "pos")})

    df_final = pd.DataFrame(data)

    csv_path = '../sentiment_data/{}.csv'.format(csv_name)
    df_final[['text', 'label']].to_csv(csv_path, index=False, sep="\t", encoding="utf-8")
    print('completed loading data and saving file: {}'.format(csv_name))


load_folder('../sentiment_data/aclImdb/test', 'test')
load_folder('../sentiment_data/aclImdb/train', 'train')
