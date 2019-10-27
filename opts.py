import argparse

parser = argparse.ArgumentParser()

######################## Model parameters ########################

parser.add_argument('--classes', default=2, type=int,
                    help='# of classes')

parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='momentum for sgd, beta1 for adam')

parser.add_argument('--num_epochs', default=50, type=int,
                    help='epochs to train for')
parser.add_argument('--embedding_dims', default=100, type=int,
                    help='dimensions for the glove embedding')
parser.add_argument('--hidden_dims', default=128, type=int,
                    help='dimensions of hidden outputs')
parser.add_argument('--num_layers', default=2, type=int,
                    help='number of bilstm layers')
parser.add_argument('--review_length', default=200, type=int,
                    help='length to pad or regularize review length')

parser.add_argument('--batch_size', default=8, type=int,
                    help='input batch size')


args = parser.parse_args()