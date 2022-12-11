import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.layer1 = nn.ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.layer2 = nn.ConvBN(16, 32, 3, 2, device=device, dtype=dtype)

        self.layer3 = nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.layer4 = nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype)

        self.layer5 = nn.ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.layer6 = nn.ConvBN(64, 128, 3, 2, device=device, dtype=dtype)

        self.layer7 = nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        self.layer8 = nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype)

        self.layer9 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Linear(128, 10, device=device, dtype=dtype)

    def forward(self, x):
        res = x
        res = self.layer1(res)
        res = self.layer2(res)
        y = res
        res = self.layer3(res)
        res = self.layer4(res)
        res += y
        res = self.layer5(res)
        res = self.layer6(res)
        y = res
        res = self.layer7(res)
        res = self.layer8(res)
        res += y

        # flatten before linear
        res = res.reshape((np.prod(res.shape)//128, 128))
        res = self.layer9(res)
        res = self.layer10(res)
        res = self.layer11(res)
        return res


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise ValueError('seq_model must be either rnn or lstm')
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """

        x = self.embedding(x)
        x, h = self.seq_model(x, h)
        # print("seq_model:", self.seq_model, type(h))
        x_shape = x.shape
        x = x.reshape((np.prod(x_shape)//self.hidden_size, self.hidden_size))
        x = self.linear(x)
        # print("\nh:", h.shape)
        return x, h


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)