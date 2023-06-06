import math
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='YouYuan')

import torch.nn as nn
import torch


'''
自行设计一个规则，完成一个基于神经网络的nlp文本分类任务。
'''


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
        self.lstm_layer = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.activation = torch.sigmoid
        self.liner_layer = nn.Linear(hidden_size, 3)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentence_length, hidden_size)

        _, (x, _) = self.lstm_layer(x)  # (batch_size, hidden_size)
        x = x.squeeze()
        x = self.activation(x)

        x = self.liner_layer(x)
        y_pred = self.activation(x)

        if y is not None:
            return self.loss(x, y)
        else:
            return y_pred


def build_vocab():
    chars = "好坏中一二三四五六七八九十abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(chars) + 1
    return vocab


def build_sample(vocab, sentence_length):
    x = random.choices(list(vocab.keys()), k=sentence_length)
    if "好" in x:
        y = 0
    elif "坏" in x:
        y = 1
    else:
        y = 2
    x = [vocab.get(e) for e in x]
    return x, y


def build_dataset(batch_size, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(batch_size):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def evaluate(model, vocab, sentence_length):
    model.eval()

    correct = 0
    wrong = 0
    x, y = build_dataset(200, vocab, sentence_length)
    y_pred = model(x)
    for i in range(len(x)):
        if y[i] == torch.argmax(y_pred[i]):
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)


epoch_num = 30
hidden_size = 64
sentence_length = 8
lr = 0.001


def main():
    vocab = build_vocab()

    model = Model(len(vocab), hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    batch_size = 32
    sample_size = 500
    for epoch in range(epoch_num):
        epoch_loss = []
        model.train()
        for batch in range(math.ceil(sample_size / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()

            epoch_loss.append(loss.item())
        print("=========\n第%d轮平均loss：%f" % (epoch + 1, np.mean(epoch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        print("准确率：%f" % (acc))
        log.append([np.mean(epoch_loss), acc])

    plt.plot(range(len(log)), [e[0] for e in log], label="loss曲线")
    plt.plot(range(len(log)), [e[1] for e in log], label="acc曲线")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "week3.pth")


def predict(tests):
    vocab = build_vocab()
    model = Model(len(vocab), hidden_size)
    model.load_state_dict(torch.load("week3.pth"))
    model.eval()
    dataset_x = []
    for test in tests:
        x = [vocab.get(e, vocab['unk']) for e in test]
        x = x[:sentence_length]
        x += [0] * (sentence_length - len(x))
        dataset_x.append(x)
    dataset_x = torch.LongTensor(dataset_x)
    y_pred = torch.argmax(model(dataset_x), dim=1)
    dic = {0: "带好字", 1: "带坏字", 2: "都不带"}
    for index, test in enumerate(tests):
        print("输入：%s，   预测第%d类（%s）" % (test, y_pred[index], dic[y_pred[index].item()]))


if __name__ == "__main__":
    # main()
    predict(["坏好", "22A坏", "中国", "dddd坏", "aau好", "好好好", "坏坏坏", "A"])

