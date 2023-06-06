import torch.nn as nn
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import jieba
import os


'''
尝试实现基于lstm的分词
'''
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WordCutModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(WordCutModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  batch_first=True,
                                  num_layers=2,
                                  dropout=0.1)
        self.linear_layer = nn.Linear(hidden_size, 2)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        x = self.embedding(x)   # (batch_size, sentence_len) -> (batch_size, sentence_len, hidden_size)
        x, _ = self.lstm_layer(x)
        y_pred = self.linear_layer(x)
        if y is not None:
            return self.loss(y_pred.view(-1, 2), y.view(-1))
        else:
            return y_pred


class Dataset:
    def __init__(self, vocab, sentence_length, corpus_path):
        self.vocab = vocab
        self.sentence_length = sentence_length
        self.corpus_path = corpus_path
        self.load()

    def load(self):
        self.data = []
        with open(self.corpus_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                line = line.strip()
                x = self.sentence_to_index(line)
                label = self.sentence_to_label(line)
                x, label = self.padding(x, label)
                self.data.append([torch.LongTensor(x), torch.LongTensor(label)])

    def sentence_to_index(self, line):
        index = [self.vocab.get(e, self.vocab["unk"]) for e in line]
        return index

    def sentence_to_label(self, line):
        words = jieba.lcut(line)
        labels = [0] * len(line)
        point = 0
        for word in words:
            point += len(word)
            labels[point - 1] = 1
        return labels

    def padding(self, x, label):
        x = x[:self.sentence_length]
        x += [0] * (self.sentence_length - len(x))
        label = label[:self.sentence_length]
        label += [-100] * (self.sentence_length - len(label))
        return x, label

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            vocab[line] = len(vocab) + 1
    vocab["unk"] = len(vocab) + 1
    return vocab


def build_dataset(vocab, batch_size, sentence_length, corpus_path):
    dataset = Dataset(vocab, sentence_length, corpus_path)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader


vocab = get_vocab("./chars.txt")

sentence_length = 40
hidden_size = 128
lr = 0.001
batch_size = 256


def main():
    model = WordCutModel(len(vocab), hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_num = 30
    if os.path.exists("week4.pth"):
        # 训了30轮，发现loss还能降，继续每次再训几轮看看效果
        model.load_state_dict(torch.load("week4.pth"))
        epoch_num = 5

    loss_watch = []
    for epoch in range(epoch_num):
        logger.info("开始第%d轮训练" % (epoch + 1))
        loss_list = []
        for x, y in build_dataset(vocab, batch_size, sentence_length, "./corpus.txt"):
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            loss_list.append(loss.item())
        avg_loss = np.average(loss_list)
        logger.info("平均loss：%f" % (avg_loss))
        loss_watch.append(avg_loss)
    plt.plot(range(epoch_num), loss_watch, label='loss')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "week4.pth")

def predict(test_string):
    model = WordCutModel(len(vocab), hidden_size)
    model.load_state_dict(torch.load("week4.pth"))
    model.eval()

    x = [vocab.get(e, vocab.get("unk")) for e in test_string]
    x = torch.LongTensor([x])
    with torch.no_grad():
        pred = model(x)
        pred = pred[0]
        for index, e in enumerate(pred):
            if e[0] < e[1]:
                print(test_string[index], end=" ")
            else:
                print(test_string[index], end="")


if __name__ == '__main__':
    # main()

    predict("自然语言处理真好玩")