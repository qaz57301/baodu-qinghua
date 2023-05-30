import random
import os
import math

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import logging

'''
使用语言模型做文本纠错
以下是用lstm的方式
'''
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageModel(nn.Module):
    def __init__(self, vocab, hidden_size):
        super(LanguageModel, self).__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=0.1, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, hidden_size)
        _, (x, _) = self.lstm(x)
        x = x.squeeze()
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

    def cal_ppl(self, sentence, window_size):
        prob = 0
        with torch.no_grad():
            for i in range(1, len(sentence)):
                start = max(0, i - window_size)
                window = sentence[start:i]
                window = [self.vocab.get(e, self.vocab.get("unk")) for e in window]
                window = torch.LongTensor([window])
                target = sentence[i]
                target = self.vocab.get(target, self.vocab.get("unk"))
                vocab_pred = torch.softmax(self.forward(window), dim=-1)
                prob += math.log(vocab_pred[target].item(), 10)
        return 2 ** (- prob / len(sentence))


def get_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for word in f.readlines():
            word = word.strip()
            vocab[word] = len(vocab)
    vocab["unk"] = len(vocab)
    return vocab


def get_corpus(corpus_path):
    f = open(corpus_path, "r", encoding="utf-8")
    corpus = f.read()
    f.close()
    return corpus


def build_dataset(batch_size, vocab, corpus, window_size):
    dataset_x = []
    dataset_y = []
    for _ in range(batch_size):
        x, y = build_data(corpus, vocab, window_size)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_data(corpus, vocab, window_size):
    start = random.randint(0, len(corpus) - window_size - 1)
    end = start + window_size
    window = corpus[start:end]
    window = [vocab.get(e, vocab["unk"]) for e in window]
    target = corpus[end]
    target = vocab.get(target, vocab["unk"])
    return window, target

window_size = 10
hidden_size = 128


def train(vocab_path, corpus_path):
    vocab = get_vocab(vocab_path)
    corpus = get_corpus(corpus_path)

    epoch = 30
    batch_size = 64
    train_size = 10000
    lr = 0.001


    model = LanguageModel(vocab, hidden_size)
    if os.path.exists("model.pth"):
        # 训了30轮，发现loss还能降，继续每次再训几轮看看效果
        model.load_state_dict(torch.load("model.pth"))
        epoch = 10

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_watch = []
    for epoch_num in range(epoch):
        logger.info("开始第%d轮训练" % (epoch_num + 1))
        losses = []
        for batch in range(int(train_size / batch_size)):
            logger.info("第%d轮训练，batch进度：%d/%d" % (epoch_num + 1, batch + 1, int(train_size / batch_size)))
            x, y = build_dataset(batch_size, vocab, corpus, window_size)
            if cuda_flag:
                x = x.cuda()
                y = y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        average_loss = np.average(losses)
        logger.info("本轮训练平均loss：%.4f" % average_loss)
        loss_watch.append(average_loss)

    plt.plot([i for i in range(epoch)], loss_watch, label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train("vocab2.txt", "corpus.txt")
