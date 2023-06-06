import random
import math
import os

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np

import matplotlib.pyplot as plt
import logging

'''
使用语言模型做文本纠错
以下是用bert的方式
'''
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageModelWithBert(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModelWithBert, self).__init__()
        config = BertConfig.from_json_file(bert_path + "\\config.json")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        _, x = self.bert(x)  # (batch_size, sen_len) -> (batch_size, hidden_size)
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
                target = sentence[i]
                window = self.tokenizer.encode(window, add_special_tokens=False, padding="max_length", max_length=window_size)
                target = self.tokenizer.encode(target, add_special_tokens=False, padding="max_length", max_length=1)
                if not window:
                    window = [0] * window_size
                if not target:
                    target = [0]
                window = torch.LongTensor([window])
                vocab_pred = torch.softmax(self.forward(window), dim=-1)
                prob += math.log(vocab_pred[0][target[0]].item(), 10)
        return 2 ** (- prob / len(sentence))


def get_corpus(corpus_path):
    f = open(corpus_path, "r", encoding="utf-8")
    corpus = f.read()
    f.close()
    return corpus


def build_dataset(batch_size, corpus, tokenizer, window_size):
    dataset_x = []
    dataset_y = []
    for _ in range(batch_size):
        x, y = build_data(corpus, tokenizer, window_size)
        dataset_x.append(x)
        dataset_y.append(y[0])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_data(corpus, tokenizer, window_size):
    start = random.randint(0, len(corpus) - window_size - 1)
    end = start + window_size
    window = corpus[start:end]
    window = tokenizer.encode(window, add_special_tokens=False, padding="max_length", max_length=window_size)
    target = corpus[end]
    target = tokenizer.encode(target, add_special_tokens=False, padding="max_length", max_length=1)
    '''
    试了下，当句首句尾[CLS]、[SEP]关闭时，使用padding和max_length的方式进行补全，如果没有一个在词表里的字，就是个空list，不会进行补全，比如：
    tokenizer.encode(" ", add_special_tokens=False, padding="max_length", max_length=10)
    tokenizer.encode("\n \n", add_special_tokens=False, padding="max_length", max_length=10)
    这两个都是返回空list，这种情况下手动进行补全
    '''
    if not window:
        window = [0] * window_size
    if not target:
        target = [0]
    return window, target


def train(bert_path, corpus_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    corpus = get_corpus(corpus_path)

    epoch = 30
    batch_size = 64
    train_size = 10000
    lr = 1e-5
    window_size = 10

    model = LanguageModelWithBert(bert_path)
    if os.path.exists("nn_model.pth"):
        # 训了30轮，发现loss还能降，继续每次再训几轮看看效果
        model.load_state_dict(torch.load("nn_model.pth"))
        epoch = 5

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
            x, y = build_dataset(batch_size, corpus, tokenizer, window_size)
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

    torch.save(model.state_dict(), "nn_model.pth")


if __name__ == "__main__":
    bert_path = "C:\\yuanzhouli\\AllCodeRelated\\bert\\bert-base-chinese"
    train(bert_path, "corpus.txt")
