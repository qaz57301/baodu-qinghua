# -*- coding: utf-8 -*-

import torch
import time
import random
from transformers import BertTokenizer
import json

"""
预测100条耗时
"""


class Predictor:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.x, self.y = self.load()

    def load(self):
        path = self.config["valid_data_path"]
        num = sum((1 for _ in open(path, encoding="utf8")))
        random.seed(self.config['seed'])
        # 随机生成100条数据的索引
        index = set(random.sample(list(range(num)), 100))
        x = []
        y = []
        with open(path, encoding="utf8") as f:
            for i, line in enumerate(f):
                # 根据数据索引生成100条数据用来测试
                if i in index:
                    line = json.loads(line)
                    review = line["review"]
                    label = line["label"]
                    if self.config["model_type"] == "bert":
                        input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],
                                                         pad_to_max_length=True)
                    else:
                        input_id = self.encode_sentence(review)
                    x.append(input_id)
                    y.append([label])
        return torch.LongTensor(x), torch.LongTensor(y)

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def pred(self):
        self.model.eval()
        if torch.cuda.is_available():
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        # 预测100条数据耗时
        begin = time.time()
        with torch.no_grad():
            self.model(self.x)  # 不输入labels，使用模型当前参数进行预测
        end = time.time()
        self.logger.info("预测100条耗时：" + str(end - begin) + " 秒"+"\n\n" + "-" * 80 + "\n")
        return end - begin


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict
