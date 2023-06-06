import random

import torch
from torch.utils.data import DataLoader
import json


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.load_data(config)

    def load_data(self, config):
        self.schema = json.loads(open(config["schema_path"], "r", encoding="utf-8").read())
        self.data = []
        self.vocab = self.get_vocab(config)
        with open(config["train_data_path"], "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                self.data.append([line["questions"], line["target"]])

    def get_questions_all_dict(self):
        questions = []
        question_target_dict = {}
        question_no = 0
        for data in self.data:
            target = self.schema[data[1]]
            for e in data[0]:
                questions.append(self.encode_sentence(e))
                question_target_dict[question_no] = target
                question_no += 1
        return torch.LongTensor(questions), question_target_dict

    def encode_sentence(self, sentence):
        encode = [self.vocab.get(e, self.vocab["[UNK]"]) for e in sentence]
        encode = self.padding(encode)
        return encode

    def padding(self, encode):
        max_len = self.config["max_length"]
        encode = encode[:max_len]
        encode += (max_len - len(encode)) * [0]
        return encode

    def get_vocab(self, config):
        vocab = {}
        with open(config["vocab_path"], "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                vocab[line] = len(vocab) + 1
        config["vocab_size"] = len(vocab)
        return vocab

    def __getitem__(self, item):
        data1, data2 = random.sample(self.data, 2)
        questions1 = data1[0]
        if len(questions1) < 2:
            return self.__getitem__(item)
        x1, x2 = random.sample(questions1, 2)
        y = random.choice(data2[0])
        x1 = self.encode_sentence(x1)
        x2 = self.encode_sentence(x2)
        y = self.encode_sentence(y)
        return [torch.LongTensor(x1), torch.LongTensor(x2), torch.LongTensor(y)]

    def __len__(self):
        return self.config["data_num"]


class ValidDataGenerator:
    def __init__(self, config, vocab):
        self.vocab = vocab
        self.config = config
        self.load_data(config)

    def load_data(self, config):
        self.valid_data = []
        self.schema = json.loads(open(config["schema_path"], "r", encoding="utf-8").read())
        with open(config["valid_data_path"], "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                question = line[0]
                target = line[1]
                question = self.encode_sentence(question)
                target = self.schema.get(target)
                self.valid_data.append([torch.LongTensor(question), target])

    def encode_sentence(self, sentence):
        encode = [self.vocab.get(e, self.vocab["[UNK]"]) for e in sentence]
        encode = self.padding(encode)
        return encode

    def padding(self, encode):
        max_len = self.config["max_length"]
        encode = encode[:max_len]
        encode += (max_len - len(encode)) * [0]
        return encode

    def __getitem__(self, item):
        return self.valid_data[item]

    def __len__(self):
        return len(self.valid_data)


def load_train_data(config):
    dg = DataGenerator(config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=True)
    return dl


def load_valid_data(config):
    dg_train = DataGenerator(config)
    dg = ValidDataGenerator(config, dg_train.vocab)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=True)
    return dl, dg_train.vocab, dg_train.get_questions_all_dict()[0], dg_train.get_questions_all_dict()[1]