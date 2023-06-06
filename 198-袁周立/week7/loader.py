import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os


class DataGenerator:
    def __init__(self, config):
        self.data_path = config["data_path"]
        self.config = config
        self.load_csv()

    def load_csv(self):
        self.data = []

        np_data = pd.read_csv(self.data_path).values
        if not os.path.exists(self.config["vocab_path"]):
            self.generate_vocab_file(np_data[:, 1])

        self.vocab = self.load_vocab(self.config["vocab_path"])

        class_set = set()
        for e in np_data:
            label, sentence = e[0], e[1]
            sentence_encode = self.encode_sentence(sentence)
            self.data.append([torch.LongTensor(sentence_encode), torch.LongTensor([label])])
            class_set.add(label)

        self.config["class_num"] = len(class_set)
        return

    def generate_vocab_file(self, sentences):
        vocab_set = set("".join(sentences))
        with open(self.config["vocab_path"], "w", encoding="utf-8") as f:
            for index, char in enumerate(vocab_set):
                f.write(char)
                f.write("\n")

    def load_vocab(self, path):
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    vocab[line] = len(vocab) + 1
        vocab["unk"] = len(vocab) + 1
        self.config["vocab_size"] = len(vocab)
        return vocab

    def encode_sentence(self, sentence):
        encode = [self.vocab.get(e, self.vocab["unk"]) for e in sentence]
        encode = self.padding(encode)
        return encode

    def padding(self, vector):
        max_len = self.config["max_length"]
        vector = vector[:max_len]
        vector += [0] * (max_len - len(vector))
        return vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(config, shuffle=True):
    dataGenerator = DataGenerator(config)
    dl = DataLoader(dataGenerator, batch_size=config["batch_size"], shuffle=shuffle)
    return dl