import torch
from torch.utils.data import DataLoader
import json
from transformers import BertTokenizer


class DataContext:
    def __init__(self, config):
        self.config = config
        self.max_length = config["max_len"]
        self.vocab = self.load_vocab(config["vocab_path"])
        self.schema = self.load_schema(config["schema_path"])
        if self.config["use_bert"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"], add_special_tokens=False)

        self.train_data = self.__load_data(config["trade_data_path"])
        self.valid_data = self.__load_data(config["valid_data_path"])

        config["vocab_size"] = len(self.vocab)
        config["class_num"] = len(self.schema)

    def load_vocab(self, vocab_path):
        vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                vocab[line.strip()] = len(vocab) + 1
        return vocab

    def load_schema(self, schema_path):
        return json.loads(open(schema_path, "r", encoding="utf-8").read())

    def __load_data(self, data_path):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            samples = f.read().split("\n\n")
            for sample in samples:
                sentence = ""
                labels = []
                for line in sample.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.strip().split(" ")
                    sentence += char
                    labels.append(self.schema[label])
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return data

    def encode_sentence(self, sentence):
        if self.config["use_bert"]:
            return self.tokenizer.encode(sentence, padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = [self.vocab.get(char, self.vocab["[UNK]"]) for char in sentence]
        input_ids = self.padding(input_ids)
        return input_ids

    def padding(self, input_ids, padding_index=0):
        input_ids = input_ids[:self.max_length]
        input_ids += [padding_index] * (self.max_length - len(input_ids))
        return input_ids

    class DataGenerator:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, item):
            return self.data[item]

        def __len__(self):
            return len(self.data)

    def get_train_data(self, shuffle=True):
        dg = DataContext.DataGenerator(self.train_data)
        dl = DataLoader(dg, batch_size=self.config["batch_size"], shuffle=shuffle)
        return dl

    def get_valid_data(self, shuffle=True):
        dg = DataContext.DataGenerator(self.valid_data)
        dl = DataLoader(dg, batch_size=self.config["batch_size"], shuffle=shuffle)
        return dl
