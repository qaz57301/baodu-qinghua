# -*- coding: utf-8 -*-

import torch
from config import Config
from transformers import BertModel, BertTokenizer
import json
import jieba
from torch.utils.data import DataLoader

"""
DataGernerator:
1、init 参数。 config, path = data_path(ner_data/train.txt),vocab_size, sentences=[], schema=load_schema
2、load: 
3、encode_sentences
4、padding
5、__len__, __getitem__
6、load_schema

load_vocab
load_data
"""
class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.vocab_size = len(self.vocab)
        self.sentences=[]
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.max_length = config["max_length"]
        # 加载 bert 的分字器，注意第二个参数，阻止自动向编码序列中加入cls和sep
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model"], add_special_tokens=False)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip()=="":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"]=="word.txt":
            for word in jieba.lcut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或者截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token]*(self.config["max_length"]-len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, schema_path):
        with open(schema_path,encoding="utf8") as f:
            return json.load(f)

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index+1
    return token_dict

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    data_path = "ner_data/train.txt"
    dg = DataGenerator(data_path, Config)
    dl = load_data(data_path, config=Config, shuffle=False)
    # for key, value in enumerate(dl):
    #     print(key, value)
    for input_ids, label in dg:
        print(input_ids, label)
