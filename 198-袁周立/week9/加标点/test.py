import torch
import json
import re

from model import Model
from config import Config
from collections import defaultdict
from transformers import BertTokenizer

def test(test_strings):
    Config["class_num"] = 6
    model = Model(Config)
    model.load_state_dict(torch.load("./output/model.pth"))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])

    schema = json.loads(open(Config["schema_path"], "r", encoding="utf-8").read())
    schema_index_to_label = dict([y, x] for x, y in schema.items())

    data = []
    for test_string in test_strings:
        input_ids = tokenizer.encode(test_string, padding='max_length', max_length=Config["max_len"], truncation=True, add_special_tokens=False)
        data.append(input_ids)
    tensor = torch.LongTensor(data)
    pred = model(tensor)
    pred = torch.argmax(pred, dim=-1)

    for index, labels in enumerate(pred):
        cur_sentence = test_strings[index]
        labels = "".join([str(e) for e in labels.tolist()])
        labels = labels[:len(cur_sentence)]
        print("当前句子：" + test_strings[index])
        cur_result = ""
        for i in range(len(cur_sentence)):
            cur_result += cur_sentence[i] + schema_index_to_label[int(labels[i])]
        print("输出：" + cur_result)
        print("-------------")



if __name__ == "__main__":
    test(["衡山是五岳之一每年都有很多人去参观旅游小明小红小强今年国庆节就准备去衡山游玩也是给自己放个假",
          "每一周都是从星期一开始星期天结束每一月都是一号开始但是结束日期可以是28号29号30号31号这倒也有意思",
          "这些未来酒店的一个最大特点就是它们的所在地陆地已然不是唯一的选择它们的身影将出现在之前不敢想象的海下天空甚至地球以外的太空"
          ])