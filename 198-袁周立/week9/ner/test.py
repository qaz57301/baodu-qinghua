import torch
import re

from model import NerModel
from config import Config
from collections import defaultdict
from transformers import BertTokenizer

def test(test_strings):
    Config["class_num"] = 9
    model = NerModel(Config)
    model.load_state_dict(torch.load("./output/ner_model.pth"))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(Config["bert_path"], add_special_tokens=False)


    data = []
    for test_string in test_strings:
        input_ids = tokenizer.encode(test_string, padding='max_length', max_length=Config["max_len"], truncation=True)
        data.append(input_ids)
    tensor = torch.LongTensor(data)
    pred = model(tensor)
    pred = torch.argmax(pred, dim=-1)

    for index, labels in enumerate(pred):
        labels = "".join([str(e) for e in labels.tolist()])
        labels = labels[:len(test_strings[index])]
        print("当前句子：" + test_strings[index])
        # print(labels)
        for location in re.finditer("(04+)", labels):
            start, end = location.span()
            print("识别出LOCATION实体：{}，位置{}至{}，".format(test_strings[index][start:end], str(start), str(end)))
        for organization in re.finditer("(15+)", labels):
            start, end = organization.span()
            print("识别出ORGANIZATION实体：{}，位置{}至{}，".format(test_strings[index][start:end], str(start), str(end)))
        for person in re.finditer("(26+)", labels):
            start, end = person.span()
            print("识别出PERSON实体：{}，位置{}至{}，".format(test_strings[index][start:end], str(start), str(end)))
        for time in re.finditer("(37+)", labels):
            start, end = time.span()
            print("识别出TIME实体：{}，位置{}至{}，".format(test_strings[index][start:end], str(start), str(end)))
        print("-------------")



if __name__ == "__main__":
    test(["今天小明去天天步行街游玩了",
          "小强认为星期五下午是每一个礼拜中特殊的时刻",
          "天安门是中国非常出名的一个地方",
          "衡山是五岳之一，每年都有很多人去参观旅游"
          ])