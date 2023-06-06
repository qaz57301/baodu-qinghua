# -*- coding: utf-8 -*-

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from config import Config

"""
对数据集【文本分类练习.csv】做数据分析：
    1.统计正负样本数 1:正样本 0：负样本
    2.获取文本平均长度 max_length：40
    3.生成相应的训练集和测试集（根据csv生成json）
        训练集：train_review_label.json
        测试集：valid_review_label.json
"""


class Analyzer:
    def __init__(self, config):
        self.config = config
        self.data_name = self.config["data_name"]
        self.data = pd.read_csv(os.path.join(self.config["data_path"], self.config["data_name"]))

    def analy(self):
        """
        1.统计正负样本数
        """
        print("-" * 10 + "数据分析" + "-" * 10)
        print("%s的样本总数：%d" % (self.data_name, len(self.data)))
        print("\n" + "-" * 10 + "%s的正负样本统计" % self.data_name + "-" * 10)
        print(self.data.groupby("label").count())

        """
        2.获取文本平均长度
        """
        self.data['rv_len'] = self.data['review'].map(lambda x: len(x))
        print("\n" + "-" * 10 + "%s的review字符数统计：" % self.data_name + "-" * 10)
        print(self.data['rv_len'].describe())
        # 将self.data['rv_len']列各个字符串长度的统计结果写入str_lens.csv中
        with open(self.config["str_lens_path"], "w", encoding="utf8", newline="") as f:
            self.data['rv_len'].value_counts().to_csv(f)
        # 通过查看str_lens.csv中的长度分布，选取40作为config["max_length"]的取值
        self.config["max_length"] = 40

        """
        3.生成相应的训练集和测试集
        """
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(self.data['review'], self.data['label'], test_size=1 / 4,
                                                            random_state=self.config["seed"], shuffle=True)
        # 根据csv生成json，分别生成训练集和测试集数据
        print("\n生成训练集数据...")
        build_data_json(x_train, y_train, self.config["train_data_path"])
        print("\n生成测试集数据...")
        build_data_json(x_test, y_test, self.config["valid_data_path"])


def build_data_json(x, y, path):
    with open(path, encoding="utf8", mode="w") as f:
        num = 0
        for i, j in zip(x, y):
            line = json.dumps({"review": i, "label": j}, ensure_ascii=False)
            f.write(line + "\n")
            num += 1
    basename = os.path.basename(path)
    print("生成完毕！\n%s的样本数：%d" % (basename, num))
    data = pd.DataFrame([x, y], index=["review", "label"]).T
    print("-" * 10 + "正负样本统计" + "-" * 10)
    print(data.groupby('label').count())


if __name__ == "__main__":
    Analyzer(Config).analy()
