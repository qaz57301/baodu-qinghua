# -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import logging
from config import Config
from analyse import Analyzer
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from predict import Predictor
import csv

# 创建保存日志的目录
if not os.path.isdir(Config["log_path"]):
    os.mkdir(Config["log_path"])
# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format=Config["log_format"],
                    filename=os.path.join(Config["log_path"], Config["log_name"]))
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    acc = None
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    # 预测100条耗时
    time = Predictor(config, model, logger).pred()
    # 保存模型参数
    save_model_weights(config, model)

    return acc, time


# 根据当前配置参数，递归创建目录，来保存对应配置的模型参数
def save_model_weights(config, model):
    base_name = "epoch_%d.pth" % config["epoch"]
    model_path = os.path.join(config["model_path"], config["model_type"],
                              ("%.0e" % config["learning_rate"]).replace("0", ""),
                              "hidden_%d" % config["hidden_size"],
                              "batch_%d" % config["batch_size"], config["pooling_style"])
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, base_name)
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    print(model_path, "模型参数已保存！")


# 将模型效果对比生成csv文件
def build_result_contrast_file(config, headers, datas):
    path = os.path.join(Config["model_path"], config["result_file"])
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerows(datas)


if __name__ == "__main__":
    # 对【文本分类练习.csv】做数据分析
    Analyzer(Config).analy()

    # main(Config)

    # headers = ["model", "learning_rate", "hidden_size", "batch_size", "pooling_style", "acc", "time(预测100条耗时)"]
    # datas = []
    #
    # for model in ["fast_text", "cnn", "gated_cnn"]:
    #     Config["model_type"] = model
    #     acc, time = main(Config)
    #     print("最后一轮准确率：", acc, "当前配置：", Config)
    #     datas.append({"model": Config["model_type"], "learning_rate": Config["learning_rate"],
    #                   "hidden_size": Config["hidden_size"], "batch_size": Config["batch_size"],
    #                   "pooling_style": Config["pooling_style"], "acc": acc,
    #                   "time(预测100条耗时)": str(time) + "秒"})
    #
    # # 生成模型效果对比文件
    # build_result_contrast_file(Config, headers, datas)


    headers = ["model", "learning_rate", "hidden_size", "batch_size", "pooling_style", "acc", "time(预测100条耗时)"]
    datas = []

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn", "rcnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style
                        acc, time = main(Config)
                        print("最后一轮准确率：", acc, "当前配置：", Config)
                        datas.append({"model": Config["model_type"], "learning_rate": Config["learning_rate"],
                                      "hidden_size": Config["hidden_size"], "batch_size": Config["batch_size"],
                                      "pooling_style": Config["pooling_style"], "acc": acc,
                                      "time(预测100条耗时)": str(time) + "秒"})

    # 生成模型效果对比文件
    build_result_contrast_file(Config, headers, datas)
