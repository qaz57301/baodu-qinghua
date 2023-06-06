# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "log_path": "log",
    "log_name": "main_模型训练和评估日志.log",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "result_file": "模型效果对比.csv",
    "data_path": "../data",
    "data_name": "文本分类练习.csv",
    "str_lens_path": "../data/str_lens.csv",
    "model_path": "output",
    "train_data_path": "../data/train_review_label.json",
    "valid_data_path": "../data/valid_review_label.json",
    "vocab_path": "chars.txt",
    "model_type": "fast_text",
    "max_length": 40,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adam",
    # "learning_rate": 1e-3,
    "learning_rate": 1e-5,
    "pretrain_model_path": r"F:\PycharmProjects\ClassProject\week6_Language_Model\huggingface_model\bert-base-Chinese",
    "seed": 987
}
