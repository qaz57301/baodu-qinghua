# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "C:/Users/minut/coding/自然语言处理/train_data.csv",
    "valid_data_path": "C:/Users/minut/coding/自然语言处理/test_data.csv",
    "vocab_path":"chars.txt",
    "model_type":"fast_text",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"C:\Users\minut\coding\自然语言处理\bert-base-chinese",
    "seed": 689
}