

Config = {
    "data_path": "./文本分类练习.csv",
    "model_path": "./output",
    "vocab_path": "char.txt",
    "batch_size": 64,
    "model_type": "rnn",
    "kernel_size": 3,
    "max_length": 25,
    "hidden_size": 128,
    "learning_rate": 1e-3,
    "use_bert_lr": 1e-5,
    "num_layers": 1,
    "epoch": 15,
    "pooling": "max",
    "optimizer": "adam",
    "pretrain_model_path": "C:\\yuanzhouli\\AllCodeRelated\\bert\\bert-base-chinese",
    "seed": 123,
    "multi_config": {
        "batch_size": [64, 128],
        "model_type": ["rnn", "bert", "cnn", "lstm", "gated_cnn"],
    }
}