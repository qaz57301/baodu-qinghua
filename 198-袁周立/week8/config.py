

Config = {
    "train_data_path": "./data/train.json",
    "valid_data_path": "./data/valid.json",
    "schema_path": "./data/schema.json",
    "model_path": "./output",
    "vocab_path": "./chars.txt",
    "batch_size": 64,
    "data_num": 10000,
    "model_type": "lstm",
    "kernel_size": 3,
    "max_length": 20,
    "hidden_size": 128,
    "learning_rate": 1e-4,
    "num_layers": 1,
    "epoch": 15,
    "pooling": "max",
    "optimizer": "adam",
    "pretrain_model_path": "C:\\yuanzhouli\\AllCodeRelated\\bert\\bert-base-chinese",
    "seed": 123,
}