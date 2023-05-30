import torch.nn as nn
import torch
import torch.nn.functional

from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"])
            hidden_size = self.encoder.config.hidden_size

        self.pool_type = config["pooling"]
        self.max_pool = nn.MaxPool1d(config["max_length"])
        self.avg_pool = nn.AvgPool1d(config["max_length"])

        self.classify_layer = nn.Linear(hidden_size, class_num)

        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if self.use_bert:
            x = self.encoder(x)[0]
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        if isinstance(x, tuple):
            x = x[0]

        if self.pool_type == "max":
            x = self.max_pool(x.transpose(1, 2)).squeeze()
        else:
            x = self.avg_pool(x.transpose(1, 2)).squeeze()

        y_pred = self.classify_layer(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return torch.softmax(y_pred, dim=-1)

    def name(self):
        return "{}-batch_size-{}".format(self.config["model_type"], str(self.config["batch_size"]))


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        padding = int((kernel_size - 1) / 2)
        self.cnn_layer = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        return self.cnn_layer(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn_layer = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn_layer(x)
        gate = self.gate(x)
        gate = torch.sigmoid(gate)
        return torch.mul(a, gate)


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.cnn = CNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


def choose_optimizer(config, model: nn.Module):
    lr = config["use_bert_lr"] if model.use_bert else config["learning_rate"]
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
