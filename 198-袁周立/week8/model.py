import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)

    def forward(self, x1, x2=None, y=None):
        if x2 is not None and y is not None:
            x1 = self.sentence_encoder(x1)     # (batch_size, hidden_size)
            x2 = self.sentence_encoder(x2)
            y = self.sentence_encoder(y)
            return self.triploss(x1, x2, y)
        elif x2 is not None:
            x1 = self.sentence_encoder(x1)  # (batch_size, hidden_size)
            x2 = self.sentence_encoder(x2)
            return self.cos(x1, x2)
        else:
            return self.sentence_encoder(x1)

    def triploss(self, x1, x2, y, margin=0.1):
        d1 = self.cos(x1, x2)
        d2 = self.cos(x1, y)
        loss = d1 - d2 + margin
        zero = torch.zeros(loss.shape)
        zero = zero.cuda()
        return torch.mean(torch.where(loss > 0, loss, zero))

    def cos(self, x1, x2):
        x1_len = torch.sqrt(torch.sum(torch.mul(x1, x1), dim=1))
        x2_len = torch.sqrt(torch.sum(torch.mul(x2, x2), dim=1))
        cos = torch.sum(torch.mul(x1, x2), dim=1) / torch.mul(x1_len, x2_len)
        return 1 - cos

    def name(self):
        return "model"


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()

        hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(config["vocab_size"] + 1, hidden_size)
        self.layer = nn.Sequential(
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=True),
            GetFirst(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        if config["model_type"] == "lstm":
            pass

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer(x)   # (batch_size, max_length, hidden_size)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        if x.ndim == 1:     # 只传入一个样本时，上面会把batch_size维度给squeeze掉，所以要补回来
            x = x.unsqueeze(0)
        return x

class GetFirst(nn.Module):
    def __init__(self):
        super(GetFirst, self).__init__()

    def forward(self, x):
        return x[0]


def choose_optimizer(config, model):
    lr = config["learning_rate"]
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)