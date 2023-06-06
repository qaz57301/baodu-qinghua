import numpy as np

import torch.nn as nn
import torch
import matplotlib.pyplot as plt


'''

字符串中数字1、2、3出现的前后顺序是2xx  类别1
字符串中数字1、2、3出现的前后顺序是3xx  类别2
字符串中数字1、2、3出现的前后顺序是1xx  类别3
字符串不包含1、2、3                 类别4

效果不稳定，不理想，跑一次模型有时候准确率0.8、0.9左右，有时候0.4左右，
应该是生成的数据样本不均匀导致的，随机数不是真正随机，规则弄复杂了
'''

class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size, 4)
        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.activation(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return torch.argmax(x, dim=1)


def build_sample(batch_size, vector_dim):
    x = []
    y = []
    while len(x) < batch_size:
        sample_x = np.random.randint(1, 5, vector_dim)
        sample_y = 3
        digits = set()
        for e in sample_x:
            if e in [1, 2, 3]:
                digits.add(str(e))
        digits = "".join(digits)
        if digits == "213" or digits == "231":
            sample_y = 0
        if digits == "312" or digits == "321":
            sample_y = 1
        if digits == "123" or digits == "132":
            sample_y = 2
        x.append(sample_x)
        y.append(sample_y)
    return torch.from_numpy(np.array(x, dtype="float32")), torch.LongTensor(y)


def predict(model, vector_dim):
    correct = 0
    wrong = 0
    x, y = build_sample(100, vector_dim)
    y_pred = model(x)
    for i in range(len(y)):
        if int(y[i]) == int(y_pred[i]):
            correct += 1
        else:
            wrong += 1
    print("预测次数：%d，正确率：%f" % (correct + wrong, correct / (correct + wrong)))
    return correct / (correct + wrong)


if __name__ == "__main__":
    epoch_num = 30
    batch_size = 20
    sample_size = 1000
    vector_dim = 7
    lr = 0.001

    model = MyModel(vector_dim)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    acc_loss = []
    for epoch in range(epoch_num):
        loss_watch = []
        acc_watch = []
        print("epoch%d：" % (epoch + 1))
        for i in range(int(sample_size / batch_size)):
            x, y = build_sample(batch_size, vector_dim)

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()

            loss_watch.append(loss)
        avg_loss = sum(loss_watch) / len(loss_watch)
        print("average loss：%f" % (avg_loss))
        acc_loss.append([float(avg_loss), predict(model, vector_dim)])

    plt.plot([i for i in range(len(acc_loss))], [e[0] for e in acc_loss], label="loss")
    plt.plot([i for i in range(len(acc_loss))], [e[1] for e in acc_loss], label="acc")
    plt.legend()
    plt.show()

    # torch.save(model.state_dict(), "week2.pth")

    # model_predict = MyModel(vector_dim)
    # model_predict.load_state_dict(torch.load("week2.pth"))
    # predict(model_predict, vector_dim)

