#coding: utf8
import torch
import torch.nn as nn
import numpy as np

"""
手动实现简单的神经网络
使用pytorch实现RNN
手动实现RNN
对比
"""

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        # batch_first: batch作为第一个参数，
        self.layer = nn.RNN(input_size=input_size,hidden_size=hidden_size,bias=False,batch_first=True)

    def forward(self,x):
        return self.layer(x)

#自定义RNN
class DiyModel:
    #初始化权重， w_ih: 输入层权重， w_hh 隐藏层权重， hidden_size: 隐藏层数
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        # ht: 1*hidden_size 的数组
        ht = np.zeros((self.hidden_size))
        # print("隐藏层shape:", ht.shape)
        # print(ht)
        output=[]
        #xt: 矩阵x中的每一行
        for xt in x:
            # ux = w_ih*xt.T
            ux = np.dot(self.w_ih, xt)
            #wh = w_hh*ht.T
            wh = np.dot(self.w_hh, ht)
            ht_next = np.tanh(ux+wh)
            output.append(ht_next)
            ht = ht_next
        return np.array(output), ht

hidden_size = 4
x = np.array([[1, 2, 3],
              [3, 4, 5],
              [5, 6, 7]])  #网络输入

torch_model = TorchRNN(3, hidden_size)
torch_x = torch.FloatTensor(x)
rnn_result = torch_model.forward(torch_x)
print("rnn_result", rnn_result)

# 输入层权重矩阵
w_ih = torch_model.state_dict()["layer.weight_ih_l0"]
# 隐藏层权重矩阵
w_hh = torch_model.state_dict()["layer.weight_hh_l0"]
print(w_ih, w_ih.shape)
print(w_hh, w_hh.shape)

diymodel = DiyModel(w_ih, w_hh, hidden_size)
print(diymodel)

output, ht = diymodel.forward(x)
print("output:", output, "ht:", ht)

ht= np.zeros(hidden_size)
for xt in x:
    ux = np.dot(w_ih, xt.T)
    print("ux", ux)
    wh = np.dot(w_hh, ht)
    print("wh", wh)
    ht_next = np.tanh()




























