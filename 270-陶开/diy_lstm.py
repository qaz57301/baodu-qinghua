import torch
import torch.nn as nn
import numpy as np

'''
用矩阵运算的方式复现一些基础的模型结构
清楚模型的计算细节，有助于加深对于模型的理解，以及模型转换等工作
'''
#构造一个输入
length = 6
input_dim = 12
hidden_size = 7
x = np.random.random((length, input_dim))

# 使用pytorch的lstm层

# batch_first=True ： 输入和输出张量按照（batch，seq，feature）的顺序提供
torch_lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_size,batch_first=True)
state_dict = torch_lstm.state_dict()
print(state_dict)

# print(state_dict.items())
for key,weight in state_dict.items():
    print(key)
    print(weight.shape)
    print(type(weight))

# x: 6行12列的矩阵
x = np.random.random((length, input_dim))
print(x)