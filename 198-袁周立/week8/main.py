
'''
尝试通过triploss完成文本匹配训练
'''

import torch
import os
import logging
import numpy as np
import json

from config import Config
from loader import load_train_data
from evaluator import Evaluator
from model import TorchModel, choose_optimizer

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_train_data(config)
    model = TorchModel(config)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(model, logger, config)

    avg_loss_watch = []
    acc_watch = []

    logger.info("当前模型：%s" % model.name())
    for epoch in range(config["epoch"]):
        model.train()
        losses = []
        logger.info("epoch%d begin" % (epoch + 1))
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [e.cuda() for e in batch_data]
            optimizer.zero_grad()
            loss = model(batch_data[0], batch_data[1], batch_data[2])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        avg_loss = np.mean(losses)
        logger.info("epoch avg loss：%.4f" % (avg_loss))
        avg_loss_watch.append(avg_loss)
        acc_watch.append(evaluator.eval(epoch))

    plt.plot(range(config["epoch"]), avg_loss_watch, label="loss")
    plt.plot(range(config["epoch"]), acc_watch, label="acc")
    plt.title(model.name())
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), config["model_path"] + "/{}.pth".format(model.name()))


def predict(test_strings, config):
    vocab = {}
    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            vocab[line] = len(vocab) + 1
    config["vocab_size"] = len(vocab)

    def encode_sentence(sentence):
        encode = [vocab.get(e, vocab["[UNK]"]) for e in sentence]
        encode = padding(encode)
        return encode

    def padding(encode):
        max_len = config["max_length"]
        encode = encode[:max_len]
        encode += (max_len - len(encode)) * [0]
        return encode

    model = TorchModel(Config)
    model.load_state_dict(torch.load("./output/model.pth"))
    model.eval()

    vectors = [encode_sentence(test_string) for test_string in test_strings]
    vectors = model(torch.LongTensor(vectors))
    questions = []
    question_target_dict = {}
    index = 0
    with open(config["train_data_path"], "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            target = line["target"]
            for question in line["questions"]:
                questions.append(encode_sentence(question))
                question_target_dict[index] = target
                index += 1
    questions = model(torch.LongTensor(questions))
    questions = torch.nn.functional.normalize(questions, dim=-1)

    pred_matrix = torch.mm(vectors, questions.T)
    pred_target = [question_target_dict[int(torch.argmax(pred_line))] for pred_line in pred_matrix]
    for index, test_string in enumerate(test_strings):
        print("{} 预测结果：{}".format(test_string, pred_target[index]))


if __name__ == "__main__":
    # if "multi_config" in Config:
    #     multi_config = Config["multi_config"]
    #     for batch_size in multi_config["batch_size"]:
    #         Config["batch_size"] = batch_size
    #         for model_type in multi_config["model_type"]:
    #             Config["model_type"] = model_type
    #             main(Config)
    # else:
    #     main(Config)
    predict(["查个花费", "加亲情号", "畅聊套餐改下", "彩信不要了", "改个密码", "挂失手机号"], Config)


