import torch
import os
import logging
import numpy as np

from config import Config
from loader import load_data
from evaluator import Evaluator
from model import TorchModel, choose_optimizer

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config)
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
            loss = model(batch_data[0], batch_data[1])
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


if __name__ == "__main__":
    if Config["multi_config"]:
        multi_config = Config["multi_config"]
        for batch_size in multi_config["batch_size"]:
            Config["batch_size"] = batch_size
            for model_type in multi_config["model_type"]:
                Config["model_type"] = model_type
                main(Config)
    else:
        main(Config)