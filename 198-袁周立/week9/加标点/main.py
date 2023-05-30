
import logging
import os
import numpy as np

import torch
from config import Config
from loader import DataContext
from model import Model, choose_optimizer
from evaluator import Evaluator

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
尝试用bert完成ner或文本加标点任务
'''
def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    dataContext = DataContext(config)

    train_data = dataContext.get_train_data()
    model = Model(config)
    if os.path.exists(config["model_path"] + "/model.pth"):
        # 训了epoch轮后，发现loss还能降，继续每次再训几轮看看效果
        model.load_state_dict(torch.load(config["model_path"] + "/model.pth"))
        config["epoch"] = 6

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(model, logger, config, dataContext)

    avg_loss_watch = []
    acc_watch = []
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
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), config["model_path"] + "/model.pth")


if __name__ == '__main__':
    main(Config)