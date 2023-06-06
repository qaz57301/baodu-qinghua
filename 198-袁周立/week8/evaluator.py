import json
import torch
from loader import load_valid_data


class Evaluator:
    def __init__(self, model, logger, config):
        self.model = model
        self.logger = logger
        self.config = config
        self.schema = json.loads(open(config["schema_path"], "r", encoding="utf-8").read())
        self.schema_reverse = dict(zip(self.schema.values(), self.schema.keys()))
        self.valid_data, self.vocab, self.questions_all, self.question_target_dict = load_valid_data(config)
        self.questions_all = self.questions_all.cuda()
        self.questions_all = self.model(self.questions_all)
        self.questions_all = torch.nn.functional.normalize(self.questions_all, dim=-1)

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % (epoch + 1))
        self.model.eval()
        self.stat_dict = {"correct": 0, "wrong": 0}
        for data, target in self.valid_data:
            if torch.cuda.is_available():
                data = data.cuda()
            data = self.model(data)
            scores = torch.mm(data, self.questions_all.T)
            for index, score in enumerate(scores):
                max_index = int(torch.argmax(score))
                pred_target = self.question_target_dict[max_index]
                if int(pred_target) == target[index]:
                    self.stat_dict["correct"] += 1
                else:
                    self.stat_dict["wrong"] += 1
        acc = self.show_status()
        return acc

    def show_status(self):
        correct = self.stat_dict["correct"]
        wrong = self.stat_dict["wrong"]
        self.logger.info("一共预测%d条数据" % (correct + wrong))
        self.logger.info("预测正确数量：%d，预测错误数量：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)


