from loader import load_data
import torch


class Evaluator:
    def __init__(self, model, logger, config):
        self.model = model
        self.logger = logger
        self.config = config
        self.valid_data = load_data(config, shuffle=False)
        self.stat_dict = {"correct": 0, "wrong": 0}

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stat_dict = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [e.cuda() for e in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)
            self.write_stats(labels, pred_results)
        acc = self.show_status()
        return acc

    def write_stats(self, true_labels, pred_results):
        assert len(true_labels) == len(pred_results)
        for true_label, pred_result in zip(true_labels, pred_results):
            pred_result = torch.argmax(pred_result).item()
            if true_label == pred_result:
                self.stat_dict["correct"] += 1
            else:
                self.stat_dict["wrong"] += 1
        return

    def show_status(self):
        correct = self.stat_dict["correct"]
        wrong = self.stat_dict["wrong"]
        self.logger.info("一共预测%d条数据" % (correct + wrong))
        self.logger.info("预测正确数量：%d，预测错误数量：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
