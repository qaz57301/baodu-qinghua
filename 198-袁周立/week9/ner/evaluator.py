from collections import defaultdict

import torch
import re
import numpy as np


class Evaluator:
    def __init__(self, model, logger, config, dataContext):
        self.model = model
        self.logger = logger
        self.config = config
        self.valid_data = dataContext.get_valid_data()
        self.entities = ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % (epoch + 1))
        self.model.eval()
        self.stat_dict = dict([e, defaultdict(int)] for e in self.entities)
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [e.cuda() for e in batch_data]
            input_ids, true_labels = batch_data
            pred_labels = self.model(input_ids)
            self.write_status(pred_labels, true_labels)
        return self.show_status()

    def write_status(self, pred_labels, true_labels):
        assert len(pred_labels) == len(true_labels)
        for pred_label, true_label in zip(pred_labels, true_labels):
            pred_label = torch.argmax(pred_label, -1)
            pred_label, true_label = pred_label.tolist(), true_label.tolist()

            sentence_len = self.config["max_len"] if -1 not in true_label else true_label.index(-1)

            pred_label, true_label = pred_label[:sentence_len], true_label[:sentence_len]

            pred_entities = self.decode(pred_label)
            true_entities = self.decode(true_label)

            for entity in self.entities:
                self.stat_dict[entity]["正确识别数"] += len([e for e in pred_entities[entity] if e in true_entities[entity]])
                self.stat_dict[entity]["样本实际实体数"] += len(true_entities[entity])
                self.stat_dict[entity]["样本预测实体数"] += len(pred_entities[entity])
        return

    def show_status(self):
        F1_Scores = []
        for entity in self.entities:
            precision = self.stat_dict[entity]["正确识别数"] / (1e-5 + self.stat_dict[entity]["样本预测实体数"])
            recall = self.stat_dict[entity]["正确识别数"] / (1e-5 + self.stat_dict[entity]["样本实际实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_Scores.append(F1)
            self.logger.info("实体{}：F1：{}，实际实体数：{}，预测实体数：{}，正确识别数：{}，".format(entity, F1,
                                                                           self.stat_dict[entity]["样本实际实体数"],
                                                                           self.stat_dict[entity]["样本预测实体数"],
                                                                           self.stat_dict[entity]["正确识别数"],
                                                                           ))
        self.logger.info("Macro-F1: %f" % np.mean(F1_Scores))
        correct_pred_num = sum([self.stat_dict[entity]["正确识别数"] for entity in self.entities])
        pred_entity_num = sum([self.stat_dict[entity]["样本预测实体数"] for entity in self.entities])
        true_entity_num = sum([self.stat_dict[entity]["样本实际实体数"] for entity in self.entities])
        micro_precision = correct_pred_num / (pred_entity_num + 1e-5)
        micro_recall = correct_pred_num / (true_entity_num + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return micro_precision

    def decode(self, labels):
        labels = "".join([str(e) for e in labels])
        result = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            start, end = location.span()
            result["LOCATION"].append(str(start) + "-" + str(end))
        for organization in re.finditer("(15+)", labels):
            start, end = organization.span()
            result["ORGANIZATION"].append(str(start) + "-" + str(end))
        for person in re.finditer("(26+)", labels):
            start, end = person.span()
            result["PERSON"].append(str(start) + "-" + str(end))
        for time in re.finditer("(37+)", labels):
            start, end = time.span()
            result["TIME"].append(str(start) + "-" + str(end))
        return result