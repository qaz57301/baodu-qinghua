import csv
import numpy as np

"""
数据生成
"""

class DataGenerator:
    def __init__(self, data_path, config) -> None:
        self.config = config
        self.path = data_path
        self.generate_train_data()
    
    def generate_train_data(self):
        self.train_data = []
        self.test_data = []
        good = 0
        bad = 0
        with open(self.path, encoding="utf8") as f:
            for line in csv.reader(f):
                if line[0] == 'label':
                    continue
                label = int(line[0])
                review = line[1]
                if label:
                    good += 1
                else:
                    bad += 1
        index_good = np.random.choice(a=np.linspace(0,good-1,good),size=1000,replace=False)
        index_bad = np.random.choice(a=np.linspace(good,good+bad-1,bad),size=1000,replace=False)
        index = 0
        with open(self.path, encoding="utf8") as f:
            for line in csv.reader(f):
                if line[0] == 'label':
                    continue
                label = int(line[0])
                review = line[1]
                if index in index_good:
                    self.test_data.append([label,review])
                    index += 1
                    continue
                if index in index_bad:
                    self.test_data.append([label,review])
                    index += 1
                    continue
                self.train_data.append([label,review])
                index += 1
        with open("part_test_data.csv", 'w', encoding="utf8", newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(['label','review'])
            for i in self.test_data:
                writer.writerow(i)
        with open("part_train_data.csv", 'w', encoding="utf8", newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(['label','review'])
            for i in self.train_data:
                writer.writerow(i)
        return

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("文本分类练习.csv", Config)
