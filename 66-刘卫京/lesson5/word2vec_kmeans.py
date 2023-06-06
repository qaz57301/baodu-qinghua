# -- coding: utf-8 --
# @Time : 2023/5/11 17:26
# @Author : liuweijing
# @File : word2vec_kmeans_m.py
import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.lcut(sentence)))
    print("获取句子数量:", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence 是分好词的，用空格切分
        vector = np.zeros(model.vector_size)
        # 将所有词的向量相加求平均作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 欧式距离
def eculid_distance(p1, p2):
    # 计算两点间距
    distance = 0
    for i in range(len(p1)):
        distance += pow(p1[i] - p2[i], 2)
    return pow(distance, 0.5)

#向量余弦距离
def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  #A/|A|
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  #B/|B|
    return np.sum(vec1 * vec2)

# 计算所有聚类的平均距离
def calc_cluster_distance(vectors, center):
    n = len(vectors)
    sum = 0
    for v in vectors:
        sum += cosine_distance(v, center)
    return sum / n

def main():
    # 加载词向量模型
    model = load_word2vec_model("model.w2v")
    # 加载所有标题
    sentences = load_sentence("titles.txt")
    # 将所有标题向量化
    vectors = sentences_to_vectors(sentences, model)

    # 开始聚类
    n_clusters = int(math.sqrt(len(sentences)))  # 总样本的开方作为聚类数量
    print("制定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 开始排序
    center_list = kmeans.cluster_centers_.tolist()
    labels_list = kmeans.labels_
    vectors_dict = defaultdict(list)
    for i in range(len(vectors)):
        vectors_dict[labels_list[i]].append(vectors[i])
    distance_dict = dict()
    for k,v in vectors_dict.items():
        center = center_list[k]
        avg_distance = calc_cluster_distance(v, center)
        distance_dict[k] = avg_distance
    print("排序前：",distance_dict)
    # 如果使用余弦距离，距离越接近1，说明两个向量越接近，和欧式距离相反，欧式距离越大说明向量间距离远
    distance_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)
    print("排序后：", distance_dict)
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_): # 取出句子和标签
        sentence_label_dict[label].append(sentence)
    print("聚类内部平均距离最小的前10个")
    for i in range(10):
        label = distance_dict[i][0]
        print("cluster:", label)
        for i in range(min(10, len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("--------------------------------")

    print("聚类内部平均距离最大的10个")
    for i in range(len(distance_dict) - 1, len(distance_dict) - 11, -1):
        label = distance_dict[i][0]
        print("cluster:", label)
        for i in range(min(10, len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("--------------------------------")
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster:", label)
    #     for i in range(min(10, len(sentences))):
    #         print(sentences[i].replace(" ", ""))
    #     print("--------------------------------")


if __name__ == "__main__":
    main()
