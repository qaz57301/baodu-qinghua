import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            sumcosine = self.__sumcosine(result)
            return result, self.points, sum,sumcosine
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __sumcosine(self,result):
        #计算总距离和(基于余弦距离)
        sum_cosine=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum_cosine+=self.__cosine(result[i][j],self.points[i])
        return sum_cosine


    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    #类内余弦相似度
    def __cosine(self, p1, p2):
        #计算两点间距
        p1_len = 0
        p2_len = 0
        # p1 和p2的长度
        for i in range(len(p1)):
            p1_len += pow(p1[i], 2)
            p2_len += pow(p2[i], 2)
        p1_len = pow(p1_len, 0.5)
        p2_len = pow(p2_len, 0.5)
        p1 = np.array(p1)
        p2 = np.array(p2)
        cosine = (p1.dot(p2.T))/(p1_len*p2_len)
        return cosine

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances, distancecosine = kmeans.cluster()
print("result", result)
print("centers", centers)
print("distances", distances)
print("distancecosine", distancecosine)
