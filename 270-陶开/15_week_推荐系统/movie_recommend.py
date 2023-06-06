"""
使用协同过滤，进行产品推荐
"""
import numpy as np


def build_u2i_matrix(user_item_score_data_path, item_name_data_path, write_file=False):
    item_id_to_item_name = {}
    with open(item_name_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            item_id, item_name = line.split("|")[:2]
            item_id_to_item_name[item_id] = item_name
    total_movie_count = len(item_id_to_item_name)
    print("total movies: ", total_movie_count)

    user_to_rating={}
    with open(user_item_score_data_path, encoding="ISO-8859-1") as f:
        for line in f:
            user_id, item_id, score = line.split("\t")[:3]
            user_id, item_id, score = int(user_id), int(item_id), int(score)

            if user_id not in user_to_rating:
                user_to_rating[user_id] = [0]*total_movie_count
            user_to_rating[user_id][item_id-1] = score
    user_to_rating = dict(sorted(user_to_rating.items(), key=lambda v:v[0]))
    return user_to_rating, item_id_to_item_name


# 向量余弦距离
def cosine_distance(vector1, vector2):
    ab = vector1.dot(vector2)
    a_norm = np.sqrt(np.sum(np.square(vector1)))
    b_norm = np.sqrt(np.sum(np.square(vector2)))
    return ab/(a_norm*b_norm)

#根据用户打分计算item相似度
def find_similar_item(user_to_rating):
    # 电影数 =682， 人数 944
    # 依照user对item的打分判断user之间的相似度
    item_to_vector = {}
    total_user = len(user_to_rating)
    for user, user_rating in user_to_rating.items():
        for movie_id, score in enumerate(user_to_rating[user]):
            movie_id+=1
            if movie_id not in item_to_vector:
                item_to_vector[movie_id] = [0]*total_user
            item_to_vector[movie_id][user] = score
        # item_to_vector记录了每个用户打分，数据结构和user_to_rating一样
        # 复用一下find_similar_user 的相似度计算方法
        return find_similar_user(item_to_vector)

# 依照user对item的打分判断user之间的相似度
def find_similar_user(user_to_rating):
    user_to_similar_user = {}
    score_buffer = {}
    for user_a, ratings_a in user_to_rating.items():
        similar_user = []
        for user_b, ratings_b in user_to_rating.items():
            if user_a == user_b or user_a>100 or user_b>100:
                continue
            # ab用户互换不用计算similarity
            if "%d_%d" %(user_b, user_a) in score_buffer:
                similarity = score_buffer["%d_%d" %(user_b, user_a)]
            # 相似度计算采取cos距离
            else:
                # print("===")
                similarity = cosine_distance(np.array(ratings_a), np.array(ratings_b))
                # print(similarity)
                score_buffer["%d_%d" %(user_a, user_b)] = similarity
                # print(score_buffer)
            similar_user.append([user_b, similarity])
        similar_user = sorted(similar_user, reverse=True, key=lambda x : x[1])
        user_to_similar_user[user_a] = similar_user
    # print(user_to_similar_user)
    return user_to_similar_user

#基于user的协同过滤
#输入user_id, item_id, 给出预测打分
#有预测打分之后就可以对该用户所有未看过的电影打分，然后给出排序结果
#所以实现打分函数即可
#topn为考虑多少相似的用户
#取前topn相似用户对该电影的打分
def user_cf(user_id, item_id, user_to_similar_user, user_to_rating, topn=10):
    # print(user_to_similar_user)
    pred_score = 0
    count = 0
    for similar_user, similarity in user_to_similar_user[user_id][:topn]:
        # 相似用户对这部电影的打分
        rating_by_similiar_user = user_to_rating[similar_user][item_id-1]
        # 分数* 用户相似度，作为一种对分数的加权，越相似的用户评分越重要
        pred_score += rating_by_similiar_user*similarity
        # 如果这个相似用户没看多， 就不计算在总数内
        if rating_by_similiar_user != 0:
            count += 1
    pred_score /= count+1e-5
    return pred_score


#基于item的协同过滤
#类似user_cf
#自己尝试实现
def item_cf(user_id, item_id, similar_items, user_to_rating, topn = 10):
    pred_score = 0
    count = 0
    # 产品之间的相似度
    for similar_item, similarity in similar_items[item_id][:topn]:
        # 相似产品的打分
        rating_by_similiar_item = similar_items[similar_item][user_id]
        # 分数*产品相似度， 作为一种对分数的加权，越相似的产品评分越重要
        # 如果这个 相似产品没有评分，就不计算在内
        if rating_by_similiar_item!=0:
            count+=1
    pred_score/=count+1e-5
    return pred_score


if __name__ == '__main__':
    user_item_score_data_path = "ml-100k/u.data"
    item_name_data_path = "ml-100k/u.item"
    user_to_rating, item_id_to_item_name = build_u2i_matrix(user_item_score_data_path, item_name_data_path, write_file=False)
    # print(user_to_rating)

    user_to_similar_user=find_similar_user(user_to_rating)
    # print(user_to_similar_user)
