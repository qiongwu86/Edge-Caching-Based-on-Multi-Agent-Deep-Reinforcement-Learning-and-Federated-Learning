# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/4/2

import numpy as np
import random
from utils import count_top_items
from options import args_parser
from dataset_processing import sampling
from local_update import cache_hit_ratio
from itertools import chain
from collections import Counter
import matplotlib.pyplot as plt


def thompson_sampling(args, data_set, test_dataset, cachesize):
    """
    TS算法
    对于不同的缓存大小，执行TS时，得到的beta分布的参数不同，所以对于每个缓存大小都要跑一边TS
    :param args:
    :param data_set: 数据集
    :param test_dataset: 测试集，用来测试缓存命中，以更新beta参数
    :param cachesize: 缓存大小
    :return:
    """
    # 初始化beta分布系数, [1,1]*电影数目*客户数目
    movie_in_dataset = np.unique(data_set[:, 1].astype(np.uint16))
    beta_hits_losses = np.array([[1, 1]] * max(movie_in_dataset))
    # 初始化推荐电影列表
    recommend_movies = []

    for epoch in range(args.epochs):
        # for epoch in [0]:
        # 初始化推荐电影
        recommend_movies = []
        # 训练回合
        for idx in range(args.clients_num):
            prob_movies = []
            # 每一个电影的概率，将其合并为[movie_id, prob] append到prob_movies
            for movie_id in movie_in_dataset:
                prob_movies.append([movie_id, random.betavariate(beta_hits_losses[movie_id - 1, 0],
                                                                 beta_hits_losses[movie_id - 1, 1])])
            # 选取概率最大的作为推荐列表
            prob_movies.sort(key=lambda x: x[1], reverse=True)
            recommend_movie_i = [prob_movies[i][0] for i in range(cachesize)]
            recommend_movies.append(recommend_movie_i)

        # 汇总所有用户，得到cachesize的最终的推荐列表
        recommend_movies = count_top_items(cachesize, recommend_movies)
        # 更新beta参数
        # 得到test_dataset中请求的电影
        movies_request = test_dataset[:, 1]
        count = Counter(movies_request)
        # 对于所有的电影，根据命中次数来调整hits值和losses值
        # 将所有电影的命中次数归一化再乘以5
        for movie_id in recommend_movies:

            if movie_id in count.keys():
                beta_hits_losses[movie_id - 1, 0] += round(count[movie_id]/max(count.values())*5)
                # beta_hits_losses[movie_id - 1, 1] +=
            else:
                # 对于未命中的电影losses加一
                beta_hits_losses[movie_id - 1, 1] += 1
    return recommend_movies


if __name__ == '__main__':
    args = args_parser()
    # 调用dataset_processing里的函数sampling得到sample, users_group_train, users_group_test
    sample, users_group_train, users_group_test = sampling(args)
    data_set = np.array(sample)
    test_dataset_idxs = []
    for idx in range(args.clients_num):
        test_dataset_idxs.append(users_group_test[idx])
    test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    test_dataset = data_set[test_dataset_idxs]

    # cachesize = 400
    # # Thompson Sampling
    # TS_recommend_movies, beta_hits_losses = thompson_sampling(args, data_set, test_dataset, cachesize)
    # TS_cache_efficiency = cache_hit_ratio(test_dataset, TS_recommend_movies)

    cachesize = args.cachesize
    TS_recommend_movies = dict([(k, []) for k in cachesize])
    TS_cache_efficiency = np.zeros(len(cachesize))
    for i in range(len(cachesize)):
        c = cachesize[i]
        # Thompson Sampling
        TS_recommend_movies[c] = thompson_sampling(args, data_set, test_dataset, c)
        TS_cache_efficiency[i] = cache_hit_ratio(test_dataset, TS_recommend_movies[c])
    plt.figure(figsize=(6, 6))
    # 设置坐标轴范围、名称
    plt.xlim(50 - 5, 400 + 5)
    plt.ylim(0, 90)
    plt.xlabel('Cache Size')
    plt.ylabel('Cache Efficiency')
    plt.title('Cache Size vs Cache Efficiency')
    # Thompson Sampling
    plt.plot(cachesize, TS_cache_efficiency, color='purple', linewidth=1.5, linestyle='-', label='Thompson Sampling')
    plt.scatter(cachesize, TS_cache_efficiency, s=50, marker='x', color='purple')
    plt.legend()
    plt.show()
