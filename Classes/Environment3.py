import numpy as np
import math
import scipy.stats as stats
import random
np.random.seed(1376)
from content_prediction_elasticFL import content_prediction
from data_set import convert
from user_cluster_recommend import recommend
from utils import count_top_items
from options import args_parser
from local_update import cache_hit3, cache_hit


class Environ:
    def __init__(self, n_veh, veh_speed, cache_size):

        self.n_Veh = n_veh
        self.veh_speed = veh_speed
        self.cache = []
        self.cache_size = cache_size
        # 100ms
        self.time_slow = 0.1

    def renew_positions(self,veh_pos):
        # ===============
        # This function updates the position of each vehicle
        # ===============
        i = 0
        while (i < len(self.veh_speed)):
            delta_distance = self.veh_speed[i] * self.time_slow
            veh_pos[i] += delta_distance
            i += 1
        return veh_pos

    def add_new_vehicle(self):
        # ===============
        # This function adds new vehicle
        # ===============
        mu, sigma = 45, 4.5
        lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
        x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        veh_speed = x.rvs(1)
        print('new add vehicle speed:', veh_speed, 'm/s')
        return veh_speed[0]

    def wwh_test_get_lr_para(self, netE, netP, netD, data_set, users_group_train):
        args = args_parser()
        netE, netP, netD, w_e_all_epochs, w_p_all_epochs, w_d_all_epochs = content_prediction(self.n_Veh, netE, netP,
                                                                                              netD, data_set,
                                                                                              users_group_train)

        return w_e_all_epochs

    def wwh_test_get_content_pop(self,w_e_all_epochs,data_set, sample,users_group_test):
        args = args_parser()
        recommend_movies = []
        # recommend movies
        for idx in range(self.n_Veh):
            test_dataset_i = data_set[users_group_test[idx]]
            user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
            recommend_list = recommend(user_movie_i, test_dataset_i, w_e_all_epochs[args.epochs - 1][idx])
            recommend_list = count_top_items(self.cache_size * 2, recommend_list)
            recommend_movies.append(list(recommend_list))
        recommend_movies_cs = count_top_items(self.cache_size * 2, recommend_movies)
        return recommend_movies_cs

    def get_content_pop(self, netE, netP, netD, data_set, sample, users_group_train, users_group_test):

        args = args_parser()
        netE, netP, netD, w_e_all_epochs, w_p_all_epochs, w_d_all_epochs = content_prediction(self.n_Veh, netE, netP, netD, data_set, users_group_train)

        print('\n -----------------------------------')
        print('\n Start content popularity prediction')
        print('\n -----------------------------------')

        # Recommend movies
        # dictionary index: client idx
        recommend_movies = []
        # recommend movies
        for idx in range(self.n_Veh):
            test_dataset_i = data_set[users_group_test[idx]]
            user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
            recommend_list = recommend(user_movie_i, test_dataset_i, w_e_all_epochs[args.epochs-1][idx])
            recommend_list = count_top_items(self.cache_size*2 , recommend_list)
            recommend_movies.append(list(recommend_list))
        recommend_movies_cs = count_top_items(self.cache_size*2 , recommend_movies)
        return recommend_movies_cs, netE, netP, netD

    def local_cache(self, recommend_movie):
        # obtain initial cache
        self.cache = np.zeros(len(recommend_movie))
        recommend_movie = recommend_movie.tolist()
        l_cache = random.sample(range(len(recommend_movie)), self.cache_size)

        for i in range(len(l_cache)):
            self.cache[l_cache[i]] = 1
        return self.cache


    def Compute_Performance_Reward_Train(self, actions, state_old_all, test_dataset1, test_dataset2, test_dataset3):
        # 3 RSU

        # RSU1
        # 300 popular contents, 100 cache contents
        popular_content1 = []       # 推荐的缓存内容
        for i in range(self.cache_size*2):
            popular_content1.append(state_old_all[0][i + 2 * self.cache_size])

        old_cache1 = []
        for i in range(self.cache_size*2):
            old_cache1.append(state_old_all[0][i])

        #RSU1 更新缓存内容损耗
        replace_num1 = 0
        for i in range(len(actions[0])):
            if actions[0][i] != old_cache1[i] and actions[0][i] == 1:
                replace_num1 += 1
        replace_elements1 = replace_num1

        cache_content1=[]                       # 做出action后缓存的内容
        for i in range(len(popular_content1)):  # 说明 1 是缓存, 0 是不缓存
            if actions[0][i] == 1:
                cache_content1.append(popular_content1[i])

        # RSU2
        popular_content2 = []
        for i in range(self.cache_size * 2):
            popular_content2.append(state_old_all[1][i + 2 * self.cache_size])

        old_cache2 = []
        for i in range(self.cache_size * 2):
            old_cache2.append(state_old_all[1][i])

        # RSU2 更新缓存内容损耗
        replace_num2 = 0
        for i in range(len(actions[1])):
            if actions[1][i] != old_cache2[i] and actions[1][i] == 1:
                replace_num2 += 1
        replace_elements2 = replace_num2

        cache_content2=[]
        for i in range(len(popular_content2)):
            if actions[1][i] == 1:
                cache_content2.append(popular_content2[i])

        # RSU3
        # 300 popular contents, 100 cache contents
        popular_content3 = []  # 推荐的缓存内容
        for i in range(self.cache_size * 2):
            popular_content3.append(state_old_all[2][i + 2 * self.cache_size])

        old_cache3 = []
        for i in range(self.cache_size * 2):
            old_cache3.append(state_old_all[2][i])

        # RSU3 更新缓存内容损耗
        replace_num3 = 0
        for i in range(len(actions[2])):
            if actions[2][i] != old_cache3[i] and actions[2][i] == 1:
                replace_num3 += 1
        replace_elements3 = replace_num3

        cache_content3 = []  # 做出action后缓存的内容
        for i in range(len(popular_content3)):  # 说明 1 是缓存, 0 是不缓存
            if actions[2][i] == 1:
                cache_content3.append(popular_content3[i])

        #RSU1内容传递
        request_number1, hit_number1, hit_number_n1 = cache_hit(test_dataset1, cache_content1, cache_content2)

        # RSU2内容传递
        request_number2, hit_number2, hit_number_n2 = cache_hit3(test_dataset2, cache_content2, cache_content1, cache_content3)

        # RSU3内容传递
        request_number3, hit_number3, hit_number_n3 = cache_hit(test_dataset3, cache_content3, cache_content2)

        return replace_elements1, replace_elements2, replace_elements3, request_number1, \
               hit_number1, hit_number_n1, request_number2, hit_number2, hit_number_n2,\
               request_number3, hit_number3, hit_number_n3


    def act_for_training(self, actions, state_old_all, test_dataset1, test_dataset2, test_dataset3):

        # 不同的缓存方式所得到的reward:本地缓存，相邻缓存，中心缓存
        self.l_cost = 1
        self.n_cost = 30
        self.c_cost = 100
        self.r_cost = 100

        per_user_hit_radio = np.zeros(len(actions))
        per_user_reward = np.zeros(len(actions))
        per_user_cost = np.zeros(len(actions))

        action_temp = actions.copy()
        replace_element1, replace_element2, replace_element3, request_number1, hit_number1, hit_number_n1,\
        request_number2, hit_number2, hit_number_n2, request_number3, hit_number3, hit_number_n3\
            = self.Compute_Performance_Reward_Train(action_temp, state_old_all, test_dataset1, test_dataset2, test_dataset3)
        per_user_reward[0] = (self.c_cost - self.l_cost) * hit_number1 + (self.c_cost - self.n_cost) * hit_number_n1\
                - self.r_cost * replace_element1

        #per_user_reward[0] = - self.r_cost * replace_element1

        #per_user_reward[0] = (self.c_cost - self.l_cost) * hit_number1 + (self.c_cost - self.n_cost) * hit_number_n1
        #print('hit_number1 : ', hit_number1, 'hit_number_n1 : ', hit_number_n1, 'replace_element1 : ', replace_element1,
        #      'user1_reward : ', per_user_reward[0])

        per_user_hit_radio[0] = (hit_number1 + hit_number_n1) / request_number1 * 100
        per_user_cost[0] = self.l_cost * hit_number1 + self.n_cost * hit_number_n1 + self.c_cost * (request_number1 - hit_number1 - hit_number_n1) + \
                             self.r_cost * replace_element1
        #per_user_cost[0] = self.l_cost * hit_number1 + self.n_cost * hit_number_n1 + self.c_cost * (request_number1 - hit_number1 - hit_number_n1)


        per_user_reward[1] = (self.c_cost - self.l_cost) * hit_number2 + (self.c_cost - self.n_cost) * hit_number_n2\
                - self.r_cost * replace_element2

        per_user_reward[2] = (self.c_cost - self.l_cost) * hit_number3 + (self.c_cost - self.n_cost) * hit_number_n3\
                - self.r_cost * replace_element3

        #per_user_reward[1] = - self.r_cost * replace_element2

        #print('hit_number2 : ', hit_number2, 'hit_number_n2 : ', hit_number_n2, 'replace_element2 : ', replace_element2,
        #      'user2_reward : ', per_user_reward[1])
        #per_user_reward[1] = (self.c_cost - self.l_cost) * hit_number2 + (self.c_cost - self.n_cost) * hit_number_n2
        print('replace_element1 : ', replace_element1, 'user1_reward : ', per_user_reward[0],
              'replace_element2 : ', replace_element2, 'user2_reward : ', per_user_reward[1],
              'replace_element3 : ', replace_element3, 'user3_reward : ', per_user_reward[2],)

        per_user_hit_radio[1] = (hit_number2 + hit_number_n2) / request_number2 * 100
        per_user_cost[1] = self.l_cost * hit_number2 + self.n_cost * hit_number_n2 + self.c_cost * (request_number2 - hit_number2 - hit_number_n2) + \
                             self.r_cost * replace_element2

        per_user_hit_radio[2] = (hit_number3 + hit_number_n3) / request_number3 * 100
        per_user_cost[2] = self.l_cost * hit_number3 + self.n_cost * hit_number_n3 + self.c_cost * (
                    request_number3 - hit_number3 - hit_number_n3) + \
                           self.r_cost * replace_element3
        #per_user_cost[1] = self.l_cost * hit_number2 + self.n_cost * hit_number_n2 + self.c_cost * (request_number2 - hit_number2 - hit_number_n2)

        global_reward = np.mean(per_user_reward)

        return per_user_reward, global_reward, per_user_hit_radio, per_user_cost

    def new_random_game(self, n_Veh=0):

        # make a new game
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
