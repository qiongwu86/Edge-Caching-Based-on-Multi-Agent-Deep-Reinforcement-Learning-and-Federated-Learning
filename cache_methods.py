import numpy as np
import scipy.stats as stats
import torch

from options import args_parser
from dataset_processing import sampling
from itertools import chain

import Classes.Environment as ENV
from ddpg_torch import Agent
from user_cluster_recommend import recommend, Oracle_recommend
from Thompson_Sampling import thompson_sampling

from model import Encoder,Decoder,Discriminator
from local_update import cache_hit
from utils import exp_details, ModelManager, count_top_items


#RSU 2, cache size = 100
n_rsu = 2
n_veh = 10
l_s = 1000
cache_size = 100

agents = []
batch_size = 32
memory_size = 100000
gamma = 0.99
alpha = 0.00005
beta = 0.0005
update_actor_interval = 2
noise = 0.2
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256
A_fc1_dims = 1024
A_fc2_dims = 512
tau = 0.005

mu, sigma = 45, 4.5
lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
veh_speed = x.rvs(int(n_veh*n_rsu))
env = ENV.Environ(n_veh, veh_speed, cache_size)
env.new_random_game()
args = args_parser()
# gpu or cpu
if args.gpu: torch.cuda.set_device(args.gpu)
device = 'cuda' if args.gpu else 'cpu'

# load sample users_group_train users_group_test
# RSU1
sample1, users_group_train1, users_group_test1, _ = sampling(args, n_veh)
data_set1 = np.array(sample1)
# test_dataset & test_dataset_idx
test_dataset_idxs1 = []
for idx in range(n_veh):
    test_dataset_idxs1.append(users_group_test1[idx])
test_dataset_idxs1 = list(chain.from_iterable(test_dataset_idxs1))
test_dataset1 = data_set1[test_dataset_idxs1]
sample2, users_group_train2, users_group_test2, _ = sampling(args, n_veh)
data_set2 = np.array(sample2)
# test_dataset & test_dataset_idx
test_dataset_idxs2 = []
for idx in range(n_veh):
    test_dataset_idxs2.append(users_group_test2[idx])
test_dataset_idxs2 = list(chain.from_iterable(test_dataset_idxs2))
test_dataset2 = data_set2[test_dataset_idxs2]


n_input = cache_size * 4
n_output = cache_size * 2
# for index_agent in range(n_rsu):
#     print("Initializing agent (RSU) ", index_agent)
#     agent = Agent(alpha, beta, n_input, tau, n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
#                   A_fc1_dims, A_fc2_dims, batch_size, n_rsu, index_agent, noise)
#     agents.append(agent)
#
# for i in range(n_rsu):
#     agents[i].actor.load_checkpoint()
#
# def get_state(env, recommend_movies):
#     """ Get state from the environment """
#     # get initial action
#     cache = env.local_cache(recommend_movies)
#     return np.concatenate((np.reshape(cache, -1), np.reshape(recommend_movies, -1)), axis=0)
#
# def get_new_state(action, recommend_movies):
#     """ Get state from the environment """
#     return np.concatenate((np.reshape(action, -1), np.reshape(recommend_movies, -1)), axis=0)
#
# def actionfunction(action, n):
#     action = action.copy()
#     a = action.reshape(1,-1)
#     idx = np.argpartition(a,-n,axis=1)[:,-n:]
#     out = np.zeros(a.shape, dtype=int)
#     np.put_along_axis(out,idx,1,axis=1)
#     return out
#
#
# #federated learning
# "----------------------------------------------------------------------------------------------------"
# in_si1 = int(max(data_set1[:, 1]))
# clo1 = int(max(data_set1[1, :]))
#
# netE1 = Encoder(input_size=in_si1, hidden_size=[256, 64, 16, 4, 2])
# netE1.to(device)
# netE1.load_checkpoint()
#
# netP1 = Decoder(output_size=in_si1, hidden_size=[2, 4, 16, 64, 256])
# netP1.to(device)
# netP1.load_checkpoint()
#
# netD1 = Discriminator(input_size=2, hidden_size=[256, 16, 4, 2], output_size=1)
# netD1.to(device)
# netD1.load_checkpoint()
#
# in_si2 = int(max(data_set2[:, 1]))
# clo2 = int(max(data_set2[1, :]))
#
# netE2 = Encoder(input_size=in_si2, hidden_size=[256, 64, 16, 4, 2])
# netE2.to(device)
# netE2.load_checkpoint()
#
# netP2 = Decoder(output_size=in_si2, hidden_size=[2, 4, 16, 64, 256])
# netP2.to(device)
# netP2.load_checkpoint()
#
# netD2 = Discriminator(input_size=2, hidden_size=[256, 16, 4, 2], output_size=1)
# netD2.to(device)
# netD2.load_checkpoint()
#
# recommend_movies_cs1, netE1, netP1, netD1 = env.get_content_pop(netE1, netP1, netD1, data_set1, sample1,
#                                                                 users_group_train1, users_group_test1)
# recommend_movies_cs2, netE2, netP2, netD2 = env.get_content_pop(netE2, netP2, netD2, data_set2, sample2,
#                                                                 users_group_train2, users_group_test2)
# "----------------------------------------------------------------------------------------------------"
# state_old_all = []
# state1 = get_state(env, recommend_movies_cs1)
# state_old_all.append(state1)
# state2 = get_state(env, recommend_movies_cs2)
# state_old_all.append(state2)

n_test = 100

record_reward = np.zeros([n_rsu, n_test], dtype=np.float64)
record_hit_radio = np.zeros([n_rsu, n_test], dtype=np.float64)
record_cost = np.zeros([n_rsu, n_test], dtype=np.float64)

# step_global_reward = []
#
# for i_step in range(n_test):
#     state_new_all = []
#     action_all = []
#     action_new_all = []
#     state_old_all = []
#
#     state1 = get_state(env, recommend_movies_cs1)
#     state_old_all.append(state1)
#
#     state2 = get_state(env, recommend_movies_cs2)
#     state_old_all.append(state2)
#
#     for i in range(n_rsu):
#         a = state_old_all[i]
#         action = agents[i].choose_action(state_old_all[i])
#         action_all.append(action)
#         action_new = action.copy()
#         action_new = np.clip(action_new, 0, 0.999)
#         action_new = actionfunction(action_new, n=cache_size)
#         action_new = action_new[0]
#         action_new_all.append(action_new)
#
#     action_temp = action_new_all.copy()
#     rsu_reward, global_reward, hit_radio, cost = env.act_for_training(action_temp, state_old_all, test_dataset1, test_dataset2)
#
#     for i in range(n_rsu):
#         record_reward[i, i_step] = rsu_reward[i]
#         record_hit_radio[i, i_step] = hit_radio[i]
#         record_cost[i, i_step] = cost[i]
#     step_global_reward.append(global_reward)
#
#     state_new1 = get_new_state(action_temp[0], recommend_movies_cs1)
#     state_new_all.append(state_new1)
#
#     state_new2 = get_new_state(action_temp[1], recommend_movies_cs2)
#     state_new_all.append(state_new2)
#
#     for i in range(n_rsu):
#         state_old_all[i] = state_new_all[i]
#
# print(record_reward, record_hit_radio, record_cost, step_global_reward)

l_cost = 1
n_cost = 30
c_cost = 100
r_cost = 100

Oracle_cache1 = []
Oracle_cache2 = []
Oracle_cost1 = []
Oracle_cost2 = []
Oracle_hit_radio1 = []
Oracle_hit_radio2 = []

random_cache1 = []
random_cache2 = []
random_cost1 = []
random_cost2 = []
random_hit_radio1 = []
random_hit_radio2 = []

TS_cache1 = []
TS_cache2 = []
TS_cost1 = []
TS_cost2 = []
TS_hit_radio1 = []
TS_hit_radio2 = []

greedy_cost1 = []
greedy_cost2 = []
greedy_hit_radio1 = []
greedy_hit_radio2 = []

for i_step in range(n_test):
    #Oracle
    Oracle_recommend_movies1 = []
    Oracle_recommend_movies2 = []
    for i in range(args.clients_num):
        test_dataset_i1 = data_set1[users_group_test1[i]]
        Oracle_recommend_movies1.append(list(Oracle_recommend(test_dataset_i1, cache_size)))
        Oracle_recommend_movies2.append(list(Oracle_recommend(test_dataset_i1, cache_size)))
    Oracle_recommend_movies1 = count_top_items(cache_size, Oracle_recommend_movies1)
    Oracle_cache1.append(Oracle_recommend_movies1)
    Oracle_recommend_movies2 = count_top_items(cache_size, Oracle_recommend_movies2)
    Oracle_cache2.append(Oracle_recommend_movies2)
    # RSU1内容传递
    request_number1_oracle, hit_number1_oracle, hit_number_n1_oracle = cache_hit(test_dataset1, Oracle_recommend_movies1, Oracle_recommend_movies2)
    # RSU2内容传递
    request_number2_oracle, hit_number2_oracle, hit_number_n2_oracle = cache_hit(test_dataset2, Oracle_recommend_movies2, Oracle_recommend_movies1)

    if i_step == 0:
        replace_num_oracle1 = cache_size
        replace_num_oracle2 = cache_size
    else:
        replace_num_oracle1 = 0
        for i in range(len(Oracle_cache1[i_step])):
            for j in range(len(Oracle_cache1[i_step-1])):
                if Oracle_cache1[i_step][i] == Oracle_cache1[i_step-1][j]:
                    replace_num_oracle1 += 1
        replace_num_oracle1 = cache_size - replace_num_oracle1

        replace_num_oracle2 = 0
        for i in range(len(Oracle_cache2[i_step])):
            for j in range(len(Oracle_cache2[i_step-1])):
                if Oracle_cache2[i_step][i] == Oracle_cache2[i_step-1][j]:
                    replace_num_oracle2 += 1
        replace_num_oracle2 = cache_size - replace_num_oracle2
    Oracle_per_user_hit_radio1 = (hit_number1_oracle + hit_number_n1_oracle) / request_number1_oracle * 100
    Oracle_hit_radio1.append(Oracle_per_user_hit_radio1)
    Oracle_per_user_cost1 = l_cost * hit_number1_oracle + n_cost * hit_number_n1_oracle + c_cost * (
                request_number1_oracle - hit_number1_oracle - hit_number_n1_oracle) + r_cost * replace_num_oracle1
    Oracle_cost1.append(Oracle_per_user_cost1)
    Oracle_per_user_hit_radio2 = (hit_number2_oracle + hit_number_n2_oracle) / request_number2_oracle * 100
    Oracle_hit_radio2.append(Oracle_per_user_hit_radio2)
    Oracle_per_user_cost2 = l_cost * hit_number2_oracle + n_cost * hit_number_n2_oracle + c_cost * (
                request_number2_oracle - hit_number2_oracle - hit_number_n2_oracle) + r_cost * replace_num_oracle2
    Oracle_cost2.append(Oracle_per_user_cost2)

    # random caching
    random_caching_movies1 = list(np.random.choice(range(1, max(sample1['movie_id']) + 1), cache_size, replace=False))
    random_cache1.append(random_caching_movies1)
    random_caching_movies2 = list(np.random.choice(range(1, max(sample2['movie_id']) + 1), cache_size, replace=False))
    random_cache2.append(random_caching_movies2)
    request_number1_random, hit_number1_random, hit_number_n1_random = cache_hit(test_dataset1,
                                                                                 random_caching_movies1,
                                                                                 random_caching_movies2)
    request_number2_random, hit_number2_random, hit_number_n2_random = cache_hit(test_dataset2,
                                                                                 random_caching_movies2,
                                                                                 random_caching_movies1)
    if i_step == 0:
        replace_num_random1 = cache_size
        replace_num_random2 = cache_size
    else:
        replace_num_random1 = 0
        for i in range(len(random_cache1[i_step])):
            for j in range(len(random_cache1[i_step-1])):
                if random_cache1[i_step][i] == random_cache1[i_step-1][j]:
                    replace_num_random1 += 1
        replace_num_random1 = cache_size - replace_num_random1

        replace_num_random2 = 0
        for i in range(len(random_cache2[i_step])):
            for j in range(len(random_cache2[i_step-1])):
                if random_cache2[i_step][i] == random_cache2[i_step-1][j]:
                    replace_num_random2 += 1
        replace_num_random2 = cache_size - replace_num_random2
    random_per_user_hit_radio1 = (hit_number1_random+ hit_number_n1_random) / request_number1_random * 100
    random_hit_radio1.append(random_per_user_hit_radio1)
    random_per_user_cost1 = l_cost * hit_number1_random + n_cost * hit_number_n1_random + c_cost * (
            request_number1_random - hit_number1_random - hit_number_n1_random) + r_cost * replace_num_random1
    random_cost1.append(random_per_user_cost1)
    random_per_user_hit_radio2 = (hit_number2_random + hit_number_n2_random) / request_number2_random * 100
    random_hit_radio2.append(random_per_user_hit_radio2)
    random_per_user_cost2 = l_cost * hit_number2_random + n_cost * hit_number_n2_random + c_cost * (
            request_number2_random - hit_number2_random - hit_number_n2_random) + r_cost * replace_num_random2
    random_cost2.append(random_per_user_cost2)

    # Thompson Sampling caching
    TS_recommend_movies1 = thompson_sampling(args, data_set1, test_dataset1, cache_size)
    TS_cache1.append(TS_recommend_movies1)
    TS_recommend_movies2 = thompson_sampling(args, data_set2, test_dataset2, cache_size)
    TS_cache2.append(TS_recommend_movies2)

    request_number1_TS, hit_number1_TS, hit_number_n1_TS = cache_hit(test_dataset1,
                                                                                 TS_recommend_movies1,
                                                                                 TS_recommend_movies2)
    request_number2_TS, hit_number2_TS, hit_number_n2_TS = cache_hit(test_dataset2,
                                                                                 TS_recommend_movies2,
                                                                                 TS_recommend_movies1)
    if i_step == 0:
        replace_num_TS1 = cache_size
        replace_num_TS2 = cache_size
    else:
        replace_num_TS1 = 0
        for i in range(len(TS_cache1[i_step])):
            for j in range(len(TS_cache1[i_step-1])):
                if TS_cache1[i_step][i] == TS_cache1[i_step-1][j]:
                    replace_num_TS1 += 1
        replace_num_TS1 = cache_size - replace_num_TS1

        replace_num_TS2 = 0
        for i in range(len(TS_cache2[i_step])):
            for j in range(len(TS_cache2[i_step-1])):
                if TS_cache2[i_step][i] == TS_cache2[i_step-1][j]:
                    replace_num_TS2 += 1
        replace_num_TS2 = cache_size - replace_num_TS2

    TS_per_user_hit_radio1 = (hit_number1_TS + hit_number_n1_TS) / request_number1_TS * 100
    TS_hit_radio1.append(TS_per_user_hit_radio1)
    TS_per_user_cost1 = l_cost * hit_number1_TS + n_cost * hit_number_n1_TS + c_cost * (
            request_number1_TS - hit_number1_TS - hit_number_n1_TS) + r_cost * replace_num_TS1
    TS_cost1.append(TS_per_user_cost1)
    TS_per_user_hit_radio2 = (hit_number2_TS + hit_number_n2_TS) / request_number2_TS * 100
    TS_hit_radio2.append(TS_per_user_hit_radio2)
    TS_per_user_cost2 = l_cost * hit_number2_TS + n_cost * hit_number_n2_TS + c_cost * (
            request_number2_TS - hit_number2_TS - hit_number_n2_TS) + r_cost * replace_num_TS2
    TS_cost2.append(TS_per_user_cost2)


    # m-e-greedy caching
    e = 0.1
    greedy_per_user_hit_radio1 = Oracle_per_user_hit_radio1 * (1 - e) + random_per_user_hit_radio1 * e
    greedy_hit_radio1.append(greedy_per_user_hit_radio1)
    greedy_per_user_hit_radio2 = Oracle_per_user_hit_radio2 * (1 - e) + random_per_user_hit_radio2 * e
    greedy_hit_radio2.append(greedy_per_user_hit_radio2)

    greedy_per_user_cost1 = Oracle_per_user_cost1 * (1 - e) + random_per_user_cost1 * e
    greedy_cost1.append(greedy_per_user_cost1)
    greedy_per_user_cost2 = Oracle_per_user_cost2 * (1 - e) + random_per_user_cost2 * e
    greedy_cost2.append(greedy_per_user_cost2)

print('over')














