import numpy as np
import scipy.stats as stats
import torch

from options import args_parser
from dataset_processing import sampling
from itertools import chain

import Classes.Environment as ENV
from ddpg_torch import Agent
from global_critic import Global_Critic

from model import Encoder,Decoder,Discriminator

n_rsu = 2
n_veh = 10
l_s = 1000
cache_size = 50

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
for index_agent in range(n_rsu):
    print("Initializing agent (RSU) ", index_agent)
    agent = Agent(alpha, beta, n_input, tau, n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                  A_fc1_dims, A_fc2_dims, batch_size, n_rsu, index_agent, noise)
    agents.append(agent)

for i in range(n_rsu):
    agents[i].actor.load_checkpoint()

def get_state(env, recommend_movies):
    """ Get state from the environment """
    # get initial action
    cache = env.local_cache(recommend_movies)
    return np.concatenate((np.reshape(cache, -1), np.reshape(recommend_movies, -1)), axis=0)

def get_new_state(action, recommend_movies):
    """ Get state from the environment """
    return np.concatenate((np.reshape(action, -1), np.reshape(recommend_movies, -1)), axis=0)

def actionfunction(action, n):
    action = action.copy()
    a = action.reshape(1,-1)
    idx = np.argpartition(a,-n,axis=1)[:,-n:]
    out = np.zeros(a.shape, dtype=int)
    np.put_along_axis(out,idx,1,axis=1)
    return out


#federated learning
"----------------------------------------------------------------------------------------------------"
in_si1 = int(max(data_set1[:, 1]))
clo1 = int(max(data_set1[1, :]))

netE1 = Encoder(input_size=in_si1, hidden_size=[256, 64, 16, 4, 2])
netE1.to(device)
netE1.load_checkpoint()

netP1 = Decoder(output_size=in_si1, hidden_size=[2, 4, 16, 64, 256])
netP1.to(device)
netP1.load_checkpoint()

netD1 = Discriminator(input_size=2, hidden_size=[256, 16, 4, 2], output_size=1)
netD1.to(device)
netD1.load_checkpoint()

in_si2 = int(max(data_set2[:, 1]))
clo2 = int(max(data_set2[1, :]))

netE2 = Encoder(input_size=in_si2, hidden_size=[256, 64, 16, 4, 2])
netE2.to(device)
netE2.load_checkpoint()

netP2 = Decoder(output_size=in_si2, hidden_size=[2, 4, 16, 64, 256])
netP2.to(device)
netP2.load_checkpoint()

netD2 = Discriminator(input_size=2, hidden_size=[256, 16, 4, 2], output_size=1)
netD2.to(device)
netD2.load_checkpoint()

recommend_movies_cs1, netE1, netP1, netD1 = env.get_content_pop(netE1, netP1, netD1, data_set1, sample1,
                                                                users_group_train1, users_group_test1)
recommend_movies_cs2, netE2, netP2, netD2 = env.get_content_pop(netE2, netP2, netD2, data_set2, sample2,
                                                                users_group_train2, users_group_test2)
"----------------------------------------------------------------------------------------------------"
state_old_all = []
state1 = get_state(env, recommend_movies_cs1)
state_old_all.append(state1)
state2 = get_state(env, recommend_movies_cs2)
state_old_all.append(state2)

n_test = 100

record_reward = np.zeros([n_rsu, n_test], dtype=np.float64)
record_hit_radio = np.zeros([n_rsu, n_test], dtype=np.float64)
record_cost = np.zeros([n_rsu, n_test], dtype=np.float64)

step_global_reward = []

for i_step in range(n_test):
    state_new_all = []
    action_all = []
    action_new_all = []
    state_old_all = []

    for i in range(n_rsu):
        a = state_old_all[i]
        action = agents[i].choose_action(state_old_all[i])
        action_all.append(action)
        action_new = action.copy()
        action_new = np.clip(action_new, 0, 0.999)
        action_new = actionfunction(action_new, n=cache_size)
        action_new = action_new[0]
        action_new_all.append(action_new)

    action_temp = action_new_all.copy()
    rsu_reward, global_reward, hit_radio, cost = env.act_for_training(action_temp, state_old_all, test_dataset1, test_dataset2)

    for i in range(n_rsu):
        record_reward[i, i_step] = rsu_reward[i]
        record_hit_radio[i, i_step] = hit_radio[i]
        record_cost[i, i_step] = cost[i]
    step_global_reward.append(global_reward)

    state_new1 = get_new_state(action_temp[0], recommend_movies_cs1)
    state_new_all.append(state_new1)

    state_new2 = get_new_state(action_temp[1], recommend_movies_cs2)
    state_new_all.append(state_new2)

    for i in range(n_rsu):
        state_old_all[i] = state_new_all[i]

print(record_reward, record_hit_radio, record_cost, step_global_reward)











