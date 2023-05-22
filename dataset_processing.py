import numpy as np
import pandas as pd
import copy
import torch
from data_set import DataSet
from user_info import UserInfo
from options import args_parser
from data_set import convert
import utils


def get_dataset(args):
    """
    :param: args:
    :return: ratings: dataFrame ['user_id' 'movie_id' 'rating']
    :return: user_info:  dataFrame ['user_id' 'gender' 'age' 'occupation']
    """
    model_manager = utils.ModelManager('data_set')
    user_manager = utils.UserInfoManager(args.dataset)

    '''Do you want to clean workspace and retrain model/data_set user again?'''
    '''if you want to retrain model/data_set user, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_dataset)
    user_manager.clean_workspace(args.clean_user)

    # 导入模型信息
    try:
        ratings = model_manager.load_model(args.dataset + '-ratings')
        print("Load " + args.dataset + " data_set success.\n")
    except OSError:
        ratings = DataSet.LoadDataSet(name=args.dataset)
        model_manager.save_model(ratings, args.dataset + '-ratings')

    # 导入用户信息
    try:
        user_info = user_manager.load_user_info('user_info')
        print("Load " + args.dataset + " user_info success.\n")
    except OSError:
        user_info = UserInfo.load_user_info(name=args.dataset)
        user_manager.save_user_info(user_info, 'user_info')

    return ratings, user_info


def sampling(args,clients_num):
    """
    :param args
    :return: sample: matrix user_id|movie_id|rating|gender|age|occupation|label
    :return: user_group_train, the idx of sample for each client for training
    :return: user_group_test, the idx of sample for each client for testing
    """
    # 存储每个client信息
    model_manager = utils.ModelManager('clients')
    '''Do you want to clean workspace and retrain model/clients again?'''
    '''if you want to change test_size or retrain model/clients, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_clients)
    # 导入模型信息
    try:
        users_group_train = model_manager.load_model(args.dataset + '-user_group_train')
        users_group_test = model_manager.load_model(args.dataset + '-user_group_test')
        sample = model_manager.load_model(args.dataset + '-sample')
        print("Load " + args.dataset + " clients info success.\n")
    except OSError:
        # 调用get_dataset函数，得到ratings,user_info
        ratings, user_info = get_dataset(args)
        # 每个client包含的用户数
        users_num_client = int((user_info.index[-1] + 1) / clients_num)
        # sample user_id|movie_id|rating|gender|age|occupation
        sample = pd.merge(ratings, user_info, on=['user_id'], how='inner')
        sample = sample.astype({'user_id': 'int64', 'movie_id': 'int64', 'rating': 'float64',
                                'gender': 'float64', 'age': 'float64', 'occupation': 'float64'})
        # 生成每个客户用来train和test的idx
        users_group_all, users_group_train, users_group_test, request_content = {}, {}, {}, {}
        # 生成用户数据集
        # 对用户数据集进行划分train/test
        all_test_num = 0
        all_train_num = 0
        all_num = 0
        for i in range(clients_num):
            print('loading client ' + str(i))
            index_begin = ratings[ratings['user_id'] == int(users_num_client) * i + 1].index[0]
            index_end = ratings[ratings['user_id'] == users_num_client * (i + 1)].index[-1] \
                if i != args.clients_num-1 else ratings.index[-1]
            users_group_all[i] = set(np.arange(index_begin, index_end + 1))
            NUM_train = int(0.98 * len(users_group_all[i]))
            users_group_train[i] = set(np.random.choice(list(users_group_all[i]), NUM_train, replace=False))
            users_group_test[i] = users_group_all[i] - users_group_train[i]
            # 将set转换回list，并排序
            users_group_train[i] = list(users_group_train[i])
            users_group_test[i] = list(users_group_test[i])
            users_group_train[i].sort()
            users_group_test[i].sort()
            all_test_num += NUM_train/0.98*0.2
            all_train_num += NUM_train
            all_num += int(len(users_group_all[i]))
            print('generate client ' + str(i) + ' info success\n')
        # 存储user_group_train user_group_test sample

        # for i in range(500):
        #     for j in range(clients_num):
        #         if j == 0:
        #             request_content[i] = np.random.choice(list(users_group_all[j]), int(all_num/clients_num/500), replace=True)
        #         else:
        #             request_content[i] = np.append(request_content[i], np.random.choice(list(users_group_all[j]),
        #                                                         int(all_num/clients_num/500), replace=True))
        #
        #     request_content[i] = list(request_content[i])
        #     request_content[i].sort()

        model_manager.save_model(sample, args.dataset + '-sample')
        model_manager.save_model(users_group_train, args.dataset + '-user_group_train')
        model_manager.save_model(users_group_test, args.dataset + '-user_group_test')

    return sample, users_group_train, users_group_test, request_content

def sampling_vehicle_density(args,clients_num):
    """
    :param args
    :return: sample: matrix user_id|movie_id|rating|gender|age|occupation|label
    :return: user_group_train, the idx of sample for each client for training
    :return: user_group_test, the idx of sample for each client for testing
    """
    # 存储每个client信息
    model_manager = utils.ModelManager('clients')
    '''Do you want to clean workspace and retrain model/clients again?'''
    '''if you want to change test_size or retrain model/clients, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_clients)
    # 导入模型信息
    try:
        users_group_train = model_manager.load_model(args.dataset + '-user_group_train')
        users_group_test = model_manager.load_model(args.dataset + '-user_group_test')
        sample = model_manager.load_model(args.dataset + '-sample')
        print("Load " + args.dataset + " clients info success.\n")
    except OSError:
        # 调用get_dataset函数，得到ratings,user_info
        ratings, user_info = get_dataset(args)
        # 每个client包含的用户数
        users_num_client = int((user_info.index[-1] + 1) / clients_num)
        # sample user_id|movie_id|rating|gender|age|occupation
        sample = pd.merge(ratings, user_info, on=['user_id'], how='inner')
        sample = sample.astype({'user_id': 'int64', 'movie_id': 'int64', 'rating': 'float64',
                                'gender': 'float64', 'age': 'float64', 'occupation': 'float64'})
        # 生成每个客户用来train和test的idx
        users_group_all, users_group_train, users_group_test, request_content = {}, {}, {}, {}
        # 生成用户数据集
        # 对用户数据集进行划分train/test
        all_test_num = 0
        all_train_num = 0
        all_num = 0
        for i in range(clients_num):
            print('loading client ' + str(i))
            index_begin = ratings[ratings['user_id'] == int(users_num_client) * i + 1].index[0]
            index_end = ratings[ratings['user_id'] == users_num_client * (i + 1)].index[-1] \
                if i != args.clients_num-1 else ratings.index[-1]
            users_group_all[i] = set(np.arange(index_begin, index_end + 1))
            NUM_train = int(0.96 * len(users_group_all[i]))
            users_group_train[i] = set(np.random.choice(list(users_group_all[i]), NUM_train, replace=False))
            users_group_test[i] = users_group_all[i] - users_group_train[i]
            # 将set转换回list，并排序
            users_group_train[i] = list(users_group_train[i])
            users_group_test[i] = list(users_group_test[i])
            users_group_train[i].sort()
            users_group_test[i].sort()
            all_test_num += NUM_train/0.98*0.2
            all_train_num += NUM_train
            all_num += int(len(users_group_all[i]))
            print('generate client ' + str(i) + ' info success\n')
        # 存储user_group_train user_group_test sample

        # for i in range(500):
        #     for j in range(clients_num):
        #         if j == 0:
        #             request_content[i] = np.random.choice(list(users_group_all[j]), int(all_num/clients_num/500), replace=True)
        #         else:
        #             request_content[i] = np.append(request_content[i], np.random.choice(list(users_group_all[j]),
        #                                                         int(all_num/clients_num/500), replace=True))
        #
        #     request_content[i] = list(request_content[i])
        #     request_content[i].sort()

        model_manager.save_model(sample, args.dataset + '-sample')
        model_manager.save_model(users_group_train, args.dataset + '-user_group_train')
        model_manager.save_model(users_group_test, args.dataset + '-user_group_test')

    return sample, users_group_train, users_group_test, request_content


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


if __name__ == '__main__':
    args = args_parser()
    ratings, user_info = get_dataset(args)
    sample, users_group_train, users_group_test = sampling(args)
    # 验证convert
    client_6 = np.array(sample.iloc[users_group_test[6], :])
    user_movie_6 = convert(client_6, max(sample['movie_id']))
