# -*- coding = utf-8 -*-
"""
Utils in order to simplify coding.

Created on 2018-04-16

@author: fuxuemingzhu
"""
import time
import pickle
import os
import shutil
import numpy as np
from itertools import chain
from collections import Counter
import matplotlib.pyplot as plt


class LogTime:
    """
    Time used help.
    You can use count_time() in for-loop to count how many times have looped.
    Call finish() when your for-loop work finish.
    WARNING: Consider in multi-for-loop, call count_time() too many times will slow the speed down.
            So, use count_time() in the most outer for-loop are preferred.
    """

    def __init__(self, print_step=20000, words=''):
        """
        How many steps to print a progress log.
        :param print_step: steps to print a progress log.
        :param words: help massage
        """
        self.proccess_count = 0
        self.PRINT_STEP = print_step
        # record the calculate time has spent.
        self.start_time = time.time()
        self.words = words
        self.total_time = 0.0

    def count_time(self):
        """
        Called in for-loop.
        :return:
        """
        # log steps and times.
        if self.proccess_count % self.PRINT_STEP == 0:
            curr_time = time.time()
            print(self.words + ' steps(%d), %.2f seconds have spent..' % (
                self.proccess_count, curr_time - self.start_time))
        self.proccess_count += 1

    def finish(self):
        """
        Work finished! Congratulations!
        :return:
        """
        print('total %s step number is %d' % (self.words, self.get_curr_step()))
        print('total %.2f seconds have spent\n' % self.get_total_time())

    def get_curr_step(self):
        return self.proccess_count

    def get_total_time(self):
        return time.time() - self.start_time


class ModelManager:
    """
    Model manager is designed to load and save all models.
    No matter what data_set name.
    """
    # This file_name belongs to the whole class.
    # So it should be init for only once.
    path_name = ''

    def __init__(self, folder_name=None):
        """
        cls.file_name should only init for only once.
        :param folder_name:
        """
        self.folder_name = folder_name
        if not self.path_name:
            self.path_name = "model/" + folder_name + "/"

    def save_model(self, model, save_name: str):
        """
        Save model to model/ dir.
        :param model: source model
        :param save_name: model saved name.
        :return: None
        """
        if 'pkl' not in save_name:
            save_name += '.pkl'
        if not os.path.exists("model/"):
            os.mkdir("model/")
        if not os.path.exists(self.path_name):
            os.mkdir(self.path_name)
        if os.path.exists(self.path_name + "%s" % save_name):
            os.remove(self.path_name + "%s" % save_name)
        pickle.dump(model, open(self.path_name + "%s" % save_name, "wb"))

    def load_model(self, model_name: str):
        """
        Load model from model/ dir via model name.
        :param model_name:
        :return: loaded model
        """
        if 'pkl' not in model_name:
            model_name += '.pkl'
        if not os.path.exists(self.path_name + "%s" % model_name):
            raise OSError('There is no model named %s in model/ dir' % model_name)
        return pickle.load(open(self.path_name + "%s" % model_name, "rb"))

    def clean_workspace(self, clean=False):
        """
        Clean the whole workspace.
        All File in model/ dir will be removed.
        :param clean: Boolean. Clean workspace or not.
        :return: None
        """
        if clean and os.path.exists(self.path_name):
            shutil.rmtree(self.path_name)

    def delete_file(self, file_name):
        """
        delete the chosen file
        :param file_name: delete file's name
        :return:
        """
        if 'pkl' not in file_name:
            file_name += '.pkl'
        my_file = self.path_name + "-%s" % file_name
        if os.path.exists(my_file):
            os.remove(my_file)
        else:
            print('no such file:%s' % my_file)


class UserInfoManager:
    """
    UserInfo manager is designed to load and save all user info.
    No matter what user info name.
    """
    # This user_info_name belongs to the whole class.
    # So it should be init for only once.
    path_name = ''

    @classmethod
    def __init__(cls, user_info_name=None):
        """
        cls.user_info_name should only init for only once.
        :param user_info_name: the name of user info
        """
        if not cls.path_name:
            cls.path_name = "user/" + user_info_name

    def save_user_info(self, user_info, save_name: str):
        """
        Save user info to user/ dir.
        :param user_info: user info
        :param save_name: user info saved name.
        :return: None
        """
        if 'csv' not in save_name:
            save_name += '.csv'
        if not os.path.exists('user'):
            os.mkdir('user')
        pickle.dump(user_info, open(self.path_name + "-%s" % save_name, "wb"))

    def load_user_info(self, user_info_name: str):
        """
        Load user info from user/ dir via user info name.
        :return: loaded user info
        """
        if 'csv' not in user_info_name:
            user_info_name += '.csv'
        if not os.path.exists(self.path_name + "-%s" % user_info_name):
            raise OSError('There is no user info named %s in model/ dir' % user_info_name)
        return pickle.load(open(self.path_name + "-%s" % user_info_name, "rb"))

    @staticmethod
    def clean_workspace(clean=False):
        """
        Clean the whole workspace.
        All File in user/ dir will be removed.
        :param clean: Boolean. Clean workspace or not.
        :return: None
        """
        if clean and os.path.exists('user'):
            shutil.rmtree('user')


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning Rate   : {args.lr}')
    print('DataSet parameters:')
    print(f'    Data Set  : {args.dataset}')
    print('Federated parameters:')
    print(f'    Global Rounds      : {args.epochs}')
    print(f'    The num of Clients : {args.clients_num}')
    print(f'    The size of Cache  : {args.cachesize}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def count_top_items(num, items):
    """
    在items中选择出现频次最高的num个
    :param num: 选择出现频次最高的num个
    :param items: 输入的items为二阶列表。例如[[1,2,3],[12,3,5]]
    :return:
    """
    items = list(chain.from_iterable(items))
    # 找到观看历史中次数最多的前num部电影
    count = Counter(items)
    top_items = np.array(count.most_common(num))[:, 0]
    return top_items

