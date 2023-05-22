# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/3/10

import os
import pandas as pd
from collections import namedtuple

BuiltinData_set = namedtuple('BuiltinData_set', ['url', 'path_user', 'path_occupation', 'sep', 'reader_params'])

BUILTIN_DATA_SETS = {
    'ml-100k':
        BuiltinData_set(
            url='http://files.grouplens.org/data_sets/movielens/ml-100k.zip',
            path_user='data/ml-100k/u.user',
            path_occupation='data/ml-100k/u.occupation',
            sep='|',
            reader_params=dict(line_format='user id | age | gender | occupation | zip code',
                               sep='|')
        ),
    'ml-1m':
        BuiltinData_set(
            url='http://files.grouplens.org/data_sets/movielens/ml-1m.zip',
            path_user='data/ml-1m/users.dat',
            path_occupation=None,
            sep='::',
            reader_params=dict(line_format='user item rating timestamp',
                               sep='::')
        ),
}


class UserInfo:
    """Base class for loading Usrinfo.

      Note that you should never instantiate the :class:`UserInfo` class directly
      (same goes for its derived classes), but instead use one of the below
      available methods for loading data_sets."""

    def __init__(self):
        pass

    @classmethod
    def load_user_info(cls, name='ml-100k'):
        """Load a built-in data_set user_info.

            :param name:string: The name of the built-in data_set to load.
                    Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                    Default is 'ml-100k'.
            :return: user_info.
            """
        try:
            data_set = BUILTIN_DATA_SETS[name]
        except KeyError:
            raise ValueError('unknown data_set ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATA_SETS.keys()) + '.')
        if not os.path.isfile(data_set.path_user):
            raise OSError(
                "Dataset user/" + name + " could not be found in this project.\n"
                                         "Please download it from " + data_set.url +
                ' manually and unzip it to data/ directory.')

        user_info_header = ['user_id', 'age', 'gender', 'occupation', 'zip'] if name == 'ml-100k' else ['user_id',
                                                                                                        'gender', 'age',
                                                                                                        'occupation',
                                                                                                        'zip']
        user_info = pd.read_csv(data_set.path_user, sep=data_set.sep, header=None, names=user_info_header,
                                engine='python',
                                encoding='latin-1')
        user_info = cls.process_user_info(user_info, name)
        print("Load " + name + " user_info success.\n")
        return user_info

    @classmethod
    def process_user_info(cls, user_info, name):
        """process user_info.

           :param name:string: The name of the built-in data_set to load.
                   Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                   Default is 'ml-100k'.
           :return: processed user_info
           """
        data_set = BUILTIN_DATA_SETS[name]

        # occupation
        # occupation_map构造职业到索引的dictionary
        def occupation_map(occupation, user_info_occu):
            occupation_dic = {}
            for num, occu in user_info_occu.iterrows():
                occupation_dic[occu[0]] = num
            return occupation_dic[occupation]

        # 只有ml-100k数据集需要对职业进行映射处理
        if name == 'ml-100k':
            user_info_occupation = pd.read_csv(data_set.path_occupation, sep=data_set.sep, header=None, engine='python',
                                               encoding='latin-1')
            user_info['occupation'] = user_info['occupation'].apply(
                lambda occupation: occupation_map(occupation, user_info_occupation))
        # 对职业进行归一化0~20映射到0~1
        # diff 0.05
        user_info['occupation'] = user_info['occupation'] / 20

        # gender
        # 对gender进行映射
        # diff 0.15
        user_info['gender'] = user_info['gender'].map({'M': 0.3, 'F': 0.15})

        # age
        # 对age进行映射  1~10:1/7   11~20:2/7   21~29:3/7   30~38:4/7   39~47:5/7   48~55:6/7   56+:1
        # diff 1/7 0.1428
        def age_map(age):
            if 0 <= age <= 10:
                return 1 / 7
            elif 11 <= age <= 20:
                return 2 / 7
            elif 21 <= age <= 29:
                return 3 / 7
            elif 30 <= age <= 38:
                return 4 / 7
            elif 39 <= age <= 47:
                return 5 / 7
            elif 48 <= age <= 55:
                return 6 / 7
            else:
                return 1

        user_info['age'] = user_info['age'].apply(lambda age: age_map(age))

        # zip
        # 删除zip
        user_info = user_info.drop(['zip'], axis=1)

        return user_info
