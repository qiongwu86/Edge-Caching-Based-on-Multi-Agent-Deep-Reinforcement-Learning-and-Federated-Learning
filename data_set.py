import os
import pandas as pd
import numpy as np
import random
from collections import namedtuple

BuiltinData_set = namedtuple('BuiltinData_set', ['url', 'path', 'sep', 'reader_params'])

BUILTIN_DATA_SETS = {
    'ml-100k':
        BuiltinData_set(
            url='http://files.grouplens.org/data_sets/movielens/ml-100k.zip',
            path='data/ml-100k/u.data',
            sep='\t',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
        ),
    'ml-1m':
        BuiltinData_set(
            url='http://files.grouplens.org/data_sets/movielens/ml-1m.zip',
            path='data/ml-1m/ratings.dat',
            sep='::',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
}


class DataSet:
    """Base class for loading data_sets.

    Note that you should never instantiate the :class:`Data_set` class directly
    (same goes for its derived classes), but instead use one of the below
    available methods for loading data_sets."""

    def __init__(self):
        pass

    @classmethod
    def LoadDataSet(cls, name='ml-100k'):
        """Load a built-in data_set.

        :param name:string: The name of the built-in data_set to load.
                Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                Default is 'ml-100k'.
        :return: ratings for each line.
        """
        try:
            data_set = BUILTIN_DATA_SETS[name]
        except KeyError:
            raise ValueError('unknown data_set ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATA_SETS.keys()) + '.')
        if not os.path.isfile(data_set.path):
            raise OSError(
                "Data_set data/" + name + " could not be found in this project.\n"
                                          "Please download it from " + data_set.url +
                ' manually and unzip it to data/ directory.')
        ratings_header = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(data_set.path, sep=data_set.sep, header=None, names=ratings_header, engine='python',
                              encoding='latin-1')
        ratings = ratings.drop(['timestamp'], axis=1)  # 删除timestamp
        ratings['rating'] = (ratings['rating']) / 5  # 归一化处理
        if name == 'ml-100k':
            # 如果数据集是ml-100k则对数据集进行重新排序
            ratings = ratings.sort_values(by='user_id', ignore_index=True)
        print("Load " + name + " data_set success.\n")
        return ratings

    @classmethod
    def SplitTrainTest(cls, ratings, test_size=0.2):
        """
        Split rating data to training set and test set.

        The default `test_size` is the test percentage of test size.

        The rating file should be a instance of DataSet.

        :param ratings: raw data_set (dataFrame)
        :param test_size: the percentage of test size.
        :return: train_set and test_set
        """

        train_list = []
        test_list = []
        train_set_len = 0
        test_set_len = 0

        for num in np.arange(0, ratings.shape[0]):
            if random.random() <= test_size:
                test_list.extend(ratings.iloc[num])
                test_set_len += 1
            else:
                train_list.extend(ratings.iloc[num])
                train_set_len += 1
        # 得到train和test的数组形式 并对其进行reshape
        # 6代表着user_id|movie_id|rating|gender|age|occupation|label
        train_array = np.array(train_list, dtype=float)
        train_array = train_array.reshape((int(train_array.shape[0] / 7), 7))
        test_array = np.array(test_list, dtype=float)
        test_array = test_array.reshape((int(test_array.shape[0] / 7), 7))
        # 将array转换为dataFrame
        train = pd.DataFrame(train_array)
        test = pd.DataFrame(test_array)
        print('split rating data to training set and test set success.')
        print('train set size = %s' % train_set_len)
        print('test set size = %s' % test_set_len)
        # return train, test, train_array, test_array
        return train_array, test_array


def convert(data_set, movie_max):
    """
    convert data_set into an array with users in lines and movies in columns

    :param movie_max: num of movie
    :param data_set: the input array with the first col--user; the second col--item; the third col--rating
    :return: an array with users in lines and movies in columns
    """
    new_data_set = []
    user_in_dataset = np.unique(data_set[:, 0].astype(np.uint16))
    for id_users in user_in_dataset:
        id_movies = data_set[:, 1][data_set[:, 0] == id_users].astype(np.uint16)
        id_ratings = data_set[:, 2][data_set[:, 0] == id_users]
        # ratings第一列保存用户id， 后movie_max列为对应电影索引，最后三列为age、gender、occupation
        ratings = np.zeros(movie_max + 1 + 3)
        ratings[0] = id_users
        ratings[id_movies] = id_ratings
        ratings[[-1, -2, -3]] = data_set[:, [-1, -2, -3]][data_set[:, 0] == id_users][0]
        new_data_set.append(list(ratings))
    new_data_set = np.array(new_data_set, dtype=float)
    return new_data_set
