import os
import random
import pickle

import numpy as np
import pandas as pd
import math

import torch.utils.data as data
import logging


class DataIter(data.IterableDataset):

    def __init__(self, data_root, data_set):
        super(DataIter, self).__init__()
        # load params
        self.data_root = data_root
        self.data_set = data_set
        self.prompt_path = '../prompt'

        self.mode = None
        self.val_fold = 0

        # load data
        self.train_x, self.test_x, self.train_y, self.test_y, self.train_hff, self.test_hff, self.train_pmpt, self.test_pmpt = self._get_data(data_root=data_root, data_set=data_set)

        self.folded_train_x, self.folded_train_y, self.folded_train_hff, self.folded_train_prmpt = self.cross_fold(
            [self.train_x, self.train_y, self.train_hff, self.train_pmpt])

        self.initial()
        logging.info("DataIter:: initialize the dataloader")

    def _get_data(self, data_root, data_set, test_ratio=0.2, seed=None):

        data_x_pt = os.path.join(data_root, 'dataSet_' + data_set + '.csv')
        data_y_pt = os.path.join(data_root, 'labelSet' + data_set + '.csv')
        data_hff_pt = os.path.join(data_root, 'featSet' + data_set + '.csv')
        data_pmpt_pt = os.path.join(data_root, 'featSet' + data_set + '.csv')

        data_x = pd.read_csv(data_x_pt, sep=" ", header=None)
        data_y = pd.read_csv(data_y_pt, sep=" ", header=None)
        data_hff = pd.read_csv(data_hff_pt, sep=" ", header=None)
        data_pmpt = pd.read_csv(data_pmpt_pt, sep=" ", header=None)

        assert data_x.shape[0] == data_y.shape[0], '样本和标签个数不一致'
        assert 0 <= test_ratio < 1, '无效的测试比例'
        if seed:
            np.random.seed(seed)
        shuffled_indexes = np.random.permutation(len(data_x))
        test_size = int(len(data_x) * test_ratio)
        train_index = shuffled_indexes[test_size:]
        test_index = shuffled_indexes[:test_size]
        return data_x[train_index], data_x[test_index], data_y[train_index], data_y[test_index], \
            data_hff[train_index], data_hff[test_index], data_pmpt[train_index], data_pmpt[test_index]

    def cross_fold(self, data_list):
        ref_data = data_list[0]
        num_data = len(ref_data)
        group_size = num_data // 5

        zip_list = list(zip(data_list[0], data_list[1], data_list[2], data_list[3]))
        random.shuffle(zip_list)
        train_x, train_y, train_hff, train_pmpt = zip(*zip_list)

        grouped_train_x = []
        grouped_train_y = []
        grouped_train_hff = []
        grouped_prmpt = []

        for g_id in range(4):
            group_train_x = train_x[0 + g_id * group_size:(g_id + 1) * group_size]
            group_train_y = train_y[0 + g_id * group_size:(g_id + 1) * group_size]
            group_train_hff = train_hff[0 + g_id * group_size:(g_id + 1) * group_size]
            group_prmpt = train_pmpt[0 + g_id * group_size:(g_id + 1) * group_size]

            grouped_train_x.append(group_train_x)
            grouped_train_y.append(group_train_y)
            grouped_train_hff.append(group_train_hff)
            grouped_prmpt.append(group_prmpt)

        grouped_train_x.append(train_x[4 * group_size:])
        grouped_train_y.append(train_y[4 * group_size:])
        grouped_train_hff.append(train_hff[4 * group_size:])
        grouped_prmpt.append(train_pmpt[4 * group_size:])

        return grouped_train_x, grouped_train_y, grouped_train_hff, grouped_prmpt

    def initial(self):
        val_fold_ind = 0
        train_x = list(self.folded_train_x)
        train_y = list(self.folded_train_y)
        train_hff = list(self.folded_train_hff)
        train_prmpt = list(self.folded_train_prmpt)

        self.cross_val_x = train_x.pop(val_fold_ind)
        self.cross_val_y = train_y.pop(val_fold_ind)
        self.cross_val_hff = train_hff.pop(val_fold_ind)
        self.cross_val_prompt = train_prmpt.pop(val_fold_ind)

        self.cross_train_x = train_x[0] + train_x[1] + train_x[2] + train_x[3]
        self.cross_train_y = train_y[0] + train_y[1] + train_y[2] + train_y[3]
        self.cross_train_hff = train_hff[0] + train_hff[1] + train_hff[2] + train_hff[3]
        self.cross_train_prompt = train_prmpt[0] + train_prmpt[1] + train_prmpt[2] + train_prmpt[3]

        self.out_x = self.cross_train_x
        self.out_y = self.cross_train_y
        self.out_hff = self.cross_train_hff
        self.out_prompt = self.cross_train_prompt

        self.start = 0
        self.end = len(self.out_x)

    def reset(self, mode):

        if mode == 'train':
            self.mode = 'train'
            val_fold_ind = self.val_fold % 5

            train_x = list(self.folded_train_x)
            train_y = list(self.folded_train_y)
            train_hff = list(self.folded_train_hff)
            train_prmpt = list(self.folded_train_prmpt)

            self.cross_val_x = train_x.pop(val_fold_ind)
            self.cross_val_y = train_y.pop(val_fold_ind)
            self.cross_val_hff = train_hff.pop(val_fold_ind)
            self.cross_val_prompt = train_prmpt.pop(val_fold_ind)

            self.cross_train_x = train_x[0] + train_x[1] + train_x[2] + train_x[3]
            self.cross_train_y = train_y[0] + train_y[1] + train_y[2] + train_y[3]
            self.cross_train_hff = train_hff[0] + train_hff[1] + train_hff[2] + train_hff[3]
            self.cross_train_prompt = train_prmpt[0] + train_prmpt[1] + train_prmpt[2] + train_prmpt[3]

            self.val_fold += 1

            self.out_x = self.cross_train_x
            self.out_y = self.cross_train_y
            self.out_hff = self.cross_train_hff
            self.out_prompt = self.cross_train_prompt

            self.end = len(self.out_x)
        elif mode == 'val':
            self.mode = 'val'
            self.out_x = self.cross_val_x
            self.out_y = self.cross_val_y
            self.out_hff = self.cross_val_hff
            self.out_prompt = self.cross_val_prompt

            self.end = len(self.out_x)

        elif mode == 'test':
            self.mode = 'test'
            self.out_x = self.test_x
            self.out_y = self.test_y
            self.out_hff = self.test_hff
            self.out_prompt = self.test_pmpt

            self.end = len(self.out_x)

    def __iter__(self):
        out_x = self.out_x[self.start: self.end]
        out_y = self.out_y[self.start: self.end]
        out_hff = self.out_hff[self.start: self.end]
        out_prompt = self.out_prompt[self.start: self.end]
        sum_iter = zip(out_x, out_y, out_hff, out_prompt)

        return iter(sum_iter)

    def __len__(self):
        return len(self.out_x)


def worker_init_fn(worker_id):
    """
    初始化 DataLoader 的每个 worker 进程。

    该函数确保每个 worker 处理不同的数据分片，避免数据重复。
    它通过调整 dataset 的 start 和 end 属性来实现数据分片。

    Args:
        worker_id (int): worker 进程的 ID。
    """
    worker_info = data.get_worker_info()  # 获取 worker 的信息，包括 dataset, num_workers, id 等
    dataset = worker_info.dataset  # 获取当前 worker 进程中的 dataset 副本

    overall_start = dataset.start  # 获取 dataset 的起始索引
    overall_end = dataset.end  # 获取 dataset 的结束索引

    # 计算每个 worker 应该处理的数据量
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))  # 向上取整，确保所有数据都被处理

    worker_id = worker_info.id  # 获取当前 worker 的 ID

    # 计算当前 worker 应该处理的数据范围
    dataset.start = overall_start + worker_id * per_worker  # 计算当前 worker 的起始索引
    dataset.end = min(dataset.start + per_worker, overall_end)  # 计算当前 worker 的结束索引，避免超出 overall_end
