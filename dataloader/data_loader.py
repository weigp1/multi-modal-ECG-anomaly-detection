import math
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch.utils.data as data


class DataIter(data.IterableDataset):

    def __init__(self, data_root, data_set):
        super(DataIter, self).__init__()
        # load params
        self.start = None
        self.end = None

        self.cross_val_x = None
        self.cross_val_y = None
        self.cross_val_hff = None
        self.cross_val_prompt = None

        self.cross_train_x = None
        self.cross_train_y = None
        self.cross_train_hff = None
        self.cross_train_prompt = None

        self.out_x = None
        self.out_y = None
        self.out_hff = None
        self.out_prompt = None

        self.data_root = data_root
        self.data_set = data_set
        self.pmpt_path = "./dataset/"

        self.mode = None
        self.val_fold = 0

        # load prompt
        self.pmpt = self.load_prompt(self.pmpt_path)

        # load dataset
        self.train_x, self.test_x, self.train_y, self.test_y, self.train_hff, self.test_hff, self.train_pmpt, self.test_pmpt = self._get_data(data_root=data_root, data_set=data_set)

        self.folded_train_x, self.folded_train_y, self.folded_train_hff, self.folded_train_prmpt = self.cross_fold(self.train_x, self.train_y, self.train_hff, self.train_pmpt)

        self.initial()

        logging.info("DataIter:: initialize the dataloader")


    def _get_data(self, data_root, data_set, test_ratio=0.2):
        data_x_pt = Path(data_root) / f'dataSet_{data_set}.csv'
        data_y_pt = Path(data_root) / f'labelSet_{data_set}.csv'
        data_hff_pt = Path(data_root) / f'featSet_{data_set}.csv'
        data_pmpt_pt = Path(data_root) / f'pmptSet_{data_set}.csv'

        data_x = pd.read_csv(data_x_pt, sep=",", header=None)
        data_y = pd.read_csv(data_y_pt, sep=",", header=None)
        data_hff = pd.read_csv(data_hff_pt, sep=",", header=None)
        data_pmpt = pd.read_csv(data_pmpt_pt, sep=",", header=None)

        assert data_x.shape[0] == data_y.shape[0], 'The number of samples and labels is inconsistent'
        assert 0 <= test_ratio < 1, 'The ratio of the test set is invalid'

        np.random.seed(seed)

        shuffled_indexes = np.random.permutation(len(data_x))
        test_size = int(len(data_x) * test_ratio)
        train_index = shuffled_indexes[test_size:]
        test_index = shuffled_indexes[:test_size]

        return data_x.iloc[train_index], data_x.iloc[test_index], data_y.iloc[train_index], data_y.iloc[test_index], \
            data_hff.iloc[train_index], data_hff.iloc[test_index], data_pmpt.iloc[train_index], data_pmpt.iloc[test_index]


    def cross_fold(self, train_x, train_y, train_hff, train_pmpt):
        num_data = len(train_x)
        group_size = num_data // 5

        grouped_train_x = []
        grouped_train_y = []
        grouped_train_hff = []
        grouped_train_prmpt = []

        for g_id in range(4):
            group_train_x = train_x[0 + g_id * group_size : (g_id + 1) * group_size]
            group_train_y = train_y[0 + g_id * group_size : (g_id + 1) * group_size]
            group_train_hff = train_hff[0 + g_id * group_size : (g_id + 1) * group_size]
            group_train_prmpt = train_pmpt[0 + g_id * group_size : (g_id + 1) * group_size]

            grouped_train_x.append(group_train_x)
            grouped_train_y.append(group_train_y)
            grouped_train_hff.append(group_train_hff)
            grouped_train_prmpt.append(group_train_prmpt)

        grouped_train_x.append(train_x[4 * group_size:])
        grouped_train_y.append(train_y[4 * group_size:])
        grouped_train_hff.append(train_hff[4 * group_size:])
        grouped_train_prmpt.append(train_pmpt[4 * group_size:])

        return grouped_train_x, grouped_train_y, grouped_train_hff, grouped_train_prmpt


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

        self.cross_train_x = pd.concat(train_x, ignore_index=True)
        self.cross_train_y = pd.concat(train_y, ignore_index=True)
        self.cross_train_hff = pd.concat(train_hff, ignore_index=True)
        self.cross_train_prompt = pd.concat(train_prmpt, ignore_index=True)

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

            self.cross_train_x = pd.concat(train_x, ignore_index=True)
            self.cross_train_y = pd.concat(train_y, ignore_index=True)
            self.cross_train_hff = pd.concat(train_hff, ignore_index=True)
            self.cross_train_prompt = pd.concat(train_prmpt, ignore_index=True)
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
        out_x = torch.tensor(self.out_x.iloc[self.start: self.end].values, dtype=torch.float32)
        out_y = torch.tensor(self.out_y.iloc[self.start: self.end].values, dtype=torch.float32)
        out_hff = torch.tensor(self.out_hff.iloc[self.start: self.end].values, dtype=torch.float32)
        out_prompt = torch.tensor(self.out_prompt.iloc[self.start: self.end].values, dtype=torch.float32)

        sum_iter = zip(out_x, out_y, out_hff, out_prompt)

        return iter(sum_iter)

    def load_prompt(self, pmpt_path):
        pmpt_pt = Path(pmpt_path) / f'pmptSet_{self.data_set}.csv'
        pmpt = pd.read_csv(pmpt_pt)
        return pmpt


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
    worker_info = data.get_worker_info()    # 获取 worker 的信息，包括 dataset, num_workers, id 等
    dataset = worker_info.dataset           # 获取当前 worker 进程中的 dataset 副本

    overall_start = dataset.start           # 获取 dataset 的起始索引
    overall_end = dataset.end               # 获取 dataset 的结束索引

    # 计算每个 worker 应该处理的数据量
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))     # 向上取整，确保所有数据都被处理

    worker_id = worker_info.id                                                                      # 获取当前 worker 的 ID

    # 计算当前 worker 应该处理的数据范围
    dataset.start = overall_start + worker_id * per_worker                                          # 计算当前 worker 的起始索引
    dataset.end = min(dataset.start + per_worker, overall_end)                                      # 计算当前 worker 的结束索引，避免超出 overall_end
