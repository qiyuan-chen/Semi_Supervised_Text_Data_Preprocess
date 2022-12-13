import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from numpy import linspace
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from semi_dataset import SemiDataset


class ImbalanceSemiDataset(SemiDataset):
    def __init__(self, file_path, N_max_sup=0, N_max_unsup=0, num_dev=0,
                 num_test=0, sup_ratio=100, unsup_ratio=100, seed=666):
        SemiDataset.__init__(self, file_path, N_max_sup,
                             N_max_unsup, num_dev, num_test, seed)

        self._sup_ratio = sup_ratio
        self._unsup_ratio = unsup_ratio
        self._seed = seed
        random.seed(self._seed)
        self._N_max_sup = N_max_sup
        self._N_max_unsup = N_max_unsup

        self._unsup_num_label = None
        self._sup_num_label = None

        self._N_min_sup = (int)(self._N_max_sup / self._sup_ratio)
        self._N_min_unsup = (int)(self._N_max_unsup / self._unsup_ratio)

        self._N_sup = self._get_N(
            self._N_max_sup, self._N_min_sup, len(self._label), self._sup_ratio)

        self._N_unsup = self._get_N(
            self._N_max_unsup, self._N_min_unsup, len(self._label), self._unsup_ratio)

    def _imbalance_sample_target_dataset(self, target_dataset, N, type="sup"):
        return_dataset = pd.DataFrame(columns=["sentence", "label"])
        remain_dataset = pd.DataFrame(columns=["sentence", "label"])
        for i, label in enumerate(self._label):
            current_data = target_dataset[target_dataset["label"] == label]
            target_length = N[i]
            if target_length < current_data.shape[0]:
                return_data, remain_data = train_test_split(
                    current_data, train_size=target_length, random_state=self._seed)
            else:
                return_data = current_data
                remain_data = None

            return_dataset = pd.concat(
                [return_dataset, return_data], ignore_index=True)
            remain_dataset = pd.concat(
                [remain_dataset, remain_data], ignore_index=True)

        print("The {} dataset's length is:".format(
            type), return_dataset.shape[0])
        return return_dataset, remain_dataset

    def _get_N(self, N_max, N_min, num_label, ratio):
        N = [N_max]
        for i in range(2, num_label):
            current_N = N_max*(ratio ** ((1-i)/(num_label-1)))
            N.append((int)(current_N))
        N.append(N_min)
        assert len(N) == num_label
        return N

    def _write_txt_file(self):
        txt_path = os.path.join(self._save_path, "readme.txt")
        with open(txt_path, "w") as f:
            f.write(
                "The dataset consists of:\n \
                sup ratio:{} N_max:{},\n \
                unsup ratio:{} N_max:{},\n \
                {} dev examples,\n \
                {} test examples".format(self._sup_ratio, self._N_max_sup, self._unsup_ratio,
                                         self._N_max_unsup, self._num_dev, self._num_test))

    def sample(self):
        # "type:ignore" means ignore the issues reported by pylance
        # sup dataset
        if self._origin_train_dataset is not None and self._num_sup != 0:
            self._sup_dataset, self._origin_train_dataset = self._imbalance_sample_target_dataset(
                self._origin_train_dataset, self._N_sup, type='sup')  # type:ignore
        # test dataset
        if self._origin_test_dataset is not None and self._num_test != 0:
            self._test_dataset, self._origin_test_dataset = self._sample_target_dataset(
                self._origin_test_dataset, self._num_test, type='test')  # type:ignore
        # dev dataset
        if self._origin_dev_dataset is None and self._num_dev != 0:
            self._dev_dataset, self._origin_train_dataset = self._sample_target_dataset(
                self._origin_train_dataset, self._num_dev, type='dev')  # type:ignore
        elif self._origin_dev_dataset is not None and self._num_dev != 0:
            self._dev_dataset, self._origin_dev_dataset = self._sample_target_dataset(
                self._origin_dev_dataset, self._num_dev, type='dev')  # type:ignore

        if self._num_unsup != 0:
            self._unsup_dataset, self._origin_train_dataset = self._imbalance_sample_target_dataset(
                self._origin_train_dataset, self._N_unsup, type="unsup")  # type:ignore


if __name__ == "__main__":
    print("-----TEST-----")
    semi_data = ImbalanceSemiDataset(file_path="/home/chenqy/Dataset/IMDB", N_max_sup=500,
                                     num_test=12500, num_dev=5000, N_max_unsup=5000,
                                     sup_ratio=10, unsup_ratio=1000)

    data = semi_data.get_semi_dataset()
