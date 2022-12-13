import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from semi_dataset import SemiDataset
from imbalance_semi_dataset import ImbalanceSemiDataset


class SemiDatasetsGenerator(SemiDataset):
    def __init__(self, file_path, num_sup_list=[1000, 300, 100], num_unsup=0, num_dev=0, num_test=0, seed=666):
        SemiDataset.__init__(file_path, max(num_sup_list), num_unsup,
                             num_dev, num_test, seed)

        self._num_sup_list = num_sup_list
        self._sup_dataset_list = {}

    def dataset_to_csv(self):

        if self._sup_dataset_list is not None:
            for key in self._sup_dataset_list.keys():
                file_name = "sup_"+str(key)+".csv"
                current_path = os.path.join(self._save_path, file_name)
                self._sup_dataset_list[key].to_csv(current_path, index=False)

        if self._unsup_dataset is not None:
            file_name = "unsup_"+str(self._num_unsup)+".csv"
            current_path = os.path.join(self._save_path, file_name)
            self._unsup_dataset.to_csv(current_path, index=False)

        if self._dev_dataset is not None:
            file_name = "dev.csv"
            current_path = os.path.join(self._save_path, file_name)
            self._dev_dataset.to_csv(current_path, index=False)

        if self._test_dataset is not None:
            file_name = "test.csv"
            current_path = os.path.join(self._save_path, file_name)
            self._test_dataset.to_csv(current_path, index=False)

        print("The whole dataset has been saved in {}\n".format(self._save_path))

    def sample(self):
        # "type:ignore" means ignore the issues reported by pylance
        # sup dataset
        if self._origin_train_dataset is not None and self._num_sup_list is not None:
            for current_length in self._num_sup_list:
                self._sup_dataset_list[current_length], self._origin_train_dataset = self._sample_target_dataset(
                    self._origin_train_dataset, current_length, type='sup')  # type:ignore
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
            self._unsup_dataset, self._origin_train_dataset = self._sample_target_dataset(
                self._origin_train_dataset, self._num_unsup, type="unsup")  # type:ignore

        def get_semi_dataset(self, clean=True, data_augmentation=True, return_csv=True):

            # self._sup_dataset = self.generate_unsup_dataset(self._sup_dataset)

            if clean:
                self.clean_dataset()

            self.label_rearrangement()

            self.sample()

            if data_augmentation:
                self.generate_unsup_dataset(self._unsup_dataset)

            if return_csv:
                self.dataset_to_csv()

            self._write_txt_file()
