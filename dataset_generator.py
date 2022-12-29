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
from augment.eda import eda


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


class ImbalanceSemiDatasetsGenerator(ImbalanceSemiDataset):
    def __init__(self, file_path, N_max_sups=[], N_max_unsups=[], num_dev=0,
                 num_test=0, sup_ratios=[], unsup_ratios=[], seed=666):
        ImbalanceSemiDataset.__init__(self, file_path, max(N_max_sups), max(N_max_unsups), num_dev,
                                      num_test, min(sup_ratios), min(unsup_ratios), seed)

        self._N_max_sups = N_max_sups
        self._N_max_unsups = N_max_unsups
        self._sup_ratios = sup_ratios
        self._unsup_ratios = unsup_ratios

    def _generate_current_folder_name(self, N_max_sup, N_max_unsup, sup_ratio, unsup_ratio):
        folder_name = "N_max_sup_{}_N_max_unsup_{}_sup_ratio_{}_unsup_ratio_{}".format(
            N_max_sup, N_max_unsup, sup_ratio, unsup_ratio)

        save_path = os.path.join(self._file_path, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return folder_name, save_path

    def _sample_origin_datasets(self, N_max_sup, N_max_unsup, sup_ratio, unsup_ratio):
        sup_dataset = self._sup_dataset
        unsup_dataset = self._unsup_dataset

        N_sup = self._get_N(N_max_sup, (int)(N_max_sup/sup_ratio),
                            len(self._label), sup_ratio)
        N_unsup = self._get_N(N_max_unsup, (int)(N_max_unsup/unsup_ratio),
                              len(self._label), unsup_ratio)

        current_dataset = {"test": self._test_dataset,
                           "dev": self._dev_dataset}

        current_sup_dataset, _ = self._imbalance_sample_target_dataset(
            sup_dataset, N_sup, type="sup")
        current_unsup_dataset, _ = self._imbalance_sample_target_dataset(
            unsup_dataset, N_unsup, type="unsup")

        current_dataset["sup"] = current_sup_dataset
        current_dataset["unsup"] = current_unsup_dataset

        return current_dataset

    def _current_dataset_to_csv(self, current_dataset, file_path):

        if current_dataset["sup"] is not None:
            file_name = "sup.csv"
            current_path = os.path.join(file_path, file_name)
            current_dataset["sup"].to_csv(current_path, index=False)

        if current_dataset["unsup"] is not None:
            file_name = "unsup.csv"
            current_path = os.path.join(file_path, file_name)
            current_dataset["unsup"].to_csv(current_path, index=False)

        if current_dataset["dev"] is not None:
            file_name = "dev.csv"
            current_path = os.path.join(file_path, file_name)
            current_dataset["dev"].to_csv(current_path, index=False)

        if current_dataset["test"] is not None:
            file_name = "test.csv"
            current_path = os.path.join(file_path, file_name)
            current_dataset["test"].to_csv(current_path, index=False)

        print("The whole dataset has been saved in {}\n".format(file_path))

    def generate_current_unsup_dataset(self, unsup_dataset_origin):
        unsup_dataset_dict = {"sentence": [], "aug": []}
        unsuccess_num = 0
        for i in tqdm(range(unsup_dataset_origin.shape[0])):
            # print(eda(unsup_dataset_origin["sentence"][i]))
            try:
                unsup_dataset_dict["aug"].append(
                    eda(unsup_dataset_origin["sentence"][i])[0])
            except:
                unsuccess_num += 1
                unsup_dataset_dict["aug"].append(
                    unsup_dataset_origin["sentence"][i])

            unsup_dataset_dict["sentence"].append(
                unsup_dataset_origin["sentence"][i])

        if unsuccess_num != 0:
            print("{} samples without correct augmentation".format(unsuccess_num))

        # print(pd.DataFrame.from_dict(unsup_dataset_dict).head())
        return pd.DataFrame.from_dict(unsup_dataset_dict)

    def generate_datasets(self, clean=True, data_augmentation=True, return_csv=True):

        if clean:
            self.clean_dataset()

        self.label_rearrangement()

        self.sample()

        for N_max_sup in self._N_max_sups:
            for N_max_unsup in self._N_max_unsups:
                for sup_ratio in self._sup_ratios:
                    for unsup_ratio in self._unsup_ratios:
                        print("N_max_sup:{},N_max_unsup:{},sup_ratio:{},unsup_ratio:{}".format(
                            N_max_sup, N_max_unsup, sup_ratio, unsup_ratio))
                        current_dataset = self._sample_origin_datasets(
                            N_max_sup, N_max_unsup, sup_ratio, unsup_ratio)
                        current_dataset["unsup"] = self.generate_current_unsup_dataset(
                            current_dataset["unsup"])
                        _, file_path = self._generate_current_folder_name(
                            N_max_sup, N_max_unsup, sup_ratio, unsup_ratio)
                        self._current_dataset_to_csv(
                            current_dataset, file_path)


if __name__ == "__main__":
    print("-----TEST-----")
    semi_data = ImbalanceSemiDatasetsGenerator(file_path="/home/chenqy/Dataset/IMDB", N_max_sups=[500, 2500],
                                               num_test=12500, num_dev=5000, N_max_unsups=[5000],
                                               sup_ratios=[10, 100], unsup_ratios=[10, 100])

    data = semi_data.generate_datasets()
