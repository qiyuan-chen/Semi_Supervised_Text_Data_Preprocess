import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
import random
from torch.utils.data import DataLoader, Dataset
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split



class SemiDataset():
    def __init__(self, file_path, num_sup=0, num_unsup=0, num_dev=0, num_test=0, seed=666):

        self.__file_path = file_path
        self.__num_sup = num_sup
        self.__num_unsup = num_unsup
        self.__num_dev = num_dev
        self.__num_test = num_test
        # the origin dataset without any process
        self.__origin_train_dataset = self.__read_dataset(type='train')
        self.__origin_dev_dataset = self.__read_dataset(type='dev')
        self.__origin_test_dataset = self.__read_dataset(type='test')
        # the dataset after process
        self.__num_sup = num_sup
        self.__num_unsup = num_unsup
        self.__num_dev = num_dev

        self.__sup_dataset = None
        self.__unsup_dataset = None
        self.__dev_dataset = None
        self.__test_dataset = None
        self.__seed = seed

        # set the random seed for the whole environment
        os.environ['PYTHONHASHSEED'] = str(self.__seed)
        random.seed(self.__seed)
        np.random.seed(self.__seed)
        torch.manual_seed(self.__seed)
        torch.cuda.manual_seed(self.__seed)
        torch.cuda.manual_seed_all(self.__seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    def __read_dataset(self, type='train'):
        file_name = type+'.csv'
        file_path = os.path.join(self.__file_path, file_name)
        if os.path.exists(file_path):
            print(file_path)
            dataset = pd.read_csv(file_path)
            print("Read original {} dataset successful!".format(type))
            return dataset
        else:
            print("ERROR! Read original {} dataset unsuccessful!".format(type))
            return None

    def clean_sentence(self, scale="all"):
        if scale == "all":
            print("clean the whole dataset")

    def __clean_sentence(self, sent):
        try:
            sent = sent.replace('\n', ' ').replace(
                '\\n', ' ').replace('\\', ' ')

            sent = re.sub('<[^<]+?>', '', sent)

            return sent.lower()
        except:
            print(sent)
            return ' '

    def get_semi_dataset(self):
        semi_dataset = {}
        if self.__num_sup != 0:
            semi_dataset['sup'] = self.__sup_dataset
        if self.__num_unsup != 0:
            semi_dataset['unsup'] = self.__unsup_dataset
        if self.__num_dev != 0:
            semi_dataset['dev'] = self.__dev_dataset
        if self.__num_test != 0:
            semi_dataset['test'] = self.__test_dataset

        if len(semi_dataset) != 0:
            return semi_dataset
        else:
            print("No Data Returned!")
            return None

    def get_semi_dataset_length(self):
        semi_dataset_length = {"sup": self.__num_sup,
                               "unsup": self.__num_unsup,
                               "dev": self.__num_dev,
                               "test": self.__num_test}
        return semi_dataset_length

    def __sample_target_dataset(self, target_dataset: pd.DataFrame, target_length, type='sup'):
        if type != "unsup":
            return_dataset = pd.DataFrame(columns=["sentence", "label"])
            remain_dataset = pd.DataFrame(columns=["sentence", "label"])
            for label in target_dataset["label"].unique():
                current_data = target_dataset[target_dataset["label"] == label]

                if target_length < current_data.shape[0]:
                    return_data, remain_data = train_test_split(
                        current_data, train_size=target_length, random_state=self.__seed)
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
        elif type == "unsup":
            return_dataset = {"sentence": [], "aug_sentence": []}
            for label in target_dataset["label"].unique():
                sampled_data = target_dataset[target_dataset["label"] == label].sample(
                    target_length)
                return_dataset["label"].extend(sampled_data["label"])
                return_dataset["sentence"].extend(sampled_data["sentence"])

            return_dataset = pd.DataFrame.from_dict(
                return_dataset).sample(frac=1).reset_index(drop=True)

            print("The {} dataset's length is:".format(type), str(
                len(return_dataset["label"])))

            remain_dataset = target_dataset[~(
                target_dataset['sentence'].isin(return_dataset['sentence']))]
            return return_dataset, remain_dataset

    def sample(self):
        # "type:ignore" means ignore the issues reported by pylance
        if self.__origin_train_dataset is not None and self.__num_sup != 0:
            self.__sup_dataset, self.__origin_train_dataset = self.__sample_target_dataset(
                self.__origin_train_dataset, self.__num_sup, type='sup')  # type:ignore

        if self.__origin_test_dataset is not None and self.__num_test != 0:
            self.__test_dataset, self.__origin_test_dataset = self.__sample_target_dataset(
                self.__origin_test_dataset, self.__num_test, type='test')  # type:ignore

        if self.__origin_dev_dataset is None and self.__num_dev != 0:
            self.__dev_dataset, self.__origin_train_dataset = self.__sample_target_dataset(
                self.__origin_train_dataset, self.__num_dev, type='dev')  # type:ignore
        elif self.__origin_dev_dataset is not None and self.__num_dev != 0:
            self.__dev_dataset, self.__origin_dev_dataset = self.__sample_target_dataset(
                self.__origin_dev_dataset, self.__num_dev, type='dev')  # type:ignore


if __name__ == "__main__":
    semi_data = SemiDataset(file_path="/home/chenqy/Dataset/IMDB",
                            num_sup=500, num_test=12500, num_dev=500)
    semi_data.sample()
    data = semi_data.get_semi_dataset()
