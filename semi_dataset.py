import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class SemiDataloader(DataLoader):
    def __init__(self, file_path, data_name, num_sup=0, num_unsup=0, num_dev=0, num_test=0):
        super(DataLoader, self).__init__()
        self.data_file = SemiDataset(
            file_path, data_name, num_sup, num_unsup, num_dev, num_test)


class SemiDataset():
    def __init__(self, file_path, data_name, num_sup=0, num_unsup=0, num_dev=0, num_test=0):

        self.__file_path = file_path
        self.__data_name = data_name
        self.__num_sup = num_sup
        self.__num_unsup = num_unsup
        self.__num_dev = num_dev
        self.__num_test = num_test
        # the origin dataset without any process
        self.__origin_sup_dataset = None
        self.__origin_dev_dataset = None
        self.__origin_test_dataset = None
        # the dataset after process
        self.__sup_dataset = None
        self.__unsup_dataset = None
        self.__dev_dataset = None
        self.__test_dataset = None

    def __read_dataset(self, type='sup'):
        file_name = type+'.csv'
        file_path = os.path.join(self.__file_path, self.__data_name, file_name)
        try:
            dataset = pd.read_csv(file_path, sep="\t", header=True)
            print("Read original {} dataset successful!".format(type))
            return dataset
        except:
            print("ERROR! Read original {} dataset unsuccessful!".format(type))
            return None

    def clean_sentence(self, scale="all"):
        if scale == "all":
            print("clean the whole dataset")

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
        semi_dataset_length = {}


if __name__ == "__main__":
    # SemiDataset = SemiDataset('imdb',)
    print("------Test------")
