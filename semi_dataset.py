import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from augment.eda import eda


class SemiDataset():
    def __init__(self, file_path, num_sup=0, num_unsup=0, num_dev=0, num_test=0, seed=666):

        self._file_path = file_path
        self._num_sup = num_sup
        self._num_unsup = num_unsup
        self._num_dev = num_dev
        self._num_test = num_test
        # the origin dataset without any process
        self._origin_train_dataset = self._read_dataset(type='train')
        self._origin_dev_dataset = self._read_dataset(type='dev')
        self._origin_test_dataset = self._read_dataset(type='test')

        if self._origin_train_dataset is not None:
            self._label = self._origin_train_dataset["label"].unique()

        print("Read the dataset successful!\n")

        # the dataset after process

        self._sup_dataset = None
        self._unsup_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        self._seed = seed

        # set the random seed for the whole environment
        os.environ['PYTHONHASHSEED'] = str(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

        self._folder_name, self._save_path = self._generate_folder_name()

        # nltk.download("omw-1.4")
        # nltk.download("wordnet")

    def _read_dataset(self, type='train'):
        """_summary_

        Args:
            type (str, optional): The type of the dataset, include train,dev,test. Defaults to 'train'.

        Returns:
            pd.DataFrame: Raw dataset, without any pre-processing 
        """
        file_name = type+'.csv'
        file_path = os.path.join(self._file_path, file_name)
        if os.path.exists(file_path):
            dataset = pd.read_csv(file_path)
            print("Read original {} dataset successful from {}".format(
                type, file_path))
            return dataset
        else:
            print("ERROR! Read original {} dataset unsuccessful!".format(type))
            return None

    def _clean_dataset(self, dataset, type='train'):
        cleaned_dataset = {"sentence": []}
        print("Clean the {} dataset!".format(type))
        for i in tqdm(range(dataset.shape[0])):
            try:
                cleaned_dataset["sentence"].append(
                    self._clean_sentence(dataset["sentence"][i]))
            except:
                cleaned_dataset["sentence"].append(dataset["sentence"][i])

        cleaned_dataset["label"] = dataset["label"]

        return pd.DataFrame.from_dict(cleaned_dataset)

    def _clean_sentence(self, sent):
        try:
            sent = sent.replace('\n', ' ').replace(
                '\\n', ' ').replace('\\', ' ')

            sent = re.sub('<[^<]+?>', '', sent)

            return sent.lower()
        except:
            return ' '

    def _sample_target_dataset(self, target_dataset: pd.DataFrame, target_length, type='sup'):
        return_dataset = pd.DataFrame(columns=["sentence", "label"])
        remain_dataset = pd.DataFrame(columns=["sentence", "label"])
        for label in target_dataset["label"].unique():
            current_data = target_dataset[target_dataset["label"] == label]

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

    def _combine_target_dataset_columns(self, dataframe: pd.DataFrame, sentence_column_names=["sentence"],
                                        label_colum="label", type='train'):
        combined_dataset = pd.DataFrame(columns=["sentence", "label"])
        combined_dataset["label"] = dataframe[label_colum]
        try:
            for column_name in sentence_column_names:
                combined_dataset["sentence"] = combined_dataset["sentence"] + \
                    " " + dataframe[column_name]

            print("Combine the {} dataset's columns successful!".format(type))
            return combined_dataset

        except:
            print("ERROR!Combine the {} dataset's columns unsuccessful!".format(type))
            return

    def _label_rearrangement(self, dataset):
        dataset["label"] = dataset["label"]-1
        return dataset

    def _generate_folder_name(self):
        folder_name = "{}sup_{}unsup_{}dev_{}test".format(
            self._num_sup, self._num_unsup, self._num_dev, self._num_test)

        save_path = os.path.join(self._file_path, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return folder_name, save_path

    def _write_txt_file(self):
        txt_path = os.path.join(self._save_path, "readme.txt")
        with open(txt_path, "w") as f:
            f.write(
                "The dataset consists of:\n \
                {} sup examples,\n \
                {} unsup examples,\n \
                {} dev examples,\n \
                {} test examples".format(self._num_sup, self._num_unsup, self._num_dev, self._num_test))

    def dataset_to_csv(self):

        if self._sup_dataset is not None:
            file_name = "sup_"+str(self._num_sup)+".csv"
            current_path = os.path.join(self._save_path, file_name)
            self._sup_dataset.to_csv(current_path, index=False)

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

    def combine_columns(self, sentence_column_names=["sentence"], label_colum="label"):
        if self._origin_train_dataset is not None:
            self._origin_train_dataset = self._combine_target_dataset_columns(
                self._origin_train_dataset)

        if self._origin_dev_dataset is not None:
            self._origin_dev_dataset = self._combine_target_dataset_columns(
                self._origin_dev_dataset)

        if self._origin_test_dataset is not None:
            self._origin_test_dataset = self._combine_target_dataset_columns(
                self._origin_test_dataset)

    def clean_dataset(self, scale=["train", "dev", "test"]):
        if "train" in scale and self._origin_train_dataset is not None:
            self._origin_train_dataset = self._clean_dataset(
                self._origin_train_dataset, type="train")

        if "dev" in scale and self._origin_dev_dataset is not None:
            self._origin_dev_dataset = self._clean_dataset(
                self._origin_dev_dataset, type="dev")

        if "test" in scale and self._origin_test_dataset is not None:
            self._origin_test_dataset = self._clean_dataset(
                self._origin_test_dataset, type="test")

        print("Finish cleaning the dataset!\n")

    def get_semi_dataset(self, clean=True, data_augmentation=True, return_csv=True):

        # self._sup_dataset = self.generate_unsup_dataset(self._sup_dataset)

        if clean:
            self.clean_dataset()

        self.label_rearrangement()

        self.sample()

        if data_augmentation:
            self.generate_unsup_dataset(self._unsup_dataset)

        semi_dataset = {}
        if self._num_sup != 0:
            semi_dataset['sup'] = self._sup_dataset
        if self._num_unsup != 0:
            semi_dataset['unsup'] = self._unsup_dataset
        if self._num_dev != 0:
            semi_dataset['dev'] = self._dev_dataset
        if self._num_test != 0:
            semi_dataset['test'] = self._test_dataset

        if return_csv:
            self.dataset_to_csv()

        self._write_txt_file()

        if len(semi_dataset) != 0:
            return semi_dataset
        else:
            print("No Data Returned!")
            return None

    def label_rearrangement(self):

        if min(self._label) == 0:
            print("Label Correct")
            return

        if self._origin_train_dataset is not None:
            self._origin_train_dataset = self._label_rearrangement(
                self._origin_train_dataset)
            self._label = self._origin_train_dataset["label"].unique()

        if self._origin_test_dataset is not None:
            self._origin_test_dataset = self._label_rearrangement(
                self._origin_test_dataset)

        if self._origin_dev_dataset is not None:
            self._origin_dev_dataset = self._label_rearrangement(
                self._origin_dev_dataset)

    def sample(self):
        # "type:ignore" means ignore the issues reported by pylance

        # sup dataset
        if self._origin_train_dataset is not None and self._num_sup != 0:
            self._sup_dataset, self._origin_train_dataset = self._sample_target_dataset(
                self._origin_train_dataset, self._num_sup, type='sup')  # type:ignore
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

    def generate_unsup_dataset(self, unsup_dataset_origin):
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
        self._unsup_dataset = pd.DataFrame.from_dict(unsup_dataset_dict)


if __name__ == "__main__":
    semi_data = SemiDataset(file_path="/home/chenqy/Dataset/IMDB",
                            num_sup=500, num_test=12500, num_dev=5000, num_unsup=5000)

    data = semi_data.get_semi_dataset()
