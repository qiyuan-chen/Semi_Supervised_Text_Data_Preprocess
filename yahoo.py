from dataset_generator import ImbalanceSemiDatasetsGenerator
import pandas as pd
import os


class YahooGenerator(ImbalanceSemiDatasetsGenerator):
    def __init__(self, file_path, N_max_sups=[], N_max_unsups=[], num_dev=0,
                 num_test=0, sup_ratios=[], unsup_ratios=[], seed=666):
        ImbalanceSemiDatasetsGenerator.__init__(self, file_path, N_max_sups, N_max_unsups, num_dev,
                                                num_test, sup_ratios, unsup_ratios, seed)

    def _read_dataset(self, type='train'):
        file_name = type+'.csv'
        file_path = os.path.join(self._file_path, file_name)
        if os.path.exists(file_path):
            dataset = pd.read_csv(file_path, header=None,
                                  names=["label", "sentence", "content", "answer"])
            print("Read original {} dataset successful from {}".format(
                type, file_path))
            return dataset
        else:
            print("ERROR! Read original {} dataset unsuccessful!".format(type))
            return None



semi_data = YahooGenerator(file_path="/home/chenqy/Dataset/yahoo", N_max_sups=[500, 2500],
                           num_test=6000, num_dev=5000, N_max_unsups=[5000],
                           sup_ratios=[10, 100], unsup_ratios=[10, 100])

semi_data.combine_columns(
    ["sentence", "content", "answer"], label_colum="label")
semi_data.generate_datasets()
