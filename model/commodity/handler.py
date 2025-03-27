"""Handler for commodity price"""
import pandas as pd
from model.commodity.model import CommodityModel
from model.commodity.dataset import CommodityDataset
from helper import env_helper
class CommodityHandler:
    def __init__(self, testing=False):
        self.model = CommodityModel()
        self.testing = testing
        self.gather_initial_dataset()
        self.finetune_and_evaluate()

    def gather_initial_dataset(self):
        try:
            self.train_dset = CommodityDataset(env_helper.traing_dset_path)
            self.test_dset = CommodityDataset(env_helper.testing_dset_path, True)
        except Exception as e:
            print(f'Gathering initial dataset: {e}')

    def finetune_and_evaluate(self):
        try:
            self.model.finetune_model(
                dataset=self.train_dset,
                commodity_mapping=self.train_dset.commodity_mapping,
                province_mapping=self.train_dset.province_mapping,
                testing=self.testing
            )
            
            self.model.test_with_dataset(self.test_dset)
            
        except Exception as e:
            print(f'Finetuning and evaluate error: {e}')