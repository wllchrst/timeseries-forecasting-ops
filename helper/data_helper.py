"""Data helper python file to hold class DataHelper"""
import pandas as pd
from helper import EnvHelper
from app_decorator.singleton import singleton
@singleton
class DataHelper:
    """Class for getting initial datasett for training and testing """
    def __init__(self):
        self.env_helper = EnvHelper()
        self.gather_dataset()

    def gather_dataset(self):
        try:
            self.commodity_initial_training = pd.read_csv(self.env_helper.training_dset_path)
            self.commodity_initial_testing = pd.read_csv(self.env_helper.testing_dset_path)
        except Exception as e:
            print(f'Gathering dataset error: {e}')
