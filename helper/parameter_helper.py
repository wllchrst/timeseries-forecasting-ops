"""Class for collecting training paramter"""
import yaml
import os
from helper.env_helper import EnvHelper
from app_decorator import singleton
@singleton
class ParameterHelper:
    def __init__(self):
        self.env_helper = EnvHelper()
        self.collect_commodity_parameter()

    def collect_commodity_parameter(self):
        """Collect commodity parameter for training purposes"""
        self.commodity_parameters = {}
        try:
            path = self.env_helper.commodity_training_parameter_path
            for filepath in os.listdir(path):
                filename = filepath.split(".")[0]

                with open(f'{path}/{filepath}', "r") as file:
                    config = yaml.safe_load(file)

                    self.commodity_parameters[filename] = config['finetuning_parameter']
            
            print('Loaded Parameter:')
            print(self.commodity_parameters)
        except Exception as e:
            print(f'Error collecting commodity parameter: {e}')
