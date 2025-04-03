"""Main Script for running the project
"""
from helper import EnvHelper, DataHelper, ParameterHelper
from model import CommodityHandler
from api.handler import APIHandler

class Main:
    """Main class for running all application"""
    def __init__(self):
        self.env_helper = EnvHelper()
        DataHelper()
        ParameterHelper()
        if self.env_helper.all_available:
            self.start_commodity()
            self.start_api()
        else:
            print("Application is not going to be started, all env is not available")

    def start_commodity(self):
        """Function to start commodity related functions"""
        self.commodity_handler = CommodityHandler(False, True)

    def start_api(self):
        """Function to start API related functions"""
        self.api_handler = APIHandler()

if __name__ == '__main__':
    Main()
