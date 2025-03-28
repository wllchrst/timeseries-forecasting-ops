"""Main Script for running the project
"""
from helper import EnvHelper, DataHelper
from model import CommodityHandler
from api import APIHandler

class Main:
    def __init__(self):
        EnvHelper()
        DataHelper()
        # self.start_commodity()
        self.start_api()
    
    def start_commodity(self):
        self.commodity_handler = CommodityHandler(True)
    
    def start_api(self):
        self.api_handler = APIHandler()

if __name__ == '__main__':
    Main()
