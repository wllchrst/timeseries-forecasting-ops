"""Main Script for running the project
"""
from helper import EnvHelper, DataHelper
from model import CommodityHandler

class Main:
    def __init__(self):
        EnvHelper()
        DataHelper()
        self.start_commodity()
    
    def start_commodity(self):
        self.commodity_handler = CommodityHandler(True)

if __name__ == '__main__':
    Main()
