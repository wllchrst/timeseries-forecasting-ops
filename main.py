"""Main Script for running the project
"""
import uvicorn
from api import app
from helper import env_helper

"""Commodity"""
from model.commodity import commodity_model, CommodityDataset

if __name__ == '__main__':
    # uvicorn.run(app, host=env_helper.api_host)
    commodity_dataset = CommodityDataset('./dataset/commodity/training.csv')
    commodity_model.finetune_model(dataset=commodity_dataset.dataset)