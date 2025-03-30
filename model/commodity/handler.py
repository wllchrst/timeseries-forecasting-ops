"""Handler for commodity price"""
import pandas as pd
from model.commodity.model import CommodityModel
from model.commodity.dataset import CommodityDataset
from api.types import CommodityPriceRequestDTO
from helper import DataHelper
from app_decorator import singleton
@singleton
class CommodityHandler:
    def __init__(self, testing=False):
        self.data_helper = DataHelper()
        self.model = CommodityModel()
        self.testing = testing
        self.finetune_and_evaluate()

    def finetune_and_evaluate(self):
        try:
            self.train_dset = CommodityDataset(self.data_helper.commodity_initial_training)
            self.test_dset = CommodityDataset(self.data_helper.commodity_initial_testing, True)

            self.model.finetune_model(
                dataset=self.train_dset.dataset,
                commodity_mapping=self.train_dset.commodity_mapping,
                province_mapping=self.train_dset.province_mapping,
                testing=self.testing
            )

            self.model.test_with_dataset(self.test_dset.dataset)

        except Exception as e:
            print(f'Finetuning and evaluate error: {e}')

    def predict_price(self, commodity_request_dto: CommodityPriceRequestDTO) -> float:
        prediction = self.model.predict_price(commodity_request_dto)
        return float(prediction)
