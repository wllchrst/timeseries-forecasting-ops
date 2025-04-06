"""Handler for commodity price"""
import torch
from logger.commodity_logger import CommodityLogger
from model.commodity.model import CommodityModel, FinetuningParameter
from model.commodity.dataset import CommodityDataset
from model.commodity.constant import DEFAULT_PARAMETER
from api.types import CommodityPriceRequestDTO
from helper import DataHelper, ParameterHelper
from app_decorator import singleton
@singleton
class CommodityHandler:
    def __init__(self, testing=False, finetuned=True):
        self.data_helper = DataHelper()
        self.parameter_helper = ParameterHelper()
        self.model = CommodityModel()
        self.testing = testing
        self.parameter = None
        self.logger = CommodityLogger()
        if finetuned:
            self.finetune_and_evaluate()

    def get_input_example(self):
        df_sample = self.train_dset.dataset.dataframe.iloc[0]

        example_input = torch.tensor(df_sample.values, dtype=torch.float32)

        example_input = example_input.unsqueeze(0)

        return example_input

    def finetune_and_evaluate(self, use_default=True):
        print("Finetuning and evaluating")
        try:
            finetuning_parameter = None
            if use_default:
                parameter = self.parameter_helper.\
                    commodity_parameters[DEFAULT_PARAMETER]
                finetuning_parameter = FinetuningParameter(**parameter)
                self.parameter = finetuning_parameter

            self.train_dset = CommodityDataset(self.data_helper.commodity_initial_training)
            self.test_dset = CommodityDataset(self.data_helper.commodity_initial_testing, True)

            self.model.finetune_model(
                dataset=self.train_dset.dataset,
                commodity_mapping=self.train_dset.commodity_mapping,
                province_mapping=self.train_dset.province_mapping,
                testing=self.testing,
                parameter=finetuning_parameter
            )

            self.model.test_with_dataset(self.test_dset.dataset,
                                        finetuning_parameter.context_length,
                                        finetuning_parameter.forecast_length)

            input_example = self.get_input_example()
            self.logger.log_finetuning(parameter, self.model.get_model(), input_example)
        except Exception as e:
            print(f'Finetuning and evaluate error: {e}')

    def predict_price(self, commodity_request_dto: CommodityPriceRequestDTO) -> float:
        prediction = self.model.predict_price(
            commodity_request_dto,
            context_length=self.parameter.context_length,
            forecast_length=self.parameter.forecast_length)
        return float(prediction)
