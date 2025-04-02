"""
Class for holding commodity model
"""
import traceback
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import Trainer
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tsfm_public import TimeSeriesPreprocessor
from model.commodity.training import fewshot_finetune_eval, predict_new_data
from api.types import CommodityPriceRequestDTO

@dataclass
class FinetuningParameter:
    """Data Class for finetuning"""
    learning_rate: float
    fewshot_percentage: int
    epochs: int
    batchsize: int
    loss: str
    context_length: int
    forecast_length: int

class CommodityModel:
    """Model for training, testing, and predicting data for commodity price
    """
    def __init__(self):
        self.trainer: Trainer = None
        self.tsp: TimeSeriesPreprocessor = None
        self.province_mapping: dict = None
        self.commodity_mapping: dict = None

    def get_model(self) -> nn.Module:
        """Return the trainer's model for looging"""
        return self.trainer.model

    def finetune_model(
        self,
        dataset: pd.DataFrame,
        province_mapping: dict,
        commodity_mapping: dict,
        parameter: FinetuningParameter,
        testing=False):
        """Finetune model using the dataset given in the commodity model.
        """
        dummy_dataset = dataset[0: 500]
        print(dataset.head())
        print(dataset.columns)
        try:
            self.trainer, self.tsp = fewshot_finetune_eval(
                training_dataset=dummy_dataset if testing else dataset,
                batch_size=parameter.batchsize,
                context_length=parameter.context_length,
                forecast_length=parameter.forecast_length,
                fewshot_percent=parameter.fewshot_percentage,
                learning_rate=parameter.learning_rate,
                num_epochs=1 if testing else parameter.epochs,
                save_dir='./save_dir'
            )
            self.province_mapping = province_mapping
            self.commodity_mapping = commodity_mapping
            print('Finetuning model success')
        except Exception as e:
            print(f'Finetuning model error: {e}')

    def test_with_dataset(self,
                    test_dataset: pd.DataFrame,
                    context_length: int,
                    forecast_length:int):
        """Test Finetune using test dataset in the parameter

        Args:
            test_dataset (pd.DataFrame): dataset that is going to predicted
        """
        if self.trainer is None or self.tsp is None:
            print("Trainer have not been trained, cannot do testing for now!")

        try:
            predictions_df, _, _ = predict_new_data(
                trainer=self.trainer,
                new_data=test_dataset,
                tsp=self.tsp,
                context_length=context_length,
                forecast_length=forecast_length,
                dataset_name="new_prediction",
                with_plot=False
            )

            print(predictions_df.head())

        except Exception as e:
            traceback.print_exc()
            print(f'Dataset testing error: {e}')

    def predict_price(self,
                commodity_price_request: CommodityPriceRequestDTO,
                context_length: int,
                forecast_length: int) -> np.float64:
        """Function to predict commodity price

        Args:
            commodity_price_request (CommodityPriceRequestDTO): commodity information

        Returns:
            np.float64: Price for that day
        """
        if self.tsp is None:
            return -1

        try:
            predict_df = self.create_df(commodity_price_request)

            results_df, _, _ = predict_new_data(
                trainer=self.trainer,
                new_data=predict_df,
                tsp=self.tsp,
                context_length=context_length,
                forecast_length=forecast_length,
                dataset_name="new_prediction",
                with_plot=False
            )

            print(results_df)
            final_result = results_df[results_df['date'] == commodity_price_request.date]
            print(final_result.head())

            return final_result['price'][0]

        except Exception as e:
            traceback.print_exc()
            print(f'Predicting price errpr: {e}')
            return -1

    def create_df(
        self,
        commodity_price_request: CommodityPriceRequestDTO,
    ) -> pd.DataFrame:
        """Create dataset for single commodity price request

        Args:
            commodity_price_request (CommodityPriceRequestDTO): data that is going to be turned to df

        Returns:
            pd.DataFrame: final result ready for predicting
        """
        data_dict = {
            'Date': [pd.to_datetime(commodity_price_request.date)],
            'commodity': [self.commodity_mapping[commodity_price_request.commodity]],
            'province': [self.province_mapping[commodity_price_request.province]],
            'GlobalOpen': [commodity_price_request.global_open],
            'GlobalHigh': [commodity_price_request.global_high],
            'GlobalLow': [commodity_price_request.global_low],
            'GlobalVol.': [commodity_price_request.global_volume],
            'GlobalChange %': [commodity_price_request.global_change_percent],
            'GlobalPrice': [commodity_price_request.global_price],
            'CE_Close': [commodity_price_request.ce_close],
            'CE_High': [commodity_price_request.ce_high],
            'CE_Low': [commodity_price_request.ce_low],
            'CE_Open': [commodity_price_request.ce_open],
            'GTPrice': [commodity_price_request.gt_price],
            'price': [0],
        }

        df = pd.DataFrame(data_dict)
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        return df
