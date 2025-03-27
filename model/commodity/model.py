"""
Class for holding commodity model
"""
import pandas as pd
import numpy as np
from transformers import Trainer
from tsfm_public import TimeSeriesPreprocessor
from model.commodity.training import fewshot_finetune_eval, predict_new_data
from api.types import CommodityPriceRequestDTO
class CommodityModel:
    """Model for training, testing, and predicting data for commodity price
    """
    def __init__(self):
        self.trainer: Trainer = None
        self.tsp: TimeSeriesPreprocessor = None
        self.context_length = 512
        self.prediction_length = 96

    def finetune_model(self, dataset: pd.DataFrame, testing=False):
        """Finetune model using the dataset given in the commodity model.
        """
        try:
            self.trainer, self.tsp = fewshot_finetune_eval(
                training_dataset=dataset,
                batch_size=32,
                context_length=self.context_length,
                forecast_length=self.prediction_length,
                fewshot_percent=30,
                learning_rate=0.001,
                num_epochs=20 if not testing else 1,
                save_dir='./save_dir'
            )
            print('Finetuning model success')
        except Exception as e:
            print(f'Finetuning model error: {e}')

    def test_with_dataset(self, test_dataset: pd.DataFrame):
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
                context_length=self.context_length,
                forecast_length=self.prediction_length,
                dataset_name="new_prediction",
                with_plot=False
            )

            print(predictions_df.head())
        except Exception as e:
            print(f'Dataset testing error: {e}')

    def predict_price(self, commodity_price_request: CommodityPriceRequestDTO) -> np.float64:
        """Function to predict commodity price

        Args:
            commodity_price_request (CommodityPriceRequestDTO): commodity information

        Returns:
            np.float64: Price for that day
        """
        pass
