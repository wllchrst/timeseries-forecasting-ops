"""Class for commodity dataset"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model.commodity.constant import feature_to_scale
class CommodityDataset:
    def __init__(self, dataset_path: str, testing_dataset: bool=False):
        self.dataset_path = dataset_path
        self.dataset: pd.DataFrame = pd.read_csv(self.dataset_path)
        self.labels: list[str] = []
        self.province_encoder = LabelEncoder()
        self.commodity_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.testing_dataset = testing_dataset
        self.process_dataset()
        
    def process_dataset(self) -> bool:
        """
        Process dataset with encoder and scaling
        """
        try:
            self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
            self.dataset['timestamp'] = self.dataset['Date'].astype('int64') // 10**9
            self.dataset['province'] = self.province_encoder.fit_transform(self.dataset['province'])
            self.dataset['commodity'] = self.commodity_encoder.fit_transform(self.dataset['commodity'])

            self.dataset = self.dataset.drop(columns=['Unnamed: 0'])

            for col in feature_to_scale:
                self.dataset[col] = self.scaler.fit_transform(self.dataset[[col]])

            self.province_mapping = {idx: label for idx, label in enumerate(self.province_encoder.classes_)}
            self.commodity_mapping = {idx: label for idx, label in enumerate(self.commodity_encoder.classes_)}

            if self.testing_dataset:
                self.dataset['price'] = 0
        except Exception as e:
            print(f'Error processing dataset: {e}')
