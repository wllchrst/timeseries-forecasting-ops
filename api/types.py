import numpy as np
from pydantic import BaseModel

class CommodityPriceRequestDTO(BaseModel):
    """Class for commodity price prediction"""
    date: np.datetime64
    commodity: np.int64
    province: np.int64
    global_open: np.float64
    global_high: np.float64
    global_low: np.float64
    global_volume: np.float64
    global_change_percent: np.float64
    global_price: np.float64
    ce_close: np.float64
    ce_high: np.float64
    ce_low: np.float64
    ce_open: np.float64
    gt_price: np.float64
    timestamp: np.int64
