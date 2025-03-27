from pydantic import BaseModel

class CommodityPriceRequestDTO(BaseModel):
    """Class for commodity price prediction"""
    date: str  # Use str for date or datetime.datetime if needed
    commodity: str
    province: str
    global_open: float
    global_high: float
    global_low: float
    global_volume: float
    global_change_percent: float
    global_price: float
    ce_close: float
    ce_high: float
    ce_low: float
    ce_open: float
    gt_price: float
