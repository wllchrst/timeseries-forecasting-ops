from pydantic import BaseModel

class CommodityPriceRequestDTO(BaseModel):
    """Class for commodity price prediction
    
    commodity_price_request = CommodityPriceRequestDTO(
        date='2024-10-01',
        commodity='bawang merah',
        province='Aceh',
        global_open=118.97541666666666,
        global_high=602.5,
        global_low=3.253,
        global_volume=686450.0,
        global_change_percent=1.0614285714285714,
        global_price=121.82983333333334,
        ce_close=3792.9128202488646,
        ce_high=15250.099609375,
        ce_low=0.0306372549384832,
        ce_open=3792.9128202488646,
        gt_price=82.66666666666667,
    )
    
    """
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
