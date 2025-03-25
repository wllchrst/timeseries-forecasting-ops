from pydantic import BaseModel

class CommodityPriceRequestDTO(BaseModel):
    data: float
