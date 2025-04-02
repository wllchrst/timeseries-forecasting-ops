"""Class for handling all API related"""
from fastapi import FastAPI, APIRouter
from api.types import CommodityPriceRequestDTO
from model import CommodityHandler

app = FastAPI()
router = APIRouter()
commodity_handler = CommodityHandler(testing=False, finetuned=True)

class APIRouterHandler:
    """Class for handling all router"""
    @router.post("/commodity/prediction")
    async def commodity_price_prediction(self, commodity_price_request: CommodityPriceRequestDTO):
        """API for getting commodity data and predicting price
        
        Input Example:
        {
            "date": "2024-10-01",
            "commodity": "bawang merah",
            "province": "Aceh",
            "global_open": 118.97541666666666,
            "global_high": 602.5,
            "global_low": 3.253,
            "global_volume": 686450.0,
            "global_change_percent": 1.0614285714285714,
            "global_price": 121.82983333333334,
            "ce_close": 3792.9128202488646,
            "ce_high": 15250.099609375,
            "ce_low": 0.0306372549384832,
            "ce_open": 3792.9128202488646,
            "gt_price": 82.66666666666667
        }
        """
        result = commodity_handler.predict_price(commodity_price_request)
        return result

    @router.get("/")
    async def api_docs_information(self):
        """Information for API docs endpoint"""
        return {"message": "/docs for opening the API documentation"}

# Instantiate the class to bind the routes
commodity_api = APIRouterHandler()
app.include_router(router)
