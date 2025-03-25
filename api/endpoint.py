from fastapi import FastAPI
from api.types import CommodityPriceRequestDTO

app = FastAPI()
@app.get("/commodity/prediction")
def commodity_price_prediction(commodity_price_request: CommodityPriceRequestDTO):
    return commodity_price_prediction

@app.get("/")
def api_docs_information():
    """Information for api docs endpoint"""
    return "/docs for opening the api documentation"
