from fastapi import FastAPI

app = FastAPI()
@app.get("/commodity/prediction")
def commodity_price_prediction():
    return "hello"
