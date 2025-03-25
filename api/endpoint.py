from fastapi import FastAPI

app = FastAPI()
@app.get("/commodity/prediction")
def commodity_price_prediction():
    return "hello"

@app.get("/")
def api_docs_information():
    return "/docs for opening the api documentation"