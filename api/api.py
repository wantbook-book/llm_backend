
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from entity.llama import LlamaRequest, LlamaResponse
def create_app(model)->FastAPI:
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"data": "pong!"}


    @app.post("/llama", response_model=LlamaResponse)
    def llama(llama_request: LlamaRequest):
        response = model(llama_request.prompt)
        return {"status": 0, "data": response}
