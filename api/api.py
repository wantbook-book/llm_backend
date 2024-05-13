
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from entity.llama import LlamaRequest, LlamaResponse
def create_app(models)->FastAPI:
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"data": "pong!"}


    # @app.post("/llama", response_model=LlamaResponse)
    # def llama(llama_request: LlamaRequest):
    #     response = models['v1'](user_prompt=llama_request.user_prompt, system_prompt=llama_request.system_prompt)
    #     return {"status": 0, "data": response}
    
    @app.post("/llama2_70b", response_model=LlamaResponse)
    def llama(llama_request: LlamaRequest):
        response = models['llama2_70b'](user_prompt=llama_request.user_prompt, system_prompt=llama_request.system_prompt)
        return {"status": 0, "data": response}
    
    @app.post("/llama3_70b", response_model=LlamaResponse)
    def llama(llama_request: LlamaRequest):
        response = models['llama3_70b'](user_prompt=llama_request.user_prompt, system_prompt=llama_request.system_prompt)
        return {"status": 0, "data": response}
    
    @app.post("/llama3_8b", response_model=LlamaResponse)
    def llama(llama_request: LlamaRequest):
        response = models['llama3_8b'](user_prompt=llama_request.user_prompt, system_prompt=llama_request.system_prompt)
        return {"status": 0, "data": response}
    
    # @app.post("/llama2", response_model=LlamaResponse)
    # def llama(llama_request: LlamaRequest):
    #     response = models['llama2'](user_prompt=llama_request.user_prompt, system_prompt=llama_request.system_prompt)
    #     return {"status": 0, "data": response}
    
    @app.post("/llama2_13b_v3", response_model=LlamaResponse)
    def llama(llama_request: LlamaRequest):
        response = models['llama2_13b_v3'](user_prompt=llama_request.user_prompt, system_prompt=llama_request.system_prompt)
        return {"status": 0, "data": response}

    @app.post("/llama2_13b_v4", response_model=LlamaResponse)
    def llama(llama_request: LlamaRequest):
        response = models['llama2_13b_v4'](user_prompt=llama_request.user_prompt, system_prompt=llama_request.system_prompt)
        return {"status": 0, "data": response}
    return app