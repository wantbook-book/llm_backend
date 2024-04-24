


from api.api import create_app
from model.llama import Llama
import uvicorn

if __name__ == "__main__":
    model = Llama(model_path="/home/jovyan/notebook/MineLLaMa")
    app = create_app(model)
    uvicorn.run(app, host="0.0.0.0", port=9999)