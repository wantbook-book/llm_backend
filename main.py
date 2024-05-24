


from api.api import create_app
from model.llama import Llama2, Llama3
import uvicorn
import os
from conf import config_manager
os.environ["CUDA_VISIBLE_DEVICES"] = config_manager.get('CUDA_VISIBLE_DEVICES') #指定cuda可见显卡编号
if __name__ == "__main__":
    models_path = config_manager.get('models')
    models = {}
    for name in models_path:
        if name.startswith('llama3'):
            models[name] = Llama3(model_path=models_path[name], device_map="auto")
        elif name.startswith('llama2'):
            models[name] = Llama2(model_path=models_path[name], device_map="auto")
    # models = {
    #     name: Llama(model_path=models_path[name], device_map="auto")
    #     for name in models_path
    # }
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" #指定cuda可见显卡编号
    # models['v1'] = Llama(model_path="/home/jovyan/notebook/MineLLaMa", device_map={"":5})
    # models['llama3_8b'] = Llama(model_path="/home/jovyan/notebook/LLAMA-3-8B", device_map="auto")
    # models['llama2'] = Llama(model_path="/home/jovyan/notebook/LLAMA_13B", device_map="auto")
    # models['llama3_70b'] = Llama(model_path="/pubshare/LLM/Meta-Llama-3-70B-Instruct", device_map="auto")
    # models['llama2_13b_v3'] = Llama(model_path="/home/jovyan/notebook/MineLLaMa-v3", device_map="auto")
    # models['llama2_13b_v4'] = Llama(model_path="/home/jovyan/notebook/MineLLaMa-v4", device_map="auto")
    # models['llama2_70b'] = Llama(model_path="/pubshare/LLM/Llama-2-70b-chat-hf", device_map="auto")
    
    app = create_app(models)
    uvicorn.run(app, host="0.0.0.0", port=config_manager.get('port'))