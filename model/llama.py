
import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" #指定cuda可见显卡编号
import torch
from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,

)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import EarlyStoppingCallback

from transformers import AutoModelForCausalLM, AutoTokenizer



model_path = "/home/jovyan/notebook/MineLLaMa"  # 更改为保存模型的路径,可用范围：MineLLaMa、MineLLaMa-v3、MineLLaMa-v4

class Llama:
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False

    def __init__(self, model_path: Path, max_length:int =1024, 
                    repetition_penalty:float = 1.18, no_repeat_ngram_size:int = 5, return_full_text:bool = False, temperature: float = 0.8) -> None:
        # self.model_path = model_path
        self.compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=bnb_config,
                                            device_map='auto')  #device_map={"": 5}或device_map={"": 6}或device_map={"": 7}，使用前可调用上代码框中的!nvidia-smi查看显卡占用情况
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
                                    task="text-generation", 
                                    model=self.model, 
                                    tokenizer=self.tokenizer,
                                    max_length=max_length,
                                    repetition_penalty=repetition_penalty,
                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                    return_full_text=return_full_text,
                                    temperature=temperature
                                )
        
        

    def __call__(self, prompt: str) -> str:
        prompttext = f'[INST]<<SYS>>\nYou are a Large Language Model, and your task is to answer questions posed by users about Minecraft. Utilize your knowledge and understanding of the game to provide detailed, accurate, and helpful responses. Use your capabilities to assist users in solving problems, understanding game mechanics, and enhancing their Minecraft experience.\n<</SYS>>\n\n ' + prompt + '[/INST]' 
        return self.pipeline(prompttext)
        
    
