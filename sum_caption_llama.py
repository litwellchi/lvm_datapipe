
from transformers import LlamaForCausalLM, LlamaTokenizer
import time


access_token = "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"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token=access_token)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token=access_token)
model.to('cuda:4')
print(model.device)
prefix_prompt = """"
Please help me summarize the following setences in a list format like["","",""] of single words. Here are the captions: 
"""
captions ="a light blue background with squares in the middle of it . a blue background with squares and squares and squares with the words  it is strongly believed to enforce rape written on top of it . a person standing in front of a fence with a text ."
prompt = prefix_prompt+str(captions)
# while True:
    # caption = input("prompts: ")
# prompt =  [captions,captions,captions,captions,captions] 
# prompt = captions
s_t = time.time()
result = tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_length=300)[0])
print(f"result is : \n {result}")
print(f"using time {time.time()-s_t}s")
    # exit(0)
    
