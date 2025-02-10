import transformers
import torch
# import bitsandbytes as bnb
# from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
# model = transformers.LlamaForCausalLM.from_pretrained(
#     "chaoyi-wu/PMC_LLAMA_7B",
#     device_map="auto",  # 自动分配到 CPU 或 GPU/device_map="cpu",
#     torch_dtype=torch.float16,  # 16-bit 计算，减少内存占用
#     load_in_4bit=True  # 4-bit 量化（如果 4-bit 不行可以换成 8-bit）
# )
sentence = 'Hello, doctor' 
batch = tokenizer(
            sentence,
            return_tensors="pt", 
            add_special_tokens=False
        )
with torch.no_grad():
    generated = model.generate(inputs = batch["input_ids"], max_length=50, do_sample=True, top_k=50)
    print('model predict: ',tokenizer.decode(generated[0]))
