---
license: apache-2.0
tags:
- medical
datasets:
- allenai/s2orc
---
This repo contains PMC_LLaMA_7B, which is LLaMA-7b finetuned on the PMC papers in S2ORC dataset.

The model was trained with the following hyperparameters:

* Epochs: 5 
* Batch size: 128 
* Cutoff length: 512
* Learning rate: 2e-5

Each epoch we sample 512 tokens per paper for training.

The model can be loaded as following:

```
import transformers
import torch
tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
sentence = 'Hello, doctor' 
batch = tokenizer(
            sentence,
            return_tensors="pt", 
            add_special_tokens=False
        )
with torch.no_grad():
    generated = model.generate(inputs = batch["input_ids"], max_length=200, do_sample=True, top_k=50)
    print('model predict: ',tokenizer.decode(generated[0]))
```