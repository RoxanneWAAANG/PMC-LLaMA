import transformers
import torch
# import bitsandbytes as bnb
# from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model.cuda()  # move the model to GPU

prompt_input = (
    'Below is an instruction that describes a task, paired with an input that provides further context.'
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
)

example = {
    "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
    "input": (
        "###Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
        "She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. "
        "She otherwise feels well and is followed by a doctor for her pregnancy. "
        "Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air."
        "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. "
        "Which of the following is the best treatment for this patient?"
        "###Options: A. Ampicillin B. Ceftriaxone C. Doxycycline D. Nitrofurantoin"
    )
}
input_str = [prompt_input.format_map(example)]

model_inputs = tokenizer(
            input_str,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        )
print( f"\033[32mmodel_inputs\033[0m: { model_inputs }" )

with torch.no_grad():
    generated = model.generate(
        inputs = model_inputs["input_ids"],
        # max_length=50,
        max_new_tokens=200,
        do_sample=True,
        top_k=50
    )
    print('model predict: ',tokenizer.decode(generated[0]))
