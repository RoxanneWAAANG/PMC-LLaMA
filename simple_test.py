import transformers
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Define the local directory where the model is saved
save_directory = "/home/jack/Projects/yixin-llm/yixin-llm-data/PMC-LLaMA/models--axiong--PMC_LLaMA_13B/snapshots/265afcdeef43b86052a2048a862d1a493cc7dffb"

# Load the tokenizer and model from the local directory
tokenizer = transformers.LlamaTokenizer.from_pretrained(save_directory)
model = transformers.LlamaForCausalLM.from_pretrained(save_directory, torch_dtype=torch.float16)
# tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
# model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B')

model.cuda()  # move the model to GPU

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Set pad_token_id explicitly
model.generation_config.pad_token_id = tokenizer.eos_token_id


prompt_input = (
    'Below is an instruction that describes a task, paired with an input that provides further context.'
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
)

# Define Few-Shot In-Context Examples
few_shot_examples = [
    {
        "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly and provide a detailed rationale explaining why you chose this option.",
        "input": (
            "###Question: A 45-year-old male presents with chest pain that radiates to his left arm. He describes the pain as a pressure-like sensation that started 30 minutes ago. He has a history of hypertension and smokes one pack of cigarettes per day. His blood pressure is 150/95 mmHg, heart rate is 110/min, and oxygen saturation is 96% on room air. An ECG shows ST-segment elevation in leads II, III, and aVF."
            "###Options: A. Stable angina B. Myocardial infarction C. Gastroesophageal reflux disease (GERD) D. Pulmonary embolism"
        ),
        "response": (
            "The best treatment for this patient is **B. Myocardial infarction**. "
            "**Rationale:** The patient's symptoms (chest pain radiating to the left arm, pressure-like sensation), risk factors (hypertension, smoking), and ECG findings (ST-segment elevation in leads II, III, and aVF) are consistent with an acute myocardial infarction (heart attack). Immediate intervention, such as reperfusion therapy, is required to prevent further damage to the heart muscle. Stable angina (A) would not cause ST-segment elevation on ECG. GERD (C) and pulmonary embolism (D) do not explain the ECG changes or the specific nature of the chest pain."
        )
    },
    {
        "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly and provide a detailed rationale explaining why you chose this option.",
        "input": (
            "###Question: A 30-year-old female presents with recurrent episodes of heartburn and regurgitation, especially after meals. She has no alarm symptoms like weight loss or difficulty swallowing. She drinks coffee daily and eats spicy foods frequently."
            "###Options: A. Peptic ulcer disease B. Gastroesophageal reflux disease (GERD) C. Gastritis D. Esophageal cancer"
        ),
        "response": (
            "The most likely diagnosis is **B. Gastroesophageal reflux disease (GERD)**. "
            "**Rationale:** The patient's symptoms (heartburn, regurgitation) and dietary habits (coffee, spicy foods) are consistent with GERD. Peptic ulcer disease (A) typically presents with epigastric pain unrelated to meals, while gastritis (C) may cause similar symptoms but is less likely without a history of NSAID use or H. pylori infection. Esophageal cancer (D) is unlikely in the absence of alarm symptoms like weight loss or dysphagia."
        )
    },
    {
        "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly and provide a detailed rationale explaining why you chose this option.",
        "input": (
            "###Question: A 65-year-old male with a history of hypertension and diabetes presents with acute onset of slurred speech and right-sided weakness. A CT scan of the head shows no hemorrhage. What is the most appropriate immediate treatment?"
            "###Options: A. Aspirin B. Intravenous thrombolysis (tPA) C. Antihypertensive medication D. Antibiotics"
        ),
        "response": (
            "The most appropriate treatment is **B. Intravenous thrombolysis (tPA)**. "
            "**Rationale:** The patient's symptoms (slurred speech, right-sided weakness) and imaging findings (CT scan showing no hemorrhage) are consistent with an acute ischemic stroke. Intravenous thrombolysis with tPA is the standard treatment for ischemic stroke if administered within 4.5 hours of symptom onset. tPA works by dissolving the blood clot, restoring blood flow to the brain, and minimizing long-term damage. Aspirin (A) is used for secondary prevention but is not the first-line treatment in the acute phase. Antihypertensive medication (C) is not indicated unless blood pressure is extremely high, as lowering blood pressure too quickly can worsen brain perfusion. Antibiotics (D) are irrelevant in this context, as there is no evidence of infection."
        )
    }
]

# Define the current example
current_example = {
    "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly and provide a detailed rationale explaining why you chose this option.",
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

# Combine Few-Shot Examples with the Current Example
input_str = "\n\n".join(
    [prompt_input.format_map(ex) for ex in few_shot_examples] + [prompt_input.format_map(current_example)]
)

# example = {
#     "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
#     "input": (
#         "###Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
#         "She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. "
#         "She otherwise feels well and is followed by a doctor for her pregnancy. "
#         "Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air."
#         "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. "
#         "Which of the following is the best treatment for this patient?"
#         "###Options: A. Ampicillin B. Ceftriaxone C. Doxycycline D. Nitrofurantoin"
#     )
# }
# input_str = [prompt_input.format_map(example)]

# Tokenize input
model_inputs = tokenizer(
    input_str,
    return_tensors='pt',
    padding=True,
).to("cuda")  # Move inputs to GPU
# print( f"\033[32mmodel_inputs\033[0m: { model_inputs }" )

# Generate output
with torch.no_grad():
    topk_output = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1000,
        top_k=50,
        do_sample=True,
        temperature=0.7
    )

output_str = tokenizer.batch_decode(topk_output)
# output_str = tokenizer.batch_decode(topk_output, skip_special_tokens=True)
print('model predict: ', output_str[0])
