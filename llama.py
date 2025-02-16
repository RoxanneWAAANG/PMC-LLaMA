from huggingface_hub import snapshot_download

save_directory = "/home/jack/Projects/yixin-llm/yixin-llm-data/PMC-LLaMA"

model_path = snapshot_download(repo_id="axiong/PMC_LLaMA_13B", cache_dir=save_directory)
print(f"Model downloaded to: {model_path}")
# Model downloaded to:
# /Users/ruoxinwang/.cache/huggingface/hub/models--chaoyi-wu--PMC_LLAMA_7B/snapshots/6caf5c19bdcd157f9d9a7d374be66d7b61d75351
# C:\Users\admin\.cache\huggingface\hub\models--chaoyi-wu--PMC_LLAMA_7B\snapshots\6caf5c19bdcd157f9d9a7d374be66d7b61d75351
# Model downloaded to: /home/jack/Projects/yixin-llm/yixin-llm-data/PMC-LLaMA/models--axiong--PMC_LLaMA_13B/snapshots/265afcdeef43b86052a2048a862d1a493cc7dffb



