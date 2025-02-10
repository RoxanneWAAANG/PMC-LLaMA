from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="chaoyi-wu/PMC_LLAMA_7B")
print(f"Model downloaded to: {model_path}")
# Model downloaded to:
# /Users/ruoxinwang/.cache/huggingface/hub/models--chaoyi-wu--PMC_LLAMA_7B/snapshots/6caf5c19bdcd157f9d9a7d374be66d7b61d75351
# C:\Users\admin\.cache\huggingface\hub\models--chaoyi-wu--PMC_LLAMA_7B\snapshots\6caf5c19bdcd157f9d9a7d374be66d7b61d75351