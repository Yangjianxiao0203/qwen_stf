import os
import requests
from modelscope.hub.snapshot_download import snapshot_download

def download_meta_llama_3_8b():
    # Define the model ID and target directory
    model_id = "LLM-Research/Meta-Llama-3-8B"
    target_dir = "/root/autodl-tmp/models/Meta-Llama-3-8B"

    # Use ModelScope's snapshot_download function to download the model
    snapshot_download(model_id, cache_dir=target_dir)

    print(f"Model {model_id} has been downloaded to {target_dir}")

def modelscope_download_model(model_name,target_dir="/root"):
    # cache_dir = os.path.join(target_dir, model_name)
    cache_dir = target_dir
    resp = snapshot_download(model_name, cache_dir=cache_dir)
    print(f"Model {model_name} has been downloaded to {resp}")
    return resp

if __name__ == "__main__":
    # download_meta_llama_3_8b()
    target_dir = "/root/autodl-tmp/models"
    cache_file = modelscope_download_model("qwen/Qwen2-1.5B",target_dir)
    print(cache_file)
