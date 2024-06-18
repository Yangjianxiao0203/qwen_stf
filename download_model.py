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

if __name__ == "__main__":
    download_meta_llama_3_8b()
