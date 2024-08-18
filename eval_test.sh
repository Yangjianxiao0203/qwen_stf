#!/bin/bash
cd /root/autodl-tmp/codes/qwen_stf/lm-evaluation-harness
export HF_ENDPOINT=https://hf-mirror.com/

# 运行评估
python -m lm_eval \
    --model hf \
    --model_args pretrained=/root/autodl-tmp/models/qwen/Qwen2-1___5B,peft=/root/autodl-tmp/codes/qwen_stf/output/qwen15/final_model_r_8 \
    --batch_size 8 \
    --tasks mmlu\
    --device cuda:0 \
    --output_path ../results

cd ..
