#!/bin/bash

# 定义基础模型和其他固定参数
BASE_MODEL="/root/autodl-tmp/models/qwen/Qwen2-1___5B"
BATCH_SIZE=8
TASKS="mmlu,hellaswag"  # 多个任务使用逗号分隔
DEVICE="cuda:0"
OUTPUT_DIR="../results/qwen"

# 定义 PEFT 路径前缀
PEFT_PREFIX="/root/autodl-tmp/codes/qwen_stf/output/qwen15"

# 定义 PEFT 路径数组
PEFT_SUFFIXES=(
    "final_model_r_2"
    "final_model_r_4"
    "final_model_r_8"
    "final_model_r_12"
    "final_model_r_16"
)

# 循环遍历 PEFT 路径数组并运行评估
for PEFT_SUFFIX in "${PEFT_SUFFIXES[@]}"; do
    PEFT_PATH="${PEFT_PREFIX}/${PEFT_SUFFIX}"
    echo "Running evaluation with PEFT model: $PEFT_PATH"

    export HF_ENDPOINT=https://hf-mirror.com/

    python -m lm_eval \
        --model hf \
        --model_args pretrained=$BASE_MODEL,peft=$PEFT_PATH \
        --batch_size $BATCH_SIZE \
        --tasks $TASKS \
        --device $DEVICE \
        --output_path $OUTPUT_DIR

    echo "Finished evaluation with PEFT model: $PEFT_PATH"
    echo "---------------------------------------------"
done
