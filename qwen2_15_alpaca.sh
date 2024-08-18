#!/bin/bash

# 定义 LoRA 参数列表
loras=(4 2 8 12 16 32 64)

# 遍历每个 lora_num 并运行 Python 脚本
for lora_num in "${loras[@]}"
do
    echo "current processing r=$lora_num"
    # 运行 Python 脚本并传递 lora_num 作为参数
    python qwen2_15_sft_lora.py --lora_num $lora_num

    # 检查是否上一个脚本执行成功
    if [ $? -ne 0 ]; then
        echo "Training failed for r=$lora_num. Exiting."
        exit 1
    fi

    echo "Finished processing r=$lora_num"
done

echo "All trainings completed successfully."
