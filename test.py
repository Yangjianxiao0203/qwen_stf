from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# 载入模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/root/model/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("/root/model/Qwen1.5-0.5B")

# 载入或准备你的数据集
from datasets import load_dataset
#/root/data/AdvertiseGen
data_files = {
    "train": "/root/data/AdvertiseGen/train.json",
    "dev": "/root/data/AdvertiseGen/dev.json"
}
dataset = load_dataset('json', data_files=data_files)
train_dataset = dataset['train']
eval_dataset = dataset['dev']
print(dataset['dev'][0])

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 输出目录
    num_train_epochs=3,                  # 训练轮数
    per_device_train_batch_size=4,       # 每个设备的批量大小
    deepspeed='./ds_config.json',        # DeepSpeed 配置文件
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 开始训练
trainer.train()