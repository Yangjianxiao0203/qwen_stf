from datasets import load_dataset
from accelerate import Accelerator

accelerator = Accelerator()

data_dir = "tatsu-lab/alpaca"
dataset = load_dataset(data_dir)

dataset = dataset["train"].select(range(10000))

# for i in dataset:
#     print(i)
#     break


def process(single_data,tokenizer,max_length=128):
    '''
    数据处理流程：
    1. 把prompt（source） 和 输出 output（target）拼接在一起，作为同一个input ids
    2. labels复制这个input ids，然后把prompt部分的input ids设置为-100，output的部分还是保持原来的
    3.设置attention masks，input ids中把padding的部分都设置为0，其他都是1
    '''

    MAX_LENGTH = max_length
    prompt_template = """
    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}
    
    {input_formatter}
    
    ### Response:
    """
    input_formatter = ""
    if single_data["input"] and len(single_data["input"]) > 0:
        input_formatter = f"###Input:\n{single_data['input']}"

    prompt = prompt_template.format(instruction=single_data["instruction"], input_formatter=input_formatter)
    tokenized_prompt = tokenizer(prompt, add_special_tokens=False)
    tokenized_response = tokenizer(single_data["output"], add_special_tokens=False)

    #注意： attention mask代表需要注意力机制看到的部分，所以eos token也是需要关注的，所以attention mask补充为1
    input_ids = tokenized_prompt["input_ids"] + tokenized_response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = tokenized_prompt["attention_mask"] + tokenized_response["attention_mask"] + [1]
    labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    else:
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length

    assert len(input_ids) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    assert len(attention_mask) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
    assert len(labels) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("C:\jianxiao_codes\python\models\qwen/Qwen2-0.5B\qwen\Qwen2-0___5B")
    for i in dataset:
        print(process(i,tokenizer))
        break
