import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification


def create_lora_forward(original_forward, lora_layer, scaling):
    def modified_forward(*args,**kwargs):
        input_tensor = args[0]
        lora_output = lora_layer(input_tensor) * scaling
        output = original_forward(*args,**kwargs)
        output += lora_output
        return output

    return modified_forward


class LoRA(nn.Module):
    def __init__(self, model, r, lora_alpha, target_modules, lora_dropout, bias="none"):
        super(LoRA, self).__init__()
        self.model = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.scaling = self.lora_alpha / self.r
        self.lora_layers = nn.ModuleDict()
        self.lora_name_mapper = {}
        self._freeze_original_layers()
        self._initialize_lora_layers()

    def _freeze_original_layers(self):
        # 先冻结原来所有参数，然后只训练lora层
        for param in self.model.parameters():
            param.requires_grad = False

    def _initialize_lora_layers(self):
        for name, module in self.model.named_modules():
            for target_name in self.target_modules:
                if target_name in name:
                    lora_name = self._replace_module_name(name)
                    lora_layer = self._create_lora_layer(module)
                    self.lora_layers[lora_name] = lora_layer
                    self.lora_name_mapper["name"] = lora_name
                    module.forward = create_lora_forward(module.forward,lora_layer,self.scaling)
            # if any(target in name for target in self.target_modules):
            #     self.lora_layers[name] = self._create_lora_layer(module)

    def _replace_module_name(self, module_name):
        return module_name.replace(".", "_")

    def _create_lora_layer(self, module):
        #  W0 (a,b) -> W0 + dW
        in_features = module.weight.size(1)
        out_features = module.weight.size(0)

        lora_layer = nn.Sequential(
            # 解决歧义：nn.Linear(in_features, out_features)的权重矩阵形状为 [out_features, in_features]，bias形状为 [1，out_features]
            # 输入为[batch_size, in_features]
            nn.Linear(in_features, self.r, bias=self.bias == "all"),
            nn.ReLU(),
            nn.Linear(self.r, out_features, bias=self.bias == "all")
        )

        if self.lora_dropout > 0:
            lora_layer.add_module('dropout', nn.Dropout(self.lora_dropout))

        return lora_layer

    def forward(self, inputs):
        #TODO: 这个forward还有问题，等会看看
        if isinstance(inputs, dict):
            # 输入可能是一个字典，包括input_ids, attention_mask, labels 有的模型可能没有labels
            # inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            return self.model(**inputs)
        else:
            return self.model(inputs)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from data_pipeline.glue_qnli import auto_get_glue_qnli_loaders
    from transformers import AdamW, get_linear_schedule_with_warmup

    train_loader, _, test_loader = auto_get_glue_qnli_loaders()
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    #print("Original Number of parameters: {}".format(count_trainable_parameters(model)))
    lora_model = LoRA(
        model=model,
        r=8,
        lora_alpha=16,
        target_modules=["encoder.layer.0.attention.self.query", "encoder.layer.0.attention.self.key"],
        lora_dropout=0.1,
        bias="none"
    )
    optimizer = AdamW(lora_model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lora_model.to(device)
    #print("lora number of parameters: {}".format(count_trainable_parameters(lora_model)))

    for epoch in range(3):
        lora_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            # output = model(**inputs)
            # print(output)
            # break
            outputs = lora_model(inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
