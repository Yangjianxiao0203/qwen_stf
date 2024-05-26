import json
import logging
import os

import torch
import torch.optim as optim
from datasets import load_metric
from torch.optim.lr_scheduler import LambdaLR

from model_pipeline.Lora import LoRA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    '''
    只适用于单卡，多卡直接用deepspeed，到时再封装一个壳
    此类目前适用于分类任务
    '''

    def __init__(self, model, data_loaders, config: dict, log = False, debug = False):
        self.model = model
        self.data_loaders = data_loaders
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.log= log
        self.debug= debug

        # self.task_type = config['task_type']
        # assert self.task_type in ['classification', 'seq2seq'], "not supported task type"

        # 设置优化器
        optimizer_config = self.config['optimizer']
        optimizer_class = getattr(optim, optimizer_config['type'])
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_config['params']
        )

        # 设置学习率调度器
        scheduler_config = self.config['scheduler']
        self.scheduler = self._create_scheduler(self.optimizer, scheduler_config) if scheduler_config else None

    def _create_scheduler(self, optimizer, scheduler_config):
        scheduler_type = scheduler_config['type']
        if scheduler_type == 'WarmupLR':
            def lr_lambda(current_step: int):
                warmup_min_lr = scheduler_config['params']['warmup_min_lr']
                warmup_max_lr = scheduler_config['params']['warmup_max_lr']
                warmup_num_steps = scheduler_config['params']['warmup_num_steps']
                if current_step < warmup_num_steps:
                    return float(current_step) / float(max(1, warmup_num_steps))
                return warmup_max_lr

            return LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def train(self):
        num_epochs = self.config['epochs']
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch in self.data_loaders["train_dataloader"]:
                self.optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(inputs)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()

                if self.debug:
                    break

            epoch_loss = running_loss / len(self.data_loaders["train_dataloader"])
            if self.debug:
                epoch_loss = running_loss
            val_accuracy = self.evaluate()
            self._log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # torch.save(self.model.state_dict(), self.config['save_path'])
                self._log(f'Saved Best Model with Accuracy: {best_accuracy:.4f}')

    def evaluate(self):
        # 根据task type计算不同的损失，task_type = classification / seq2seq
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.data_loaders["val_dataloader"]:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(inputs)
                # logits代表模型直接的输出
                # 分类任务是 batch x num_labels
                # 生成任务是 batch x seq_len x vocab_size 代表每个位置的概率
                _, predicted = torch.max(outputs.logits, 1)
                total += inputs['labels'].size(0)
                correct += (predicted == inputs['labels']).sum().item()

                if self.debug:
                    break

        accuracy = correct / total
        return accuracy

    def save_checkpoint(self, epoch, metric):
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f"checkpoint_epoch_{epoch}_metric_{metric:.4f}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metric': metric
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metric']

    def _log(self,message):
        if self.log:
            logger.info(message)
        else:
            print(message)


class Seq2SeqTrainer(Trainer):
    #TODO: 模块待测试
    def __init__(self, model, data_loaders, config, tokenizer, log=False, debug=False):
        super().__init__(model, data_loaders, config, log, debug)
        self.bleu_metric = load_metric("sacrebleu")
        self.tokenizer = tokenizer

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.data_loaders["val_dataloader"]:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)

                all_preds.extend(decoded_preds)
                all_labels.extend([[label] for label in decoded_labels])

                if self.debug:
                    break

        bleu_score = self.bleu_metric.compute(predictions=all_preds, references=all_labels)
        return bleu_score['score']


if __name__ == '__main__':
    from transformers import BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification
    from data_pipeline.glue_qnli import auto_get_glue_qnli_loaders

    train_loader, val_loader, test_loader = auto_get_glue_qnli_loaders()
    data_loaders = {
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "test_dataloader": test_loader
    }
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    # print("Original Number of parameters: {}".format(count_trainable_parameters(model)))
    lora_model = LoRA(
        model=model,
        r=8,
        lora_alpha=16,
        target_modules=["query", "key"],
        lora_dropout=0.1,
        bias="none"
    )

    with open('./train_config.json', 'r') as f:
        config = json.load(f)

    trainer = Trainer(model=lora_model, data_loaders=data_loaders, config=config,debug=True)
    trainer.train()
