import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from . import yield_data_file



class PretrainingDataModule(pl.LightningDataModule):
    def __init__(self,
                 model_name_or_path: str='',
                 max_seq_length: int = -1,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 data_dir: str = '',
                 num_workers: int = 1,
                ):


        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length if max_seq_length > 0 else 'longest'
        self.train_batch_size = train_batch_size
        self.eval_batch_size  = eval_batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def load_dataset(self):
        data_files = {
            'train': list(yield_data_file(self.data_dir))
        }
        self.raw_datasets = load_dataset('text', data_files=data_files)

    def prepare_dataset(self):
        def tokenize_function(examples):
            examples['text'] = [line for line in examples['text'] if len(line) > 0 and not line.isspace()]

            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=True
            )

        processed_datasets = self.raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            load_from_cache_file=True,
            num_proc=32
        )

        print(processed_datasets)

        processed_datasets = processed_datasets['train'].train_test_split(
            test_size=102_400, seed=42, train_size=40_960_000
        )
        # 112 587 717

        print(processed_datasets)

        self.train_dataset = processed_datasets['train']
        self.eval_dataset  = processed_datasets['test']

    def get_dataloader(self, mode, batch_size, shuffle):
        dataset = self.train_dataset if mode == 'train' else self.eval_dataset
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=self.tokenizer)
        )

        print(mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    # def test_dataloader(self):
    #     return self.get_dataloader("test", self.eval_batch_size, shuffle=False)

