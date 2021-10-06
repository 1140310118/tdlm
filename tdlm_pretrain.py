import os
import torch
import logging
import argparse
import random
import numpy as np 


import pytorch_lightning as pl 

pl.seed_everything(42)

from transformers import AutoTokenizer
from transformers.optimization import Adafactor, AdamW
from pytorch_lightning.utilities import rank_zero_info
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

arg_to_scheduler = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_w_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule_with_warmup,
}


from model.modeling_tdlm import TDLMForPretraining, TDLMConfig
from utils.pretrain_datamodule import PretrainingDataModule
from utils import params_count


logger = logging.getLogger(__name__)


class Pretraining(pl.LightningModule):
    def __init__(self, hparams, tokenizer):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.tokenizer = tokenizer

        if self.hparams.model_name_or_path == 'bert-base-uncased':
        
            self.config = TDLMConfig(
                vocab_size=len(self.tokenizer),
                embedding_size=128,
                hidden_size=512,
                num_attention_heads=8,
                num_hidden_layers=8,
                pad_token_id=self.tokenizer.pad_token_id,
                position_embedding_type=self.hparams.position_embedding_type,
                encoder_layer=self.hparams.encoder_layer,
                pre_norm=True
            )

            self.model = TDLMForPretraining(self.config)

        else:
            self.config = TDLMConfig.from_pretrained(self.hparams.model_name_or_path)
            self.model = TDLMForPretraining.from_pretrained(self.hparams.model_name_or_path)

        print(self.config)
        print('---------------------------------------')
        print('total params_count:', params_count(self.model))
        print('tdlm  params_count:', params_count(self.model.tdlm))
        print('---------------------------------------')

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = os.path.join(self.hparams.output_dir, 'model', f'step={self.global_step+1}')

        print(f'## save model to {dir_name}')
        self.model.tdlm.save_pretrained(dir_name)
        self.tokenizer.save_pretrained(dir_name)

    def load_model(self):
        dir_name = os.path.join(self.hparams.output_dir, 'model')
        print(f'## load model to {dir_name}')
        self.model = TDLMForPretraining.from_pretrained(dir_name, config=self.config)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs
        
    def training_step(self, batch, batch_idx):
        # inputs = batch.pretrain_input()
        inputs = batch
        loss = self(**inputs)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # inputs = batch.pretrain_input()
        inputs = batch
        loss = self(**inputs)

        self.log('valid_loss', loss)
        
    def validation_epoch_end(self, outputs):
        dir_name = os.path.join(self.hparams.output_dir, 'model')
        self.save_model()

    def setup(self, stage):
        if stage == 'fit':
            self.total_steps = self.hparams.max_steps # // self.hparams.accumulate_grad_batches

    def get_lr_scheduler(self):
        get_scheduler_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == 'constant':
            scheduler = get_scheduler_func(self.opt, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = get_scheduler_func(self.opt, num_warmup_steps=self.hparams.warmup_steps, 
                                           num_training_steps=self.total_steps)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return scheduler

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
    
        def has_keywords(n, keywords):
            return any(nd in n for nd in keywords)
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': 0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = (
            Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False)
            if self.hparams.adafactor else
            AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--lr_scheduler", type=str)
        parser.add_argument("--adafactor", action='store_true')
        
        parser.add_argument("--position_embedding_type", type=str, default='learnable')
        parser.add_argument("--encoder_layer", type=str, default='transformer')

        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--output_dir", type=str)

        return parser


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # metrics = {k:(v.item() if type(v) is torch.Tensor else v) for k,v in metrics.items()}
        metrics = {k:(v.detach() if type(v) is torch.Tensor else v) for k,v in metrics.items()}
        rank_zero_info(metrics)

    # def on_train_end(self, trainier, pl_module):
    #     torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Pretraining.add_model_specific_args(parser)
    parser = PretrainingDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    
    data_module = PretrainingDataModule.from_argparse_args(args)
    data_module.load_dataset()
    data_module.prepare_dataset()

    model = Pretraining(args, data_module.tokenizer)

    logging_callback = LoggingCallback()

    kwargs = {
        'weights_summary': None,
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        'num_sanity_val_steps': 5,
    }

    args.val_check_interval = args.val_check_interval * args.accumulate_grad_batches

    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
