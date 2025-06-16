import torch
from torch.utils.data import DataLoader
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf 
from lightning.pytorch.utilities.types import STEP_OUTPUT

import click
import os.path as osp
from typing import Any
from typing_extensions import override
import yaml

from data import get_dataset
from models import get_model


class IterationBasedProgressBar(TQDMProgressBar):
    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        pass

    @override
    def on_train_start(self, trainer: "pl.Trainer", *_: "Any") -> None:
        self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.update(trainer.global_step)

    @override
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.train_progress_bar.total is None:
            self.train_progress_bar.total = convert_inf(trainer.max_steps)
        self.train_progress_bar.update(1)
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


@click.command()
@click.option('--config_path', type=str, required=True, help='Json file path for configurations')
@click.option('--resume_path', type=str, help='Path to resume')
@click.option('--debug', is_flag=True, show_default=True, default=False, help='Enable debugging mode')
def main(config_path, resume_path, debug):
    # One-time torch configurations
    torch.set_float32_matmul_precision('medium')

    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)

    # Initializing training and validation data
    train_dataset = get_dataset(opt['datasets']['train'])
    valid_dataset = get_dataset(opt['datasets']['val'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt['datasets']['train']['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=opt['datasets']['train']['num_workers'],
        persistent_workers=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=opt['datasets']['val']['batch_size'],
        num_workers=1,
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
    )

    # Using customized checkpointing
    checkpoint_dir = opt['train']['checkpoint_dir']
    dirpath = osp.sep.join(resume_path.split(osp.sep)[:-1]) if resume_path is not None else None
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename='{step:08d}',
        save_last=True,
        save_top_k=5,
        monitor='step',
        mode='max',
        every_n_train_steps=opt['train']['save_interval'],
        every_n_epochs=None,
        enable_version_counter=True
    )

    # Preparing logger(s)
    tb_logger = pl_loggers.TensorBoardLogger(checkpoint_dir)

    trainer = L.Trainer(
        accelerator='gpu',
        num_nodes=opt['train']['num_nodes'],
        devices=opt['train']['num_gpus'],
        strategy=DDPStrategy(find_unused_parameters=True),

        # Debugging parameters
        fast_dev_run=debug,
        limit_train_batches=0.2 if debug else 1.0,
        limit_val_batches=0.2 if debug else 1.0,
        num_sanity_val_steps=0 if debug else 0,
        profiler='advanced' if debug else 'simple',

        callbacks=[
            checkpoint_callback,
            ModelSummary(max_depth=3),
            # DeviceStatsMonitor(),
            IterationBasedProgressBar(),
        ],
        logger=[tb_logger],

        # Monitoring parammeters
        log_every_n_steps=50,
        val_check_interval=opt['train']['val_check_interval'],
        check_val_every_n_epoch=None,

        # Training hyperparams
        max_epochs=-1,
        max_steps=opt['train']['max_steps'],
        precision='bf16-mixed' if opt['train']['use_mixed_precision'] else '32',
    )

    # Preparing model
    model = get_model(opt['model'])
    if opt['train']['use_compile']:
        model = torch.compile(model, mode='reduce-overhead')

    trainer.fit(
        model, train_loader, valid_loader,
        ckpt_path=resume_path,
    )


if __name__ == '__main__':
    main()
