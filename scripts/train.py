# flake8: noqa: E402
import argparse
import os
import sys

import pytorch_lightning as pl
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from train_utils import get_transforms

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import HarmonyDataset, HarmonySubDatasetType
from src.integrated import BargainNetPl
from src.utils import fix_seed, worker_init_fn


def get_args():
    parser = argparse.ArgumentParser("training")
    parser.add_argument("--config_path", type=str, default="./config/base.yaml")

    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.isfile(args.config_path)
    with open(args.config_path, encoding="utf-8") as _file:
        hparams = yaml.load(_file, Loader=yaml.SafeLoader)
    os.makedirs(hparams["output_root_dir"], exist_ok=True)
    fix_seed(hparams["seed"])
    pl.seed_everything(hparams["seed"])

    transforms_dict = get_transforms(hparams)
    train_dataset = HarmonyDataset(
        hparams["dataset"]["root_dir"],
        sub_dataset_types=[HarmonySubDatasetType[_name] for _name in hparams["dataset"]["sub_datasets"]],
        set_type="train",
        transforms=transforms_dict["train"],
        opt_train_size=hparams["dataset"]["train_size"],
        seed=hparams["seed"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["train_parameters"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=hparams["num_workers"],
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    val_dataset = HarmonyDataset(
        hparams["dataset"]["root_dir"],
        sub_dataset_types=[HarmonySubDatasetType[_name] for _name in hparams["dataset"]["sub_datasets"]],
        set_type="val",
        transforms=transforms_dict["val"],
        opt_train_size=hparams["dataset"]["train_size"],
        seed=hparams["seed"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams["val_parameters"]["batch_size"],
        num_workers=hparams["num_workers"],
    )

    trainer = Trainer(
        default_root_dir=hparams["output_root_dir"],
        max_epochs=hparams["trainer"]["max_epochs"],
        log_every_n_steps=hparams["trainer"]["log_every_n_steps"],
        devices=hparams["trainer"]["devices"],
        accelerator=hparams["trainer"]["accelerator"],
        deterministic=True,
        num_sanity_val_steps=hparams["trainer"]["num_sanity_val_steps"],
        logger=TensorBoardLogger(
            save_dir=hparams["output_root_dir"],
            version=f"{hparams['experiment_name']}_"
            f"{hparams['train_parameters']['batch_size']}_"
            f"{hparams['optimizer']['lr']}",
            name=f"{hparams['experiment_name']}",
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(
        BargainNetPl(hparams),
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()
