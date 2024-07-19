# flake8: noqa: E402
import argparse
import os
import pydoc
import sys

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from train_utils import get_transforms

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import HarmonyDataset, HarmonySubDatasetType
from src.integrated import BargainNetPl
from src.utils import fix_seed


def get_args():
    parser = argparse.ArgumentParser("training")
    parser.add_argument("--config_path", type=str, default="./config/base.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)

    return parser.parse_args()


def pad_input_data(
    composite,
    mask,
    num_downs: int,
):
    height, width = composite.shape[-2:]
    division = 2 ** num_downs
    padded_height = int(np.ceil(height / division) * division)
    padded_width = int(np.ceil(width / division) * division)

    composite = F.pad(composite, (0, padded_width - width, 0, padded_height - height), mode="constant", value=0)
    mask = F.pad(mask, (0, padded_width - width, 0, padded_height - height), mode="constant", value=0)

    return composite, mask


def main():
    args = get_args()
    assert os.path.isfile(args.config_path)
    assert os.path.isfile(args.checkpoint_path)
    with open(args.config_path, encoding="utf-8") as _file:
        hparams = yaml.load(_file, Loader=yaml.SafeLoader)
    os.makedirs(hparams["output_root_dir"], exist_ok=True)
    fix_seed(hparams["seed"])
    pl.seed_everything(hparams["seed"])

    transforms_dict = get_transforms(hparams)
    test_dataset = HarmonyDataset(
        hparams["dataset"]["root_dir"],
        sub_dataset_types=[HarmonySubDatasetType[_name] for _name in hparams["dataset"]["sub_datasets"]],
        set_type="test",
        transforms=transforms_dict["test"],
        seed=hparams["seed"],
    )
    assert args.idx < len(test_dataset)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = pydoc.locate(hparams["model"]["pl_class"]).load_from_checkpoint(
        args.checkpoint_path,
        hparams=hparams,
        map_location=device,
    )
    model.eval()
    model.to(device)

    composite, _, mask = test_dataset[args.idx]
    orig_height, orig_width = composite.shape[-2:]

    composite_uint = composite.permute(1, 2, 0).cpu().numpy()
    composite_uint = (composite_uint * 255).astype(np.uint8)
    cv2.imwrite("composite.jpg", cv2.cvtColor(composite_uint, cv2.COLOR_RGB2BGR))

    composite, mask = pad_input_data(
        composite,
        mask,
        num_downs=hparams["model"]["num_downs"],
    )

    logger.info(f"start inference for image of shape {composite.shape[-2:]}")
    with torch.no_grad():
        output = model(
            composite[None, ...].to(device),
            mask[None, ...].to(device),
        )

    output = output[0].permute(1, 2, 0)
    output.clamp_(min=0, max=1)
    output = output.cpu().numpy()
    output = (output * 255).astype(np.uint8)
    output = output[:orig_height, :orig_width]
    cv2.imwrite("output.jpg", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
