# flake8: noqa: E402
import argparse
import os
import sys

import torch
import yaml

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.models import BargainNet


def get_args():
    parser = argparse.ArgumentParser("extract onnx weights")
    parser.add_argument("--config_path", type=str, default="./config/base.yaml")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--onnx_weights_path", type=str, default="bargain_net.onnx")

    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.isfile(args.weights_path)

    with open(args.config_path, encoding="utf-8") as _file:
        hparams = yaml.load(_file, Loader=yaml.SafeLoader)

    model = BargainNet(
        style_dim=hparams["model"]["style_dim"],
        num_downs=hparams["model"]["num_downs"],
    )
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    batch_size = 1
    channels = 3
    height = 256
    width = 256

    composite = torch.randn(batch_size, channels, height, width)
    mask = torch.randn(batch_size, height, width)

    input_names = ["composite", "mask"]
    output_names = ["output"]
    dynamic_axes = {
        "composite": {0: "batch_size", 2: "height", 3: "width"},
        "mask": {0: "batch_size", 1: "height", 2: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        model,
        (composite, mask),
        args.onnx_weights_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
