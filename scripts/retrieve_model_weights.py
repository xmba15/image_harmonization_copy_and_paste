import argparse
import os
import pydoc

import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser("retrieve pytorch model weights from pytorch lightning checkpoint")
    parser.add_argument("--config_path", type=str, default="./config/base.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.isfile(args.checkpoint_path)

    with open(args.config_path, encoding="utf-8") as _file:
        hparams = yaml.load(_file, Loader=yaml.SafeLoader)

    pl_model = pydoc.locate(hparams["model"]["pl_class"]).load_from_checkpoint(
        args.checkpoint_path,
        hparams=hparams,
        map_location="cpu",
    )
    torch.save(pl_model.model.state_dict(), args.weights_path)
    print(f"saved model weights to {args.weights_path}")


if __name__ == "__main__":
    main()
