# flake8: noqa: E402
import argparse
import os
import sys

import cv2

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.deploy import BargainNetHandler


def get_args():
    parser = argparse.ArgumentParser("inference single image")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--composite_path", type=str, default="./assets/composite.jpg")
    parser.add_argument("--mask_path", type=str, default="./assets/mask.png")
    parser.add_argument("--use_gpu", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.isfile(args.weights_path)
    assert os.path.isfile(args.composite_path)
    assert os.path.isfile(args.mask_path)

    bargain_net_handler = BargainNetHandler(
        weights_path=args.weights_path,
        config={
            "use_gpu": args.use_gpu,
        },
    )

    composite = cv2.imread(args.composite_path)[:, :, ::-1]
    mask = cv2.imread(args.mask_path, 0)
    mask[mask > 0] = 1
    orig_height, orig_width = composite.shape[:2]

    output = bargain_net_handler.run(
        composite,
        mask,
    )
    cv2.imwrite("output.jpg", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
