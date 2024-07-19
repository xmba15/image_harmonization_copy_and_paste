import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser("inference by onnxruntime")
    parser.add_argument("--onnx_weights_path", type=str, required=True)
    parser.add_argument("--composite_path", type=str, default="./assets/composite.jpg")
    parser.add_argument("--mask_path", type=str, default="./assets/mask.png")

    return parser.parse_args()


def pad_input_data(
    composite,
    mask,
    num_downs: int,
):
    height, width = composite.shape[:2]
    division = 2**num_downs
    padded_height = int(np.ceil(height / division) * division)
    padded_width = int(np.ceil(width / division) * division)

    composite = cv2.copyMakeBorder(
        composite,
        0,
        padded_height - height,
        0,
        padded_width - width,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    mask = cv2.copyMakeBorder(
        mask,
        0,
        padded_height - height,
        0,
        padded_width - width,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    return composite, mask


def main():
    args = get_args()
    assert os.path.isfile(args.composite_path)
    assert os.path.isfile(args.mask_path)

    ort_session = ort.InferenceSession(
        args.onnx_weights_path,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    input1_name = ort_session.get_inputs()[0].name
    input2_name = ort_session.get_inputs()[1].name
    output_name = ort_session.get_outputs()[0].name

    composite = cv2.imread(args.composite_path)[:, :, ::-1]
    mask = cv2.imread(args.mask_path, 0)
    mask[mask > 0] = 1
    orig_height, orig_width = composite.shape[:2]

    composite, mask = pad_input_data(
        composite,
        mask,
        num_downs=8,
    )

    output = ort_session.run(
        [output_name],
        {
            input1_name: composite.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0,
            input2_name: mask[None, ...].astype(np.float32),
        },
    )[0]
    output = np.clip(output[0].transpose(1, 2, 0), 0, 1) * 255
    output = output.astype(np.uint8)
    output = output[:orig_height, :orig_width]
    cv2.imwrite("output.jpg", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
