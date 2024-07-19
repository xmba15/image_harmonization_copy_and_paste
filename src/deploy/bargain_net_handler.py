import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from loguru import logger

from src.models import BargainNet

__all__ = ("BargainNetHandler",)


class BargainNetHandler:
    __DEFAULT_CONFIG: Dict[str, Any] = {
        "num_downs": 8,
        "style_dim": 16,
        "use_gpu": False,
    }

    def __init__(
        self,
        weights_path: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        assert os.path.isfile(weights_path)

        self._config = self.__DEFAULT_CONFIG.copy()
        if config is not None:
            self._config.update(config)

        if self._config["use_gpu"] and not torch.cuda.is_available():
            logger.info("gpu environment is not available, falling back to cpu")
            self._config["use_gpu"] = False
        self._device = torch.device("cuda" if self._config["use_gpu"] else "cpu")

        self._model = BargainNet(
            style_dim=self._config["style_dim"],
            num_downs=self._config["num_downs"],
        )
        self._model.load_state_dict(torch.load(weights_path))
        self._model.eval()
        self._model.to(self._device)

    @torch.no_grad()
    def run(
        self,
        composite: np.ndarray,
        mask: np.ndarray,
    ):
        """
        composite: 3 channel RGB
        mask: 1 channel mask image that specifies foreground object by 1
        """

        orig_height, orig_width = composite.shape[:2]
        mask[mask > 0] = 1
        composite, mask = self.pad_input_data(
            composite,
            mask,
            num_downs=self._config["num_downs"],
        )

        output = self._model(*self.process_input_data(composite, mask)).cpu().numpy()
        output = np.clip(output[0].transpose(1, 2, 0), 0, 1) * 255
        output = output.astype(np.uint8)
        output = output[:orig_height, :orig_width]

        return output

    def process_input_data(self, composite, mask):
        return (
            torch.from_numpy(composite.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0).to(self._device),
            torch.from_numpy(mask[None, ...].astype(np.float32)).to(self._device),
        )

    def pad_input_data(
        self,
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
