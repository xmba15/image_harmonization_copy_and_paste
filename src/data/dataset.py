import os
from enum import Enum
from typing import Optional

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

__all__ = (
    "HarmonySubDatasetType",
    "HarmonySubDataset",
    "HarmonyDataset",
)


class HarmonySubDatasetType(Enum):
    HAdobe5k = 0
    HCOCO = 1
    Hday2night = 2
    HFlickr = 3


class HarmonySubDataset(Dataset):
    def __init__(
        self,
        subset_root_dir: str,
        sub_dataset_type: HarmonySubDatasetType,
        set_type: str = "train",
        opt_train_size: Optional[bool] = None,
        seed=2024,
    ):
        super().__init__()
        assert os.path.isdir(subset_root_dir)
        self.sub_dataset_name = sub_dataset_type.name
        self.subset_root_dir = subset_root_dir
        self.set_type = set_type
        self.opt_train_size = opt_train_size
        self.seed = seed

        self._process_gt()

    def _process_gt(self):
        csv_path = (
            f"{self.sub_dataset_name}_train.txt"
            if self.set_type in ["train", "val"]
            else f"{self.sub_dataset_name}_test.txt"
        )
        csv_path = os.path.join(self.subset_root_dir, csv_path)

        composite_images_dir = os.path.join(self.subset_root_dir, "composite_images")
        masks_dir = os.path.join(self.subset_root_dir, "masks")
        real_images_dir = os.path.join(self.subset_root_dir, "real_images")

        composite_file_names = pd.read_csv(csv_path, header=None, index_col=None)
        composite_file_names = list(composite_file_names[0])

        if self.set_type in ["train", "val"]:
            splitted = train_test_split(composite_file_names, train_size=self.opt_train_size, random_state=self.seed)
            composite_file_names = splitted[0] if self.set_type == "train" else splitted[1]
            del splitted

        self.composite_image_paths = [os.path.join(composite_images_dir, _name) for _name in composite_file_names]
        self.mask_paths = [os.path.join(masks_dir, self._get_mask_file_name(_name)) for _name in composite_file_names]
        self.real_image_paths = [
            os.path.join(real_images_dir, self._get_real_image_file_name(_name)) for _name in composite_file_names
        ]

    def __len__(self):
        return len(self.composite_image_paths)

    def _get_mask_file_name(self, composite_file_name: str):
        output = composite_file_name.split(".")[0]
        output = "_".join(output.split("_")[:-1])
        return output + ".png"

    def _get_real_image_file_name(self, composite_file_name: str):
        if self.sub_dataset_name in ["HCOCO", "Hday2night"]:
            return composite_file_name.split("_")[0] + ".jpg"
        else:
            return composite_file_name.split(".")[0][:-4] + ".jpg"


class HarmonyDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sub_dataset_types=list(HarmonySubDatasetType),
        set_type: str = "train",
        transforms=None,
        opt_train_size: Optional[bool] = None,
        seed=2024,
    ):
        super().__init__()
        assert os.path.isdir(root_dir)
        assert len(sub_dataset_types) > 0

        self.root_dir = root_dir

        self.sub_dataset_types = sub_dataset_types
        self.set_type = set_type
        self.transforms = transforms
        self.opt_train_size = opt_train_size
        self.seed = seed

        self._process_gt()

    def _process_gt(self):
        sub_datasets = [
            HarmonySubDataset(
                os.path.join(self.root_dir, sub_dataset_type.name),
                sub_dataset_type,
                self.set_type,
                self.opt_train_size,
                self.seed,
            )
            for sub_dataset_type in self.sub_dataset_types
        ]
        self.composite_image_paths = []
        self.mask_paths = []
        self.real_image_paths = []

        for sub_dataset in sub_datasets:
            self.composite_image_paths += sub_dataset.composite_image_paths
            self.mask_paths += sub_dataset.mask_paths
            self.real_image_paths += sub_dataset.real_image_paths

    def __len__(self):
        return len(self.composite_image_paths)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        composite_image_path = self.composite_image_paths[idx]
        real_image_path = self.real_image_paths[idx]
        mask_path = self.mask_paths[idx]

        composite_image = cv2.imread(composite_image_path)[:, :, ::-1]
        real_image = cv2.imread(real_image_path)[:, :, ::-1]
        mask = cv2.imread(mask_path, 0)
        mask[mask > 0] = 1

        if self.transforms is not None:
            transformed = self.transforms(image=composite_image, real=real_image, mask=mask)
            composite_image = transformed["image"]
            real_image = transformed["real"]
            mask = transformed["mask"]
            del transformed

        return composite_image, real_image, mask
