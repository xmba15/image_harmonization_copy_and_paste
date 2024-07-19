import albumentations as alb
from albumentations.pytorch import ToTensorV2

__all__ = ("get_transforms",)


def get_transforms(hparams):
    load_sizes = hparams["load_sizes"]
    crop_size = hparams["crop_size"]

    transforms = {
        "train": alb.Compose(
            [
                alb.OneOf(
                    [alb.Resize(height=load_size, width=load_size) for load_size in load_sizes],
                    p=1.0,
                ),
                alb.CropNonEmptyMaskIfExists(height=crop_size, width=crop_size, ignore_values=[0]),
                alb.HorizontalFlip(p=0.5),
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ],
            additional_targets={
                "real": "image",
            },
        ),
        "val": alb.Compose(
            [
                alb.Resize(height=crop_size, width=crop_size),
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ],
            additional_targets={
                "real": "image",
            },
        ),
        "test": alb.Compose(
            [
                alb.ToFloat(max_value=255.0),
                ToTensorV2(),
            ],
            additional_targets={
                "real": "image",
            },
        ),
    }

    return transforms
