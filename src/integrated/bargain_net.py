import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.utils import make_grid

from src.models import BargainNet
from src.utils import get_object_from_dict

__all__ = ("BargainNetPl",)


class BargainNetPl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        self.model = BargainNet(
            style_dim=self.hparams["model"]["style_dim"],
            num_downs=self.hparams["model"]["num_downs"],
        )

        self.triplet_criterion = nn.TripletMarginLoss(
            margin=hparams["loss"]["triplet_loss"]["margin"],
            p=2,
        )

        self.reconstruction_criterion = nn.L1Loss()
        self.acc = PeakSignalNoiseRatio()
        self.automatic_optimization = False

    def forward(self, composite, mask):
        return self.model(composite, mask)

    def common_step(self, batch, batch_idx):
        composite, real, mask = batch

        optimizer_sty, optimizer_g = self.optimizers()
        optimizer_sty.zero_grad()
        optimizer_g.zero_grad()

        b, _, h, w = composite.shape

        real_fg_sty_v = self.model.style_encoder(real, mask[:, None, :, :])
        bg_sty_v = self.model.style_encoder(composite, 1 - mask[:, None, :, :])

        harmonized = self.model.generator(
            torch.cat(
                [
                    composite,
                    mask[:, None, :, :],
                    bg_sty_v.expand(b, self.model.style_dim, h, w),
                ],
                1,
            )
        )

        harmonized_fg_sty_v = self.model.style_encoder(harmonized, mask[:, None, :, :])
        comp_fg_sty_v = self.model.style_encoder(composite, mask[:, None, :, :])

        reconstruction_loss = self.reconstruction_criterion(harmonized, real)
        triplet_loss = self.triplet_criterion(
            real_fg_sty_v, harmonized_fg_sty_v, comp_fg_sty_v
        ) + self.triplet_criterion(harmonized_fg_sty_v, bg_sty_v, comp_fg_sty_v)

        total_loss = reconstruction_loss + 0.01 * triplet_loss

        acc = self.acc(harmonized, real)

        return total_loss, reconstruction_loss, triplet_loss, harmonized, acc

    def training_step(self, batch, batch_idx):
        optimizer_sty, optimizer_g = self.optimizers()
        optimizer_sty.zero_grad()
        optimizer_g.zero_grad()

        total_loss, reconstruction_loss, triplet_loss, harmonized, acc = self.common_step(batch, batch_idx)

        self.log(
            "train_l1_loss",
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_triplet_loss",
            triplet_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx % 200 == 0:
            global_step = self.current_epoch * self.trainer.num_training_batches + batch_idx
            self.logger.experiment.add_image(
                "train_composite",
                make_grid(
                    batch[0],
                    nrow=batch[0].shape[0],
                ),
                global_step=global_step,
            )

            self.logger.experiment.add_image(
                "train_harmonized",
                make_grid(
                    harmonized,
                    nrow=batch[0].shape[0],
                ),
                global_step=global_step,
            )

        self.manual_backward(total_loss)
        optimizer_sty.step()
        optimizer_g.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, reconstruction_loss, triplet_loss, harmonized, acc = self.common_step(batch, batch_idx)

        self.log(
            "val_l1_loss",
            reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_triplet_loss",
            triplet_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        if batch_idx == 0:
            global_step = self.current_epoch * self.trainer.num_val_batches[0] + batch_idx
            self.logger.experiment.add_image(
                "val_composite",
                make_grid(
                    batch[0],
                    nrow=batch[0].shape[0],
                ),
                global_step=global_step,
            )

            self.logger.experiment.add_image(
                "val_harmonized",
                make_grid(
                    harmonized,
                    nrow=batch[0].shape[0],
                ),
                global_step=global_step,
            )

        return acc

    def configure_optimizers(self):
        optimizer_style_encoder = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.style_encoder.parameters() if x.requires_grad],
        )
        optimizer_generator = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.generator.parameters() if x.requires_grad],
        )

        return [optimizer_style_encoder, optimizer_generator], []
