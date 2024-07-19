import torch
from torch import nn

from .partialconv2d import PartialConv2d

__all__ = (
    "BargainNet",
    "StyleEncoder",
    "UnetGenerator",
)


class BargainNet(nn.Module):
    def __init__(
        self,
        style_dim,
        num_downs,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.num_downs = num_downs
        self.style_encoder = StyleEncoder(
            style_dim=style_dim,
        )
        self.generator = UnetGenerator(
            input_nc=4 + style_dim,
            output_nc=3,
            num_downs=num_downs,
        )

    def forward(self, composite, mask):
        """
        composite: [b, 3, h, w]
        mask: [b, h, w]
        """
        bg_sty_v = self.style_encoder(composite, 1 - mask[:, None, :, :])

        b, h, w = mask.shape
        harmonized = self.generator(
            torch.cat(
                [
                    composite,
                    mask[:, None, :, :],
                    bg_sty_v.expand(b, self.style_dim, h, w),
                ],
                1,
            )
        )

        return harmonized


class StyleEncoder(nn.Module):
    def __init__(
        self,
        style_dim=16,
    ):
        super().__init__()
        self.conv1 = PartialConv2d(3, 64, kernel_size=3, stride=2, return_mask=True)
        self.norm1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = PartialConv2d(64, 64 * 2, kernel_size=3, stride=2, return_mask=True)
        self.norm2 = nn.Sequential(
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(inplace=True),
        )

        self.conv3 = PartialConv2d(64 * 2, 64 * 4, kernel_size=3, stride=2, return_mask=True)
        self.norm3 = nn.Sequential(
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(inplace=True),
        )

        self.conv4 = PartialConv2d(64 * 4, 64 * 8, kernel_size=3, stride=2, return_mask=True)
        self.norm4 = nn.Sequential(
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(inplace=True),
        )

        self.conv5 = PartialConv2d(64 * 8, 64 * 8, kernel_size=3, stride=2, return_mask=False)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.Conv2d(64 * 8, style_dim, kernel_size=1, stride=1)

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask.float())
        x = self.norm1(x)

        x, mask = self.conv2(x, mask)
        x = self.norm2(x)

        x, mask = self.conv3(x, mask)
        x = self.norm3(x)

        x, mask = self.conv4(x, mask)
        x = self.norm4(x)

        x = self.conv5(x, mask)
        x = self.avg_pooling(x)
        x = self.convs(x)

        return x


class UnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        use_attention=True,
    ):
        super().__init__()
        weight = torch.FloatTensor([0.1])
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, use_attention=use_attention
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, use_attention=use_attention
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, use_attention=use_attention
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
        )

    def forward(self, inputs):
        return self.model(
            torch.cat(
                [
                    inputs[:, :4, :, :],
                    inputs[:, 4:, :, :] * torch.clamp(self.weight, min=0.001),
                ],
                1,
            )
        )


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        use_attention=False,
    ):
        super().__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(outer_nc + input_nc, outer_nc + input_nc, kernel_size=1),
                nn.Sigmoid(),
            )

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)

        ret = torch.cat([x, self.model(x)], 1)
        if self.use_attention:
            return self.attention(ret) * ret

        return ret
