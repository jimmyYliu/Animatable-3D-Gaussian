from PIL import Image
import torch
import torch.nn as nn
import hydra
import numpy as np
import pytorch_lightning as pl
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import os
from torch.cuda.amp import custom_fwd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from animatableGaussian.utils import ssim, l1_loss


class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


class NeRFModel(pl.LightningModule):
    def __init__(self, opt):
        super(NeRFModel, self).__init__()
        self.model = hydra.utils.instantiate(opt.deformer)
        self.training_args = opt.training_args
        self.sh_degree = opt.max_sh_degree
        self.lambda_dssim = opt.lambda_dssim
        self.evaluator = Evaluator()
        if not os.path.exists("val"):
            os.makedirs("val")
        if not os.path.exists("test"):
            os.makedirs("test")

    def forward(self, camera_params, model_param, time, render_point=False):
        is_use_ao = self.current_epoch > 3

        verts, opacity, scales, rotations, shs, aos, transforms = self.model(time=time, is_use_ao=is_use_ao,
                                                                             **model_param)

        means2D = torch.zeros_like(
            verts, dtype=verts.dtype, requires_grad=True, device=verts.device)
        try:
            means2D.retain_grad()
        except:
            pass
        raster_settings = GaussianRasterizationSettings(
            sh_degree=self.sh_degree,
            prefiltered=False,
            debug=False, **camera_params
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        cov3D_precomp = None
        if render_point:
            colors_precomp = torch.rand_like(scales)
            scales /= 10
            opacity *= 100
            shs = None
        else:
            colors_precomp = None
        image, radii = rasterizer(
            means3D=verts,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            aos=aos,
            transforms=transforms,
            cov3D_precomp=cov3D_precomp)
        return image

    def training_step(self, batch, batch_idx):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        image = self(camera_params, model_param, batch["time"])
        gt_image = batch["gt"]
        Ll1 = l1_loss(
            image, gt_image)
        loss = (1.0 - self.lambda_dssim) * Ll1 + \
            self.lambda_dssim * (1.0 - ssim(image, gt_image))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        rgb = self(camera_params, model_param, batch["time"])
        rgb_gt = batch["gt"]
        image = torch.cat((rgb, rgb_gt), dim=2)
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"val/{self.current_epoch}.png")

    @torch.no_grad()
    def test_step(self, batch, batch_idx, *args, **kwargs):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        rgb = self(camera_params, model_param, batch["time"])
        rgb_gt = batch["gt"]
        losses = {
            # add some extra loss here
            **self.evaluator(rgb[None], rgb_gt[None]),
            "rgb_loss": (rgb - rgb_gt).square().mean(),
        }
        image = rgb
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}.png")
        image = rgb_gt
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}_gt.png")

        for k, v in losses.items():
            self.log(f"test/{k}", v, on_epoch=True, batch_size=1)
        return {}

    def configure_optimizers(self):
        return self.model.configure_optimizers(self.training_args)
