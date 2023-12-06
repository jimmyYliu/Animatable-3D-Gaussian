import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import glob
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from animatableGaussian.utils import Camera, ModelParam


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "body_pose": torch.from_numpy(smpl_params["body_pose"].astype(np.float32)),
        "global_orient": torch.from_numpy(smpl_params["global_orient"].astype(np.float32)),
        "transl": torch.from_numpy(smpl_params["transl"].astype(np.float32)),
    }


def time_encoding(t, dtype, max_freq=4):
    time_enc = torch.empty(max_freq * 2 + 1, dtype=dtype)

    for i in range(max_freq):
        time_enc[2 * i] = np.sin(2 ** i * torch.pi * t)
        time_enc[2 * i + 1] = np.cos(2 ** i * torch.pi * t)
    time_enc[max_freq * 2] = t
    return time_enc


def focal2tanfov(focal, pixels):
    return pixels/(2*focal)


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def getProjectionMatrix(tanHalfFovY, tanHalfFovX, znear=0.01, zfar=100.0):
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getCamPara(dataroot, opt):
    camera = np.load(os.path.join(dataroot, "cameras.npz"))
    intr = camera["intrinsic"]
    intr[:2] /= opt.downscale
    c2w = np.linalg.inv(camera["extrinsic"])
    height = int(camera["height"] / opt.downscale)
    width = int(camera["width"] / opt.downscale)
    focal_length_x = intr[0, 0]
    focal_length_y = intr[1, 1]
    tanFovY = focal2tanfov(focal_length_y, height)
    tanFovX = focal2tanfov(focal_length_x, width)
    camera_params = Camera()
    projmatrix = getProjectionMatrix(tanFovY, tanFovX)
    viewmatrix = torch.Tensor(c2w)

    camera_params.image_height = height
    camera_params.image_width = width
    camera_params.tanfovx = tanFovX
    camera_params.tanfovy = tanFovY
    camera_params.bg = torch.ones([3, height, width])
    camera_params.scale_modifier = 1.0
    camera_params.viewmatrix = viewmatrix.T
    camera_params.projmatrix = (projmatrix@viewmatrix).T
    camera_params.campos = torch.inverse(camera_params.viewmatrix)[3, :3]
    return camera_params


def read_imgs(dataroot, opt, resolution):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    img_lists = sorted(
        glob.glob(f"{dataroot}/images/*.png"))[start:end:skip]
    msk_lists = sorted(
        glob.glob(f"{dataroot}/masks/*.npy"))[start:end:skip]
    frame_count = len(img_lists)
    imgs = []

    for index in tqdm(range(frame_count)):
        image_path = img_lists[index]
        img = Image.open(image_path)
        img = PILtoTorch(img, resolution)
        msk_path = msk_lists[index]
        msk = torch.from_numpy(np.load(msk_path).astype(np.float32))[
            None, None, ...]
        msk = F.interpolate(msk, scale_factor=1/opt.downscale,
                            mode='bilinear')
        imgs.append(img[:3, ...] * msk[0] + 1-msk[0])
    return imgs


def load_pose(dataroot, opt, split):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    if os.path.exists(os.path.join(dataroot, f"poses/anim_nerf_{split}.npz")):
        cached_path = os.path.join(dataroot, f"poses/anim_nerf_{split}.npz")
    elif os.path.exists(os.path.join(dataroot, f"poses/{split}.npz")):
        cached_path = os.path.join(dataroot, f"poses/{split}.npz")
    else:
        cached_path = None

    if cached_path and os.path.exists(cached_path):
        print(f"[{split}] Loading from", cached_path)
        smpl_params = load_smpl_param(cached_path)
    else:
        print(f"[{split}] No optimized smpl found.")
        smpl_params = load_smpl_param(os.path.join(dataroot, f"poses.npz"))
        for k, v in smpl_params.items():
            if k != "betas":
                smpl_params[k] = v[start:end:skip]
    return smpl_params


class PeopleSnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, max_freq, split, opt):
        self.split = split
        self.max_freq = max_freq
        self.camera_params = getCamPara(dataroot, opt)
        self.imgs = read_imgs(
            dataroot, opt, (self.camera_params.image_width, self.camera_params.image_height))
        self.smpl_params = load_pose(dataroot, opt, split)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Returns:
            data["camera_params"] (vars(Camera)) : Input dict for gaussian rasterizer.
            data["model_param"] (vars(ModelParam)) : Input dict for a deformer model.
            data["gt"] (torch.Tensor[3, h, w]) : Ground truth image.
            data["time"] (torch.Tensor[max_freq * 2 + 1,]) : Time normalized to 0-1.
        """
        t = index / self.__len__()

        smpl_param = ModelParam()
        smpl_param.global_orient = self.smpl_params["global_orient"][None, index]
        smpl_param.body_pose = self.smpl_params["body_pose"][index].reshape([
                                                                            1, -1, 3])
        smpl_param.transl = self.smpl_params["transl"][None, index]

        data = {"camera_params": vars(self.camera_params),
                "model_param": vars(smpl_param),
                "gt": self.imgs[index],
                "time":  time_encoding(t, self.imgs[index].dtype, self.max_freq)}
        return data


def my_collate_fn(batch):
    return batch[0]


class PeopleSnapshotDataModule(pl.LightningDataModule):
    def __init__(self, num_workers, opt, train=True, **kwargs):
        super().__init__()
        if train:
            splits = ["train", "val"]
        else:
            splits = ["test"]
        for split in splits:
            print(f"loading {split}set...")
            dataset = PeopleSnapshotDataset(
                opt.dataroot, opt.max_freq, split, opt.get(split))
            setattr(self, f"{split}set", dataset)
        self.num_workers = num_workers

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(self.trainset,
                              shuffle=True,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn)
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(self.valset,
                              shuffle=False,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn)
        else:
            return super().val_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(self.testset,
                              shuffle=False,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn)
        else:
            return super().test_dataloader()
