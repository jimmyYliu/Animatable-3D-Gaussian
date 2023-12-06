import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import math
from PIL import Image
from tqdm import tqdm
from animatableGaussian.utils import Camera, ModelParam


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


def matrix_TRS(position, rotation, scale):

    num = rotation[0] * 2.0
    num2 = rotation[1] * 2.0
    num3 = rotation[2] * 2.0
    num4 = rotation[0] * num
    num5 = rotation[1] * num2
    num6 = rotation[2] * num3
    num7 = rotation[0] * num2
    num8 = rotation[0] * num3
    num9 = rotation[1] * num3
    num10 = rotation[3] * num
    num11 = rotation[3] * num2
    num12 = rotation[3] * num3

    result = torch.zeros((4, 4))

    result[0, 0] = scale[0] * (1.0 - (num5 + num6))
    result[0, 1] = scale[1] * (num7 - num12)
    result[0, 2] = scale[2] * (num8 + num11)
    result[0, 3] = position[0]

    result[1, 0] = scale[0] * (num7 + num12)
    result[1, 1] = scale[1] * (1.0 - (num4 + num6))
    result[1, 2] = scale[2] * (num9 - num10)
    result[1, 3] = position[1]

    result[2, 0] = scale[0] * (num8 - num11)
    result[2, 1] = scale[1] * (num9 + num10)
    result[2, 2] = scale[2] * (1.0 - (num4 + num5))
    result[2, 3] = position[2]

    result[3, 0] = 0.0
    result[3, 1] = 0.0
    result[3, 2] = 0.0
    result[3, 3] = 1.0

    return result


def read_matrix(file_path):
    positions = []
    rotations = []
    scales = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            # Read bone joint's position
            pos_line = lines[i].strip().split(',')
            positions.append(list(map(float, pos_line)))

            # Read bone joint's rotation
            rot_line = lines[i + 1].strip().split(',')
            rotations.append(list(map(float, rot_line)))

            # Read bone joint's scale
            scale_line = lines[i + 2].strip().split(',')
            scales.append(list(map(float, scale_line)))

    # Convert the lists to PyTorch tensors
    positions_tensor = torch.tensor(positions, dtype=torch.float32)
    rotations_tensor = torch.tensor(rotations, dtype=torch.float32)
    scales_tensor = torch.tensor(scales, dtype=torch.float32)

    return positions_tensor, rotations_tensor, scales_tensor


def getProjectionMatrix(tanHalfFovY, tanHalfFovX, znear=0.01, zfar=1000.0):
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


def get_proj_mat(near, far, fov, screenRatio):
    edge = math.tan(math.radians(fov*.5)) * near
    scale = near / edge
    c = -(far + near) / (far - near)
    d = -(2.0 * far * near) / (far - near)
    mat = torch.zeros((4, 4))
    mat[0, 0] = scale / screenRatio
    mat[1, 1] = scale
    mat[2, 2] = c
    mat[2, 3] = d
    mat[3, 2] = -1.0
    return mat


def getCamPara(dataroot, camera_ids, opt):
    # load all camera and corresponding bg image

    camera_params = []
    for i in range(len(camera_ids)):
        file_path = os.path.join(dataroot, "camera", str(camera_ids[i])+".txt")
        with open(file_path, 'r') as file:
            lines = file.readlines()
            line = lines[0].strip().split(',')
            property = list(map(float, line))
            line = lines[1].strip().split(',')
            position = list(map(float, line))
            line = lines[2].strip().split(',')
            rotation = list(map(float, line))
            line = lines[3].strip().split(',')
            scaling = list(map(float, line))

            fov = property[0]
            height = int(property[2])
            width = int(property[1])

            wh_ratio = property[1]/property[2]

            projmatrix = get_proj_mat(0.01, 1000, fov, wh_ratio)
            viewmatrix = matrix_TRS(position, rotation, scaling).inverse()

            # works well with unity coordinate system.
            viewmatrix[[0, 0], :] *= -1
            bg_path = os.path.join(dataroot, "bg", str(camera_ids[i])+".png")
            # Follow the following code exactly to read and process images
            bg = Image.open(bg_path)
            bg = PILtoTorch(bg, [width, height])

            # strictly follow the data format defined in Camera
            camera_param = Camera()
            camera_param.image_height = height
            camera_param.image_width = width
            camera_param.tanfovx = 1./projmatrix[0, 0]
            camera_param.tanfovy = 1./projmatrix[1, 1]
            # replace with the real bg
            camera_param.bg = bg
            camera_param.scale_modifier = 1.0
            camera_param.viewmatrix = viewmatrix.T
            camera_param.projmatrix = (projmatrix@viewmatrix).T
            camera_param.campos = torch.inverse(camera_param.viewmatrix)[3, :3]
            camera_params.append(camera_param)

    return camera_params


def read_imgs(dataroot, with_mask, camera_ids, opt, resolution):
    # return [num_cameras, time] list of image Tensor

    imgs = []
    masks = []
    for i in range(len(camera_ids)):
        viewImg = []
        viewMask = []
        view_id = camera_ids[i]
        print(f"loading images from camera {view_id}")
        for t in tqdm(range(opt.start, opt.end, opt.skip)):
            image_path = os.path.join(dataroot, str(
                view_id), str(t).zfill(4)+".png")
            # Follow the following code exactly to read and process images
            img = Image.open(image_path)
            img = PILtoTorch(img, resolution)
            viewImg.append(img[:3, ...])
            if with_mask:
                mask_path = os.path.join(dataroot, str(
                    view_id), str(t).zfill(4)+"_mask.png")
                mask = Image.open(mask_path)
                mask = PILtoTorch(mask, resolution)
                viewMask.append(mask[:3, ...])
        imgs.append(viewImg)
        if with_mask:
            masks.append(viewMask)
    # imgs = torch.stack(imgs)
    # masks = torch.stack(masks)

    return imgs, masks


def read_pose(file_path):
    positions = []
    rotations = []
    scales = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            # Read bone joint's position
            pos_line = lines[i].strip().split(',')
            positions.append(list(map(float, pos_line)))

            # Read bone joint's rotation
            rot_line = lines[i + 1].strip().split(',')
            rotations.append(list(map(float, rot_line)))

            # Read bone joint's scale
            scale_line = lines[i + 2].strip().split(',')
            scales.append(list(map(float, scale_line)))

    return torch.tensor(positions), torch.tensor(rotations), torch.tensor(scales)


def load_pose(dataroot, num_players, opt, split):
    # return [num_cameras, time] list of ModelParams,strictly follow the data format defined in ModelParam
    poses = []
    for t in range(opt.start, opt.end, opt.skip):
        pose = ModelParam()
        file_path = os.path.join(dataroot, "pose", str(t).zfill(4)+".txt")
        positions, rotations, scales = read_pose(file_path)
        positions = positions.reshape([num_players, 27, 3])
        rotations = rotations.reshape([num_players, 27, 3])
        scales = scales.reshape([num_players, 27, 3])

        pose.body_pose = rotations[:, 1:]
        pose.global_orient = rotations[:, 0]
        pose.transl = positions[:, 0]
        # pose.scale = scales[0].unsqueeze(0)
        poses.append(pose)
    return poses


class GalaBasketballDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, num_players, with_mask, max_freq, split, opt):
        self.split = split
        self.max_freq = max_freq
        self.opt = opt
        self.num_frames = (opt.end-opt.start)/opt.skip

        if type(opt.camera_ids) == str:
            self.camera_ids = list(map(int, opt.camera_ids.strip().split(',')))
            self.num_cameras = len(self.camera_ids)
        else:
            self.num_cameras = 1
            self.camera_ids = [opt.camera_ids]

        self.camera_params = getCamPara(dataroot, self.camera_ids, opt)
        self.image_width = self.camera_params[0].image_width
        self.image_height = self.camera_params[0].image_height

        self.imgs, self.masks = read_imgs(
            dataroot, with_mask, self.camera_ids, opt, (self.image_width, self.image_height))

        self.poses = load_pose(dataroot, num_players, opt, split)

    def __len__(self):
        return len(self.imgs)*len(self.imgs[0])

    def __getitem__(self, index):
        """
        Returns:
            data["camera_params"] (vars(Camera)) : Input dict for gaussian rasterizer.
            data["model_param"] (vars(ModelParam)) : Input dict for a deformer model.
            data["gt"] (torch.Tensor[3, h, w]) : Ground truth image.
            data["time"] (torch.Tensor[max_freq * 2 + 1,]) : Time normalized to 0-1.
        """
        view_id = index % self.num_cameras
        t = index//self.num_cameras

        data = {"camera_params": vars(self.camera_params[view_id]),
                "model_param": vars(self.poses[t]),
                "gt": self.imgs[view_id][t],
                "time":  time_encoding(t/self.num_frames, self.imgs[view_id][t].dtype, self.max_freq)}
        return data


def my_collate_fn(batch):
    return batch[0]


class GalaBasketballDataModule(pl.LightningDataModule):
    def __init__(self, num_workers, num_players, opt, train=True, **kwargs):
        super().__init__()

        if train:
            splits = ["train", "val"]
        else:
            splits = ["test"]
        for split in splits:
            print(f"loading {split}set...")
            dataset = GalaBasketballDataset(
                opt.dataroot, num_players, opt.with_mask, opt.max_freq, split, opt.get(split))
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
