import torch.nn as nn
from animatableGaussian.deformer.encoder.position_encoder import SHEncoder, DisplacementEncoder
from animatableGaussian.deformer.encoder.time_encoder import AOEncoder
import torch
import pickle
import os
import numpy as np
from animatableGaussian.deformer.lbs import lbs
from simple_knn._C import distCUDA2


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


class SMPLModel(nn.Module):
    def __init__(self, model_path, max_sh_degree=0, max_freq=4, gender="male", num_repeat=20, num_players=1, use_point_color=False, use_point_displacement=False, enable_ambient_occlusion=False):
        super().__init__()

        self.num_players = num_players
        self.enable_ambient_occlusion = enable_ambient_occlusion
        self.use_point_displacement = use_point_displacement
        self.use_point_color = use_point_color

        smpl_path = os.path.join(
            model_path, 'SMPL_{}'.format(gender.upper()))
        v_template = np.loadtxt(os.path.join(
            smpl_path, 'v_template.txt'))
        weights = np.loadtxt(os.path.join(
            smpl_path, 'weights.txt'))
        kintree_table = np.loadtxt(os.path.join(
            smpl_path, 'kintree_table.txt'))
        J = np.loadtxt(os.path.join(
            smpl_path, 'joints.txt'))
        self.register_buffer('v_template', torch.Tensor(
            v_template)[None, ...].repeat(
                [self.num_players, 1, 1]))
        dist2 = torch.clamp_min(
            distCUDA2(self.v_template[0].cuda()), 0.0000001)[..., None].repeat([num_repeat, 3])
        self.v_template = self.v_template.repeat([1, num_repeat, 1])
        self.v_template += (torch.rand_like(self.v_template) - 0.5) * \
            dist2.cpu() * 20
        dist2 /= num_repeat
        self.weights = nn.Parameter(
            torch.Tensor(weights).repeat([num_repeat, 1]))
        self.parents = kintree_table[0].astype(np.int64)
        self.parents[0] = -1

        self.J = nn.Parameter(torch.Tensor(
            J)[None, ...].repeat([self.num_players, 1, 1]))

        minmax = [self.v_template[0].min(
            dim=0).values * 1.05,  self.v_template[0].max(dim=0).values * 1.05]
        self.register_buffer('normalized_vertices',
                             (self.v_template - minmax[0]) / (minmax[1] - minmax[0]))

        if use_point_displacement:
            self.displacements = nn.Parameter(
                torch.zeros_like(self.v_template))
        else:
            self.displacementEncoder = DisplacementEncoder(
                encoder="hash", num_players=num_players)

        n = self.v_template.shape[1] * num_players

        if use_point_color:
            self.shs_dc = nn.Parameter(torch.zeros(
                [n, 1, 3]))
            self.shs_rest = nn.Parameter(torch.zeros(
                [n, (max_sh_degree + 1) ** 2 - 1, 3]))
        else:
            self.shEncoder = SHEncoder(max_sh_degree=max_sh_degree,
                                       encoder="hash", num_players=num_players)
        self.opacity = nn.Parameter(inverse_sigmoid(
            0.2 * torch.ones((n, 1), dtype=torch.float)))
        self.scales = nn.Parameter(
            torch.log(torch.sqrt(dist2)).repeat([num_players, 1]))
        rotations = torch.zeros([n, 4])
        rotations[:, 0] = 1
        self.rotations = nn.Parameter(rotations)

        if enable_ambient_occlusion:
            self.aoEncoder = AOEncoder(
                encoder="hash", max_freq=max_freq, num_players=num_players)
        self.register_buffer("aos", torch.ones_like(self.opacity))

    def configure_optimizers(self, training_args):
        l = [
            {'params': [self.weights],
                'lr': training_args.weights_lr, "name": "weights"},
            {'params': [self.J], 'lr': training_args.joint_lr, "name": "J"},
            {'params': [self.opacity],
                'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.scales],
                'lr': training_args.scaling_lr, "name": "scales"},
            {'params': [self.rotations],
                'lr': training_args.rotation_lr, "name": "rotations"}
        ]

        if self.enable_ambient_occlusion:
            l.append({'params': self.aoEncoder.parameters(),
                      'lr': training_args.ao_lr, "name": "aoEncoder"})
        if self.use_point_displacement:
            l.append({'params': [self.displacements],
                      'lr': training_args.displacement_lr, "name": "displacements"})
        else:
            l.append({'params': self.displacementEncoder.parameters(),
                      'lr': training_args.displacement_encoder_lr, "name": "displacementEncoder"})
        if self.use_point_color:
            l.append({'params': [self.shs_dc],
                      'lr': training_args.shs_lr, "name": "shs"})
            l.append({'params': [self.shs_rest],
                      'lr': training_args.shs_lr/20.0, "name": "shs"})
        else:
            l.append({'params': self.shEncoder.parameters(),
                      'lr': training_args.sh_encoder_lr, "name": "shEncoder"})
        return torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def forward(self, body_pose, global_orient, transl, time, is_use_ao=False):
        """
        Caculate the transforms of vertices.

        Args:
            body_pose (torch.Tensor[num_players, J-1, 3]) : The local rotate angles of joints except root joint.
            global_orient (torch.Tensor[num_players, 3]) : The global rotate angle of root joint.
            transl (torch.Tensor[num_players, 3]) : The global translation of root joint.
            time (torch.Tensor[max_freq * 2 + 1]) : Time normalized to 0-1.

        Returns:
            vertices (torch.Tensor[N, 3]) : 
            opacity (torch.Tensor[N, 1]) : 
            scales (torch.Tensor[N, 3]) : 
            rotations (torch.Tensor[N, 4]) : 
            shs (torch.Tensor[N, (max_sh_degree + 1) ** 2, 3]) : 
            aos (torch.Tensor[N, 1]) : 
            transforms (torch.Tensor[N, 3]) : 
        """
        full_body_pose = torch.cat(
            [global_orient[:, None, :], body_pose], dim=1)

        if self.use_point_color:
            shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        else:
            shs = self.shEncoder(self.normalized_vertices)
        if self.enable_ambient_occlusion:
            aos = self.aoEncoder(self.normalized_vertices, time)
        else:
            aos = self.aos
        if self.use_point_displacement:
            v_displaced = self.v_template + self.displacements
        else:
            v_displaced = self.v_template + \
                self.displacementEncoder(self.normalized_vertices)

        T = lbs(full_body_pose, transl, self.J, self.parents, self.weights)

        return v_displaced.reshape([-1, 3]), torch.sigmoid(self.opacity), torch.exp(self.scales), torch.nn.functional.normalize(self.rotations), shs, aos, T[:, :, :3, :].reshape([-1, 3, 4])
