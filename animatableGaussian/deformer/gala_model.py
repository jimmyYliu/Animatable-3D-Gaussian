import os
import numpy as np
import torch
import torch.nn as nn
from simple_knn._C import distCUDA2
from animatableGaussian.deformer.encoder.position_encoder import SHEncoder, DisplacementEncoder
from animatableGaussian.deformer.encoder.time_encoder import AOEncoder


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def read_skinned_mesh_data(file_path):
    vertices = []
    normals = []
    uvs = []
    bone_weights = []
    bone_indices = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 5):
            # Read vertex position
            vertex_line = lines[i].strip().split(',')
            vertices.append(list(map(float, vertex_line)))
            # Read normal position
            normal_line = lines[i + 1].strip().split(',')
            normals.append(list(map(float, normal_line)))
            # Read UV coordinates
            uv_line = lines[i + 2].strip().split(',')
            uvs.append(list(map(float, uv_line)))
            # Read bone weights
            weight_line = lines[i + 3].strip().split(',')
            bone_weights.append(list(map(float, weight_line)))
            # Read bone indices
            index_line = lines[i + 4].strip().split(',')
            bone_indices.append(list(map(int, index_line)))

    # Convert the lists to PyTorch tensors
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    normals_tensor = torch.tensor(normals, dtype=torch.float32)
    uvs_tensor = torch.tensor(uvs, dtype=torch.float32)
    bone_weights_tensor = torch.tensor(bone_weights, dtype=torch.float32)
    bone_indices_tensor = torch.tensor(bone_indices, dtype=torch.int64)

    return vertices_tensor, normals_tensor, uvs_tensor, bone_weights_tensor, bone_indices_tensor


def read_bone_joints(file_path):
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


def read_bone_joint_mats(file_path):
    mats = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines)):
            mat_line = lines[i].strip().split(',')
            mats.append(list(map(float, mat_line)))

    # Convert the lists to PyTorch tensors
    mats_tensor = torch.tensor(mats, dtype=torch.float32).reshape([-1, 4, 4])

    return mats_tensor


def read_bone_parent_indices(file_path):
    parent_indices = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Read the parent index and convert it to an integer
            parent_index = int(line.strip())
            parent_indices.append(parent_index)

    # Convert the list to a PyTorch tensors
    parent_indices_tensor = torch.tensor(parent_indices, dtype=torch.int32)

    return parent_indices_tensor


class GalaModel(nn.Module):
    """
    Attributes:
        parents (list[J]) : Indicate the parent joint for each joint, -1 for root joint.
        bone_count (int) : The count of joints including root joint, i.e. J.
        joints (torch.Tensor[J-1, 3]) : The translations of each joint relative to the parent joint, except for root joint.
        tpose_w2l_mats (torch.Tensor[J, 3]) : The skeleton to local transform matrixes for each joint.
    """

    def __init__(self, model_path,
                 max_sh_degree=0,
                 max_freq=4,
                 num_players=1,
                 use_point_color=False,
                 use_point_displacement=False,
                 enable_ambient_occlusion=False,
                 encoder_type="hash"):
        """
        Init joints and offset matrixes from files.

        Args:
            model_path (str) : The path to the folder that holds the vertices, tpose matrix, binding weights and indexes.
            num_players (int) : Number of players.
        """
        super().__init__()

        self.model_path = model_path
        self.num_players = num_players
        self.enable_ambient_occlusion = enable_ambient_occlusion
        self.use_point_displacement = use_point_displacement
        self.use_point_color = use_point_color
        self.encoder_type = encoder_type

        model_file = os.path.join(model_path, "mesh.txt")
        vertices, normals, uvs, bone_weights, bone_indices = read_skinned_mesh_data(
            model_file)

        tpose_file = os.path.join(model_path, "tpose.txt")
        tpose_positions, tpose_rotations, tpose_scales = read_bone_joints(
            tpose_file)

        tpose_mat_file = os.path.join(model_path, "tpose_mat.txt")
        tpose_w2l_mats = read_bone_joint_mats(tpose_mat_file)

        joint_parent_file = os.path.join(model_path, "jointParent.txt")
        self.joint_parent_idx = read_bone_parent_indices(joint_parent_file)

        self.bone_count = tpose_positions.shape[0]
        self.vertex_count = vertices.shape[0]

        print("mesh loaded:")
        print("total vertices: " + str(vertices.shape[0]))
        print("num of joints: " + str(self.bone_count))

        self.register_buffer('v_template', vertices[None, ...].repeat(
            [self.num_players, 1, 1]))
        uvs = uvs * 2. - 1.
        self.register_buffer('uvs', uvs[None, ...].repeat(
            [self.num_players, 1, 1]))

        bone_weights = torch.Tensor(
            np.load(os.path.join(model_path, "weights.npy")))[None, ...].repeat([self.num_players, 1, 1])
        self.register_buffer("bone_weights", bone_weights)
        # self.bone_weights = nn.Parameter(
        #     bone_weights)

        self.J = nn.Parameter(
            tpose_positions[None, ...].repeat([self.num_players, 1, 1]))
        self.tpose_w2l_mats = tpose_w2l_mats

        minmax = [self.v_template[0].min(
            dim=0).values * 1.05,  self.v_template[0].max(dim=0).values * 1.05]
        self.register_buffer('normalized_vertices',
                             (self.v_template - minmax[0]) / (minmax[1] - minmax[0]))

        dist2 = torch.clamp_min(
            distCUDA2(vertices.float().cuda()), 0.0000001)[..., None].repeat([1, 3])

        if use_point_displacement:
            self.displacements = nn.Parameter(
                torch.zeros_like(self.v_template))
        else:
            self.displacementEncoder = DisplacementEncoder(
                encoder=encoder_type, num_players=num_players)
        n = self.v_template.shape[1] * num_players
        if use_point_color:
            self.shs_dc = nn.Parameter(torch.zeros(
                [n, 1, 3]))
            self.shs_rest = nn.Parameter(torch.zeros(
                [n, (max_sh_degree + 1) ** 2 - 1, 3]))
        else:
            self.shEncoder = SHEncoder(max_sh_degree=max_sh_degree,
                                       encoder=encoder_type, num_players=num_players)
        self.opacity = nn.Parameter(inverse_sigmoid(
            0.2 * torch.ones((n, 1), dtype=torch.float)))
        self.scales = nn.Parameter(
            torch.log(torch.sqrt(dist2)).repeat([num_players, 1]))
        rotations = torch.zeros([n, 4])
        rotations[:, 0] = 1
        self.rotations = nn.Parameter(rotations)

        if enable_ambient_occlusion:
            self.aoEncoder = AOEncoder(
                encoder=encoder_type, max_freq=max_freq, num_players=num_players)
        self.register_buffer("aos", torch.ones_like(self.opacity))

    def configure_optimizers(self, training_args):
        l = [
            # {'params': [self.bone_weights],
            #     'lr': training_args.weights_lr, "name": "weights"},
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

    def forward(self, body_pose, global_orient, transl, time, is_use_ao):
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

        # root_mats = matrix_batch_TRS(root_transls,global_orient,root_scales)

        full_body_pose = torch.cat(
            [global_orient[:, None, :], body_pose], dim=1).cpu()

        transl = transl.cpu()
        J = self.J.cpu()

        T = torch.empty([self.num_players, self.vertex_count,
                        3, 4], device=self.v_template.device)
        joint_deform_mats = torch.empty([self.num_players, self.bone_count,
                                         3, 4], device=self.v_template.device)
        for i in range(self.num_players):
            # local to world transform
            joint_local_mats = self.batch_rodrigues(
                full_body_pose[i].view(-1, 3)).view([-1, 4, 4])

            joint_local_mats[:, :3, 3] = J[i]
            joint_local_mats[0, :3, 3] = transl[i]

            joint_world_mats = []
            joint_world_mats.append(joint_local_mats[0])

            for j in range(1, self.bone_count):
                mat = joint_world_mats[self.joint_parent_idx[j]
                                       ] @  joint_local_mats[j]
                joint_world_mats.append(mat)
            joint_world_mats = torch.stack(joint_world_mats, dim=0)

            # canonical to world transform
            joint_deform_mats[i] = (
                joint_world_mats @ self.tpose_w2l_mats).to(self.v_template.device)[:, :3, :4]

        T = torch.sum(
            self.bone_weights[..., None, None] * joint_deform_mats[:, None, ...], dim=2)

        if self.use_point_color:
            shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        else:
            if self.encoder_type == "hash":
                shs = self.shEncoder(self.normalized_vertices)
            else:
                shs = self.shEncoder(self.uvs)

        # print(is_use_ao)
        if self.enable_ambient_occlusion and is_use_ao:
            if self.encoder_type == "hash":
                aos = self.aoEncoder(self.normalized_vertices, time)
            else:
                aos = self.aoEncoder(self.uvs, time)
        else:
            aos = self.aos

        if self.use_point_displacement:
            v_displaced = self.v_template + self.displacements
        else:
            if self.encoder_type == "hash":
                displacement = self.displacementEncoder(
                    self.normalized_vertices)
            else:
                displacement = self.displacementEncoder(self.uvs)
            v_displaced = self.v_template + displacement

        return v_displaced.reshape([-1, 3]), \
            torch.sigmoid(self.opacity), \
            torch.exp(self.scales), \
            torch.nn.functional.normalize(self.rotations), \
            shs, aos, T.reshape([-1, 3, 4])

    def rigid_transform(self, body_pose_mat, transl, J):
        """
        Caculate local to world transforms of joints for specific body pose.
        """
        bone_transform = body_pose_mat
        bone_transform[:, :3, 3] = J
        bone_transform[0, :3, 3] = transl

        transform_chain = [bone_transform[0]]
        for i in range(1, len(self.parents)):
            curr_res = torch.matmul(transform_chain[self.parents[i]],
                                    bone_transform[i])
            transform_chain.append(curr_res)
        transforms = torch.stack(transform_chain, dim=0)

        return transforms

    def batch_rodrigues(self, euler_angles):
        """
        Convert euler angles to rotation matrixes.

        Args:
            euler_angles (torch.Tensor[N, 3]) : Rotate angles.

        Returns:
            rotation_matrix (torch.Tensor[N, 4, 4]) : Rotation matrixes.
        """
        batch_size = euler_angles.shape[0]
        rotation_matrix = torch.zeros(
            (batch_size, 4, 4), device=euler_angles.device, dtype=euler_angles.dtype)
        x = euler_angles[:, 0] / 180 * torch.pi
        y = euler_angles[:, 1] / 180 * torch.pi
        z = euler_angles[:, 2] / 180 * torch.pi
        c1, c2, c3 = torch.cos(y), torch.cos(x), torch.cos(z)
        s1, s2, s3 = torch.sin(y), torch.sin(x), torch.sin(z)

        rotation_matrix[:, 0, 0] = c1 * c3 + s1 * s2 * s3
        rotation_matrix[:, 0, 1] = c3 * s1 * s2 - c1 * s3
        rotation_matrix[:, 0, 2] = c2 * s1
        rotation_matrix[:, 1, 0] = c2 * s3
        rotation_matrix[:, 1, 1] = c2 * c3
        rotation_matrix[:, 1, 2] = -s2
        rotation_matrix[:, 2, 0] = c1 * s2 * s3 - c3 * s1
        rotation_matrix[:, 2, 1] = c1 * c3 * s2 + s1 * s3
        rotation_matrix[:, 2, 2] = c1 * c2
        rotation_matrix[:, 3, 3] = 1
        return rotation_matrix
