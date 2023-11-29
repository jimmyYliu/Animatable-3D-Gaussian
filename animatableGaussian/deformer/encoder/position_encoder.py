import tinycudann as tcnn
import torch
import torch.nn.functional as F
import torch.nn as nn


class UVEncoder(nn.Module):
    def __init__(self, num_channels, resolution=512, num_players=1):
        super().__init__()
        self.num_channels = num_channels
        self.num_players = num_players
        self.feature_map = nn.Parameter(torch.zeros(
            [num_players, num_channels, resolution, resolution]))

    def forward(self, uvs):
        return F.grid_sample(self.feature_map, uvs[:, :, None, :], align_corners=False).transpose(1, 3).reshape(
            [self.num_players, -1, self.num_channels])


class TriPlaneEncoder(nn.Module):
    def __init__(self, num_channels, num_features=16, resolution=512, num_players=1):
        super().__init__()
        self.networks = []
        self.num_players = num_players
        self.num_features = num_features
        self.xy_plane = nn.Parameter(torch.randn(
            [num_players, num_features, resolution, resolution]) * 0.001)
        self.yz_plane = nn.Parameter(torch.randn(
            [num_players, num_features, resolution, resolution]) * 0.001)
        self.xz_plane = nn.Parameter(torch.randn(
            [num_players, num_features, resolution, resolution]) * 0.001)
        for i in range(num_players):
            self.networks.append(tcnn.Network(
                n_input_dims=3 * num_features,
                n_output_dims=num_channels,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 0,
                }
            ))
        self.networks = nn.ModuleList(self.networks)

    def forward(self, x):
        self.outputs = []
        xy = F.grid_sample(self.xy_plane, x[:, :, None, :2], align_corners=False).transpose(1, 3).reshape(
            [self.num_players, -1, self.num_features])
        yz = F.grid_sample(self.yz_plane, x[:, :, None, 1:3], align_corners=False).transpose(1, 3).reshape(
            [self.num_players, -1, self.num_features])
        xz = F.grid_sample(self.xz_plane, x[:, :, None, [0, 2]], align_corners=False).transpose(1, 3).reshape(
            [self.num_players, -1, self.num_features])
        features = torch.cat([xy, yz, xz], dim=2)
        for i in range(self.num_players):
            self.outputs.append(self.networks[i](features[i]))
        return torch.stack(self.outputs).float()


class HashEncoder(nn.Module):
    def __init__(self, num_channels, num_players=1):
        super().__init__()
        self.networks = []
        self.num_players = num_players
        for i in range(num_players):
            self.networks.append(tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=num_channels,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": 17,
                    "base_resolution": 4,
                    "per_level_scale": 1.5,
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            ))
        self.networks = nn.ModuleList(self.networks)

    def forward(self, x):
        self.outputs = []
        for i in range(self.num_players):
            self.outputs.append(self.networks[i](x[i]))
        return torch.stack(self.outputs).float()


class SHEncoder(nn.Module):
    def __init__(self, max_sh_degree=1, encoder="uv", num_players=1):
        super().__init__()
        self.num_players = num_players
        self.num_shs = (max_sh_degree + 1) ** 2
        if encoder == "uv":
            self.input_channels = 2
            self.encoder = UVEncoder(
                self.num_shs * 3, num_players=num_players)
        elif encoder == "hash":
            self.input_channels = 3
            self.encoder = HashEncoder(
                self.num_shs * 3, num_players=num_players)
        elif encoder == "triplane":
            self.input_channels = 3
            self.encoder = TriPlaneEncoder(
                self.num_shs * 3, num_players=num_players)
        else:
            raise Exception("encoder does not exist!")

    def forward(self, input):
        if input.shape[0] != self.num_players or input.shape[2] != self.input_channels:
            raise Exception("input shape is not allowed")
        shs = self.encoder(input).reshape([-1, self.num_shs, 3])
        shs[:, 0, :] *= 20
        return shs


class DisplacementEncoder(nn.Module):
    def __init__(self, encoder="uv", num_players=1):
        super().__init__()
        self.num_players = num_players
        if encoder == "uv":
            self.input_channels = 2
            self.encoder = UVEncoder(
                3, num_players=num_players)
        elif encoder == "hash":
            self.input_channels = 3
            self.encoder = HashEncoder(
                3, num_players=num_players)
        elif encoder == "triplane":
            self.input_channels = 3
            self.encoder = TriPlaneEncoder(
                3, num_players=num_players)
        else:
            raise Exception("encoder does not exist!")

    def forward(self, input):
        if input.shape[0] != self.num_players or input.shape[2] != self.input_channels:
            raise Exception("input shape is not allowed")
        return self.encoder(input)
