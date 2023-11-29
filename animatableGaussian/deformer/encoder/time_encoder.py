import tinycudann as tcnn
import torch
import torch.nn.functional as F
import torch.nn as nn


class UVTimeEncoder(nn.Module):
    def __init__(self, num_channels, time_dim=9, feature_dim=55, resolution=128, num_players=1):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        self.num_players = num_players
        self.feature_map = nn.Parameter(torch.zeros(
            [num_players, feature_dim, resolution, resolution]))
        self.time_nets = []
        for i in range(num_players):
            self.time_nets.append(tcnn.Network(
                n_input_dims=feature_dim + time_dim,
                n_output_dims=num_channels,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            ))

    def forward(self, uvs, t):
        features = F.grid_sample(self.feature_map, uvs[:, :, None, :], align_corners=False).transpose(1, 3).reshape(
            [self.num_players, -1, self.feature_dim])
        self.outputs = []
        for i in range(self.num_players):
            self.outputs.append(self.time_nets[i](
                torch.cat([features[i], t[None, :].repeat(features.shape[1], 1)], dim=-1)).float())
        return torch.stack(self.outputs)


class HashTimeEncoder(nn.Module):
    def __init__(self, num_channels, time_dim=9, num_players=1):
        super().__init__()
        self.networks = []
        self.time_nets = []
        self.num_players = num_players
        for i in range(num_players):
            self.networks.append(tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": 19,
                    "base_resolution": 4,
                    "per_level_scale": 1.5,
                },
            ))
            self.time_nets.append(tcnn.Network(
                n_input_dims=self.networks[i].n_output_dims + time_dim,
                n_output_dims=num_channels,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            ))
        self.networks = nn.ModuleList(self.networks)
        self.time_nets = nn.ModuleList(self.time_nets)

    def forward(self, x, t):

        self.outputs = []
        for i in range(self.num_players):
            features = self.networks[i](x[i])
            self.outputs.append(self.time_nets[i](
                torch.cat([features, t[None, :].repeat(features.shape[0], 1)], dim=-1)))
        return torch.stack(self.outputs).float()


class AOEncoder(nn.Module):
    def __init__(self, encoder="uv", num_players=1, max_freq=4):
        super().__init__()
        self.num_players = num_players
        self.max_freq = max_freq
        if encoder == "uv":
            self.input_channels = 2
            self.encoder = UVTimeEncoder(
                1, num_players=num_players, time_dim=max_freq*2 + 1)
        elif encoder == "hash":
            self.input_channels = 3
            self.encoder = HashTimeEncoder(
                1, num_players=num_players, time_dim=max_freq*2 + 1)
        else:
            raise Exception("encoder does not exist!")

    def forward(self, input, time):
        if input.shape[0] != self.num_players or input.shape[2] != self.input_channels:
            raise Exception("input shape is not allowed")
        return torch.sigmoid(self.encoder(input, time).reshape([-1, 1]))
