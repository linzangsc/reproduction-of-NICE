import torch
import torch.nn as nn

class NICE(nn.Module):
    def __init__(self, input_channel, batch_size, device, num_blocks=4):
        super(NICE, self).__init__()
        self.device = device
        self.num_blocks = num_blocks+1
        self.forward_mapping = nn.Sequential()
        for i in range(num_blocks):
            self.forward_mapping.append(AddictiveCouplingLayer(input_channel, batch_size, device))
        self.scale_layer = ScaleLayer(input_channel, batch_size, device)
        self.forward_mapping.append(self.scale_layer)

    def forward(self, z):
        # shape of z: (bs, c, h, w)
        batch_size, channel, height, weight = z.shape
        z = z.reshape(batch_size, -1)
        x = z
        x = self.forward_mapping(x)
        x = x.reshape(batch_size, channel, height, weight)
        return x
    
    def get_det_J(self):
        det_J = torch.sum(self.forward_mapping[-1].log_scale_factor)
        return det_J

    def reverse_mapping(self, x):
        # shape of x: (bs, c, h, w)
        batch_size, channel, height, weight = x.shape
        x = x.reshape(batch_size, -1)
        z = x
        for i in range(self.num_blocks-1, -1, -1):
            z = self.forward_mapping[i].reverse_mapping(z)
        z = z.reshape(batch_size, channel, height, weight)
        return z

class ScaleLayer(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(ScaleLayer, self).__init__()
        self.batch_size = batch_size
        self.log_scale_factor = torch.randn(input_channel)
        self.log_scale_factor = nn.Parameter(self.log_scale_factor)
    
    def forward(self, z):
        batch_size, input_channel = z.shape
        scale_factor = torch.exp(self.log_scale_factor.repeat(batch_size, 1))
        return z * scale_factor

    def reverse_mapping(self, x):
        batch_size, input_channel = x.shape
        scale_factor = torch.exp(self.log_scale_factor.repeat(batch_size, 1))
        return x / scale_factor

class ShuffleLayer(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(ShuffleLayer, self).__init__()
        self.input_channel = input_channel
        self.batch_size = batch_size
        self.shuffle_matrix = torch.eye(input_channel)[torch.randperm(input_channel)].to(device)
        self.inverse_matrix = torch.inverse(self.shuffle_matrix)
    
    def forward(self, z):
        return torch.matmul(z, self.shuffle_matrix)

    def reverse_mapping(self, x):
        return torch.matmul(x, self.inverse_matrix)

class AddictiveCouplingLayer(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(AddictiveCouplingLayer, self).__init__()
        self.batch_size = batch_size
        self.input_channel = input_channel
        self.mlp = nn.Sequential(
            nn.Linear(input_channel//2, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, input_channel//2),
            nn.ReLU(),
        )
        self.shuffle_layer = ShuffleLayer(input_channel, batch_size, device)
        
    def forward(self, z):
        # shape of z: (bs, c)
        z = self.shuffle_layer(z)
        z_1 = z[:, :self.input_channel//2]
        z_2 = z[:, self.input_channel//2:]
        bias = self.mlp(z_1)
        x = torch.cat([z_1, z_2 + bias], dim=-1)
        return x
    
    def reverse_mapping(self, x):
        # x -> z
        batch_size, channel = x.shape
        x_1 = x[:, :self.input_channel//2]
        x_2 = x[:, self.input_channel//2:]
        bias = self.mlp(x_1)
        z = torch.cat([x_1, x_2 - bias], dim=-1)
        z = self.shuffle_layer.reverse_mapping(z)
        return z
