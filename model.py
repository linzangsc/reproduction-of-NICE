import torch
import torch.nn as nn

class NICE(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(NICE, self).__init__()
        self.device = device
        self.addictive_coupling_layer1 = AddictiveCouplingLayer(input_channel, batch_size, device)
        self.addictive_coupling_layer2 = AddictiveCouplingLayer(input_channel, batch_size, device)
        self.addictive_coupling_layer3 = AddictiveCouplingLayer(input_channel, batch_size, device)
        self.addictive_coupling_layer4 = AddictiveCouplingLayer(input_channel, batch_size, device)
        self.log_scale_factor = nn.Parameter(torch.ones(input_channel))

    def reverse_mapping(self, z):
        # z->x
        # shape of z: (bs, c, h, w)
        batch_size, channel, height, weight = z.shape
        z = z.reshape(batch_size, -1)
        x = torch.matmul(z, torch.diag(torch.reciprocal(torch.exp(self.log_scale_factor))))
        x = self.addictive_coupling_layer4.reverse_mapping(x)
        x = self.addictive_coupling_layer3.reverse_mapping(x)
        x = self.addictive_coupling_layer2.reverse_mapping(x)
        x = self.addictive_coupling_layer1.reverse_mapping(x)
        x = x.reshape(batch_size, channel, height, weight)
        return x
    
    def get_det_J(self):
        det_J = torch.sum(self.log_scale_factor)
        return det_J

    def forward(self, x):
        # x->z
        # shape of x: (bs, c, h, w)
        batch_size, channel, height, weight = x.shape
        x = x.reshape(batch_size, -1)
        x = self.addictive_coupling_layer1(x)
        x = self.addictive_coupling_layer2(x)
        x = self.addictive_coupling_layer3(x)
        x = self.addictive_coupling_layer4(x)
        z = torch.matmul(x, torch.diag(torch.exp(self.log_scale_factor)))
        z = z.reshape(batch_size, channel, height, weight)
        return z

class ShuffleLayer(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(ShuffleLayer, self).__init__()
        self.input_channel = input_channel
        self.batch_size = batch_size
        self.shuffle_matrix = torch.flip(torch.eye(input_channel), dims=[0]).to(device)

    def forward(self, input):
        return torch.matmul(input, self.shuffle_matrix)

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
        )
        self.shuffle_layer = ShuffleLayer(input_channel, batch_size, device)
        
    def reverse_mapping(self, z):
        # z->x
        # shape of z: (bs, c)
        z_1 = z[:, :self.input_channel//2]
        z_2 = z[:, self.input_channel//2:]
        bias = self.mlp(z_1)
        x = torch.cat([z_1, z_2 - bias], dim=-1)
        x = self.shuffle_layer(x)
        return x
    
    def forward(self, x):
        # x->z
        batch_size, channel = x.shape
        x = self.shuffle_layer(x)
        x_1 = x[:, :self.input_channel//2]
        x_2 = x[:, self.input_channel//2:]
        bias = self.mlp(x_1)
        z = torch.cat([x_1, x_2 + bias], dim=-1)

        return z
