import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import CustomizedDataset, visualize_binary_result, sigmoid, visualize_float_result, visualize_latent_space
from model import NICE
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.image_size = config['image_size']
        self.logger = SummaryWriter(self.config['log_path'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.dataset = CustomizedDataset()
        self.train_dataset = self.dataset.train_dataset
        self.test_dataset = self.dataset.test_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                       batch_size=self.batch_size, shuffle=False)
        input_channel = self.image_size*self.image_size
        self.model = NICE(input_channel=input_channel, batch_size=self.batch_size,
                          device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

    def loss(self, z):
        log_det_J = self.model.get_det_J()
        log_p_z = -torch.sum(0.5 * z**2, dim=[1, 2, 3])
        loss = -(log_det_J + log_p_z.mean())
        
        return loss

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # translate to binary images
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                z = self.model.reverse_mapping(images)
                loss = self.loss(z)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], loss: {loss.item():.4f}')
                self.logger.add_scalar('loss/train', loss.item(), i + epoch * len(self.train_loader))
            self.save_model(self.config['ckpt_path'])
            with torch.no_grad():
                z = torch.randn((16, 1, self.image_size, self.image_size)).to(self.device)
                sample_image = self.model(z)
                self.visualize_samples(sample_image, epoch)
                # self.visualize_latent_space(epoch)

    def save_model(self, output_path):
        if not os.path.exists(output_path): os.mkdir(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, f"model.pth"))

    def visualize_samples(self, sample_images, epoch):
        sample_images = sample_images.reshape(sample_images.shape[0], self.image_size, self.image_size).to('cpu')
        npy_sampled_theta = np.array(sample_images)
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        axs = visualize_float_result(npy_sampled_theta, axs)
        self.logger.add_figure(f"sample results", plt.gcf(), epoch)
        plt.close(fig)

    def visualize_latent_space(self, epoch):
        fig, ax = plt.subplots()
        for (images, labels) in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, _, _, latents = self.model(images)
            ax = visualize_latent_space(latents, labels, ax)
        plt.colorbar(ax.collections[0], ax=ax)
        self.logger.add_figure(f"latent space", plt.gcf(), epoch)
        plt.close(fig)

if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config=config)
    trainer.train()
