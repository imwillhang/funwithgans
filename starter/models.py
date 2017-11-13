import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as vision

from DataMaster import Batcher
from torch.autograd import Variable
from torch.nn import init

class BasicConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        nn.init.kaiming_uniform(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        nn.init.kaiming_uniform(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Generator(nn.Module):
	def __init__(self, config):
		super(Generator, self).__init__()
		self.net = nn.Sequential(
				BasicConvTranspose2d(32, 512, kernel_size=3, stride=2, padding=0),
				BasicConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
				BasicConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
				BasicConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
				BasicConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
				BasicConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
				BasicConvTranspose2d(64, 3, kernel_size=2, stride=1, padding=1),
				nn.Tanh()
			)

	def forward(self, x):
		return self.net(x)

class Discriminator(nn.Module):
	def __init__(self, config):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(
				BasicConv2d(3, 64, kernel_size=11, stride=4, padding=2),
				nn.MaxPool2d(kernel_size=3, stride=2),
				BasicConv2d(64, 192, kernel_size=5, padding=2),
				nn.MaxPool2d(kernel_size=3, stride=2),
				BasicConv2d(192, 384, kernel_size=3, padding=1),
				nn.MaxPool2d(kernel_size=3, stride=2),
			)
		self.fc = nn.Linear(384, 64)
		self.readout = nn.Linear(64, 1)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		x = self.net(x)
		size = x.size()
		x = x.view(-1, size[1] * size[2] * size[3])
		x = self.dropout(F.relu(self.fc(x), inplace=True))
		return self.readout(x)

def build_and_train(config):
	G = Generator(config).cuda()
	D = Discriminator(config).cuda()

	config.G_optim = optim.Adam(G.parameters(), lr=config.lr)
	config.D_optim = optim.Adam(D.parameters(), lr=config.lr)

	batcher = Batcher()

	for i in range(config.epochs):
		run_epoch(G, D, batcher, config)

def generate_data(config):
	z = Variable(torch.randn(config.batch_sz, config.z_dim, 1, 1))
	return z

def process_data(X):
	X = np.array(X)
	X = np.transpose(X, [0, 3, 1, 2])
	X = Variable(torch.from_numpy(X))
	return X

# this code adapted from wiseodd's WGAN tutorial
# 	https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py
def run_epoch(G, D, batcher, config):
	i = 0
	for X in batcher.get_data(config.batch_sz):
		for i in range(config.d_train):
			X = process_data(X)
			z = generate_data(config)

			G_fake = G(z)
			D_fake = D(G_fake)
			D_real = D(X)

			D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

			config.D_optim.zero_grad()
			D_loss.backward()
			config.D_optim.step()

			for param in D.parameters():
				param.data.clamp_(-config.K, config.K)

		z = generate_data(config)
		G_fake = G(z)
		D_fake = D(G_fake)

		G_loss = -torch.mean(D_fake)

		config.G_optim.zero_grad()
		G_loss.backward()
		config.G_optim.step()

		i += 1

		if it % 1000 == 0:
			print('Iteration - {} | D_loss: {} | G_loss: {}'
				.format(it, D_loss.data.numpy(), G_loss.data.numpy()))

			samples = G(z).data.numpy()[:16]

			for i, sample in enumerate(samples):
				np.save('outputs/{}'.format(i), sample)

			# fig = plt.figure(figsize=(4, 4))
			# gs = gridspec.GridSpec(4, 4)
			# gs.update(wspace=0.05, hspace=0.05)

			# for i, sample in enumerate(samples):
			# 	ax = plt.subplot(gs[i])
			# 	plt.axis('off')
			# 	ax.set_xticklabels([])
			# 	ax.set_yticklabels([])
			# 	ax.set_aspect('equal')
			# 	plt.imshow(sample.reshape(72, 72, 3))



