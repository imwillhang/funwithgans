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
				BasicConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=1, output_padding=1),
				BasicConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
				BasicConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1),
				BasicConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
				BasicConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
				BasicConvTranspose2d(8, 8, kernel_size=3, stride=1, padding=1),
				BasicConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
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

def save_checkpoint(state, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)

def build_and_train(config):
	G = Generator(config).cuda()
	D = Discriminator(config).cuda()

	config.G_optim = optim.RMSprop(G.parameters(), lr=config.lr)
	config.D_optim = optim.RMSprop(D.parameters(), lr=config.lr)

	batcher = Batcher()

	for i in range(config.epochs):
		run_epoch(G, D, batcher, i, config)
		save_checkpoint(model)

def generate_data(config):
	z = Variable(torch.randn(config.batch_sz, config.z_dim, 2, 2)).cuda()
	return z

def process_data(X):
	X = np.transpose(X, [0, 3, 1, 2]).astype(np.float32)
	X = (X / 255.)
	X = Variable(torch.from_numpy(X)).cuda()
	return X

# this code adapted from wiseodd's WGAN tutorial
# 	https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py
def run_epoch(G, D, batcher, epoch, config):
	data = batcher.get_data(config.batch_sz)
	for it in range(10000):
		G.train()

		for i in range(config.d_train):
			X = next(data)
			X = process_data(X)
			z = generate_data(config)

			G_fake = G(z)
			D_fake = D(G_fake)
			D_real = D(X)

			D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

			config.D_optim.zero_grad()
			D_loss.backward()
			config.D_optim.step()
			G.zero_grad()
			D.zero_grad()

			for param in D.parameters():
				param.data.clamp_(-config.K, config.K)

		z = generate_data(config)
		G_fake = G(z)
		D_fake = D(G_fake)

		G_loss = -torch.mean(D_fake)

		config.G_optim.zero_grad()
		G_loss.backward()
		config.G_optim.step()
		G.zero_grad()
		D.zero_grad()

		if it % 100 == 0:
			print('Epoch - {} | Iteration - {} | D_loss: {} | G_loss: {}'
				.format(epoch, it, D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy()))

			G.eval()
			samples = G(z).data.cpu().numpy()[:16]

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