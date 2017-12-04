import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as vision

#from DataMaster import Batcher
from torch.autograd import Variable
from torch.nn import init

from batcher import *

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
    def __init__(self, config, sizes):
        super(Generator, self).__init__()
        self.conv_fe = []
        self.lstm_fe = []
        self.hidden_sz = 32
        for fe_name in sizes:
            fe_ = nn.Sequential(
                BasicConv2d(sizes[fe_name], 16, kernel_size=(7, 1), stride=3),
                BasicConv2d(16, 16, kernel_size=(5, 1), stride=2),
            )
            self.conv_fe.append(fe_)
        for fe_name in sizes:
            fe_ = nn.Sequential(
                nn.LSTM(
                    input_size=16, 
                    hidden_size=self.hidden_sz,
                    num_layers=3,
                    dropout=0.3,
                    bidirectional=True
                )
            )
            self.lstm_fe.append(fe_)
        self.lstm_fe = nn.ModuleList(self.lstm_fe)
        self.conv_fe = nn.ModuleList(self.conv_fe)
        self.fc = nn.Linear(len(sizes) * self.hidden_sz * 2, self.hidden_sz)
        self.cm_gen_1 = nn.LSTMCell(
            input_size=self.hidden_sz, 
            hidden_size=self.hidden_sz
        )
        self.cm_gen_2 = nn.LSTMCell(
            input_size=self.hidden_sz, 
            hidden_size=self.hidden_sz
        )
        self.transform = nn.Linear(self.hidden_sz, 1)
        self.refine_conv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        features = []
        seqlen = x[0].size()[0]
        for index, feature in enumerate(x):
            size = feature.size()
            feature = feature.view(1, size[-1], size[0], 1)
            feature = self.conv_fe[index](feature)
            feature = torch.transpose(torch.squeeze(feature), 1, 0)
            feature = feature.unsqueeze(1)
            feature = self.lstm_fe[index](feature)[0][-1, -1, :].view(64, 1)
            features.append(feature)
        features = torch.cat(tuple(features))
        features = features.view(-1, features.size()[0])
        proj = self.fc(features)
        vec1, vec2 = [], []
        hx1, cx1 = Variable(torch.randn(1, self.hidden_sz)), Variable(torch.randn(1, self.hidden_sz))
        hx2, cx2 = Variable(torch.randn(1, self.hidden_sz)), Variable(torch.randn(1, self.hidden_sz))
        if torch.cuda.is_available():
            hx1 = hx1.cuda()
            hx2 = hx2.cuda()
            cx1 = cx1.cuda()
            cx2 = cx2.cuda()
        input1, input2 = proj, proj
        for i in range(seqlen):
            hx1, cx1 = self.cm_gen_1(input1, (hx1, cx1))
            hx2, cx2 = self.cm_gen_2(input2, (hx2, cx2))
            input1, input2 = hx1, hx2
            vec1.append(self.transform(hx1))
            vec2.append(self.transform(hx2))
        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)
        cm = torch.mm(vec1, vec2.transpose(0, 1))
        cm = self.refine_conv(cm.view(1, 1, *cm.size()))
        print(cm.size())
        return cm

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.hidden_sz = 32
        self.conv_fe = nn.Sequential(
                BasicConv2d(1, 64, kernel_size=7, stride=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                BasicConv2d(64, 64, kernel_size=5, stride=2, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            )
        self.disc = nn.LSTM(
            input_size=64, 
            hidden_size=self.hidden_sz,
            num_layers=3,
            dropout=0.3,
            bidirectional=True
        )
        self.fc = nn.Linear(self.hidden_sz * 2, 2)

    def forward(self, x):
        x = self.conv_fe(x)
        size = x.size()
        x = x.view(size[2] * size[3], 1, size[1])
        x = self.disc(x)[0][-1, -1, :].unsqueeze(0)
        x = self.fc(x)
        return x

def save_checkpoint(state, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)

def build_and_train(config):
    sizes = {
        'ACC': 3,
        'DISO': 1,
        'SS3': 3,
        'SS8': 8,
        'PSSM': 20,
        'PSFM': 20
    }
    G = Generator(config, sizes)
    D = Discriminator(config)

    if torch.cuda.is_available():
        G = G.cuda()
        D = D.cuda()

    config.G_optim = optim.RMSprop(G.parameters(), lr=config.lr)
    config.D_optim = optim.RMSprop(D.parameters(), lr=config.lr)

    train_data = gen_dataset('valid')
    batcher = get_batch(train_data, 1)

    for i in range(config.epochs):
        run_epoch(G, D, batcher, i, config)
        save_checkpoint(model)

def generate_data(config):
    z = Variable(torch.randn(config.batch_sz, config.z_dim, 2, 2))
    if torch.cuda.is_available():
        z = z.cuda()
    return z

def process_data(X):
    X = X[-1]
    data = [Variable(torch.from_numpy(np.expand_dims(X[key], 1))) for key in ['ACC', 'DISO', 'SS3', 'SS8', 'PSSM', 'PSFM']]
    cm = Variable(torch.from_numpy(X['contactMatrix'].astype(np.float32)))
    cm = cm.view(1, 1, *cm.size())
    if torch.cuda.is_available():
        data = [d.cuda() for d in data]
        cm = cm.cuda()
    return data, cm

# this code adapted from wiseodd's WGAN tutorial
#   https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py
def run_epoch(G, D, batcher, epoch, config):
    for it in range(10000):
        G.train()

        for i in range(config.d_train):
            X = next(batcher)
            X, y = process_data(X)
            #z = generate_data(config)

            G_fake = G(X)
            D_fake = D(G_fake)
            D_real = D(y)

            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            config.D_optim.zero_grad()
            D_loss.backward()
            config.D_optim.step()
            G.zero_grad()
            D.zero_grad()

            for param in D.parameters():
                param.data.clamp_(-config.K, config.K)

        X = next(batcher)
        X, y = process_data(X)
        G_fake = G(X)
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
            file = open('metadata.txt', 'w')
            for i in range(2):
                X_sample = next(batcher)
                X, y = process_data(X_sample)
                sample = G(X).data.cpu().numpy()
                reference = y
                file.write('{}\n'.format(X_sample['sequence']))
                np.save('outputs/sample_{}'.format(i), sample)
                np.save('outputs/reference_{}'.format(i), reference)

            torch.save(G, 'models/shit.pth.tar')

            # fig = plt.figure(figsize=(4, 4))
            # gs = gridspec.GridSpec(4, 4)
            # gs.update(wspace=0.05, hspace=0.05)

            # for i, sample in enumerate(samples):
            #   ax = plt.subplot(gs[i])
            #   plt.axis('off')
            #   ax.set_xticklabels([])
            #   ax.set_yticklabels([])
            #   ax.set_aspect('equal')
            #   plt.imshow(sample.reshape(72, 72, 3))
