import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as vision
import copy

#from DataMaster import Batcher
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import init
import matplotlib.pyplot as plt


from batcher import *

class BasicConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        nn.init.kaiming_uniform(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        #x = self.dropout(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        nn.init.kaiming_uniform(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        #x = self.dropout(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Encode(nn.Module):
    def __init__(self, config, size):
        super(Encode, self).__init__()
        self.conv1 = BasicConv2d(1, 16, kernel_size=(5, size), stride=1)
        self.conv2 = BasicConv2d(16, 16, kernel_size=(5, 1), stride=1)
        self.conv3 = BasicConv2d(16, 16, kernel_size=(5, 1), stride=1)
        self.maxpool = nn.MaxPool2d((2, 1), 2, return_indices=True)

    def forward(self, x):
        indices = []
        sizes = []
        layers = [self.conv1, self.conv2, self.conv3]
        for layer in layers:
            x = layer(x)
            sizes.append(x.size())
            x, idx = self.maxpool(x)
            indices.append(idx)
        return x, indices, sizes

class Decode(nn.Module):
    def __init__(self, config, size):
        super(Decode, self).__init__()
        self.conv1 = BasicConvTranspose2d(112, 16, kernel_size=(5, 1), stride=1)
        self.conv2 = BasicConvTranspose2d(16, 16, kernel_size=(5, 1), stride=1)
        self.conv3 = BasicConvTranspose2d(16, 16, kernel_size=(5, size), stride=1)
        self.unpool = nn.MaxUnpool2d((2, 1), 2)

    def forward(self, x, indices, sizes):
        layers = [self.conv1, self.conv2, self.conv3]
        indices = indices[::-1]
        sizes = sizes[::-1]
        sizes[0] = torch.Size((sizes[0][0], 112, sizes[0][2], sizes[0][3]))
        indices[0] = torch.cat((indices[0],) * 7, dim=1)
        sizes[-1] = torch.Size((sizes[-1][0], 16, sizes[-1][2], sizes[-1][3]))
        #indices[0] = torch.cat((indices[0],) * 7, dim=1)
        for layer, indices_, size in zip(layers, indices, sizes):
            x = layer(self.unpool(x, indices_, output_size=size))
        return x

def loss_and_acc(config, logits, labels):
    #pred = np.argmax(logits.data.cpu().numpy(), axis=1)
    logits = logits.squeeze()
    labels = labels.squeeze()

    #acc = np.mean((logits >= 0.5) == (labels == 1))
    rocauc = None
    if config.loss_func == 'mse':
        loss = config.loss(logits, labels)
        logits = logits.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        labels /= 100
        labels = np.maximum(labels, 0)
        rocauc = average_precision_score(labels.squeeze().astype(np.uint8).ravel(), logits.squeeze().ravel())
        #plt.imshow(logits.squeeze())
        #plt.imshow(labels.squeeze())
        #plt.pause(0.1)
    elif config.loss_func == 'ce':
        logits = logits.view(1, *logits.size())
        labels = labels.view(1, *labels.size())
        labels = torch.clamp(labels, min=0).long()
        loss = config.loss(logits, labels)
        logits = logits.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        pred = np.argmax(logits, axis=1)
        rocauc = average_precision_score(labels.squeeze().astype(np.uint8).ravel(), logits[:, 1, :, :].squeeze().ravel())
        #plt.imshow(np.argmax(logits[0, :, :, :], axis=0))
        #plt.pause(0.1)
    return loss, rocauc

class NonShitGenerator(nn.Module):
    def __init__(self, config, sizes):
        super(NonShitGenerator, self).__init__()
        self.conv_fe = []
        self.conv_depth = 16
        for fe_name in sizes:
            fe_ = Encode(config, sizes[fe_name])
            self.conv_fe.append(fe_)
        self.conv_fe = nn.ModuleList(self.conv_fe)
        self.combobulator = BasicConv2d(self.conv_depth * len(sizes), self.conv_depth, kernel_size=(3, 1), stride=1)
        self.discombobulator = BasicConvTranspose2d(self.conv_depth, self.conv_depth * len(sizes), kernel_size=(3, 1), stride=1)
        self.upconv1 = Decode(config, 1)
        self.upconv2 = Decode(config, 1)
        self.classifier = nn.Sequential(
                BasicConv2d(3, 16, kernel_size=9, stride=1, padding=4),
                BasicConv2d(16, 16, kernel_size=7, stride=1, padding=3),
                BasicConv2d(16, 16, kernel_size=5, stride=1, padding=2),
                BasicConv2d(16, 1, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        features, indices, sizes = [], None, None
        seqlen = x[0].size()[0]
        x, spatialfeats = x[:-2], x[-2:]
        for index, feature in enumerate(x):
            size = feature.size()
            feature = feature.view(1, 1, size[0], size[-1])
            feature, indices, sizes = self.conv_fe[index](feature)
            features.append(feature)
        features = torch.cat(tuple(features), dim=1)
        #features = self.combobulator(features)
        #features = self.discombobulator(features)
        vec1 = self.upconv1(features, indices, sizes)
        vec2 = self.upconv2(features, indices, sizes)
        vec1 = vec1.view(vec1.size()[2], 16)
        vec2 = vec2.view(16, vec2.size()[2])
        cm = torch.mm(vec1, vec2)
        cm = cm.view(1, 1, *cm.size())
        #cm = F.sigmoid(cm)
        spatialfeats = [feat.squeeze() for feat in spatialfeats]
        spatialfeats = [feat.view(1, 1, *feat.size()) for feat in spatialfeats]
        spatialfeats = torch.cat(tuple(cm) + tuple(spatialfeats), dim=1)
        spatialfeats = self.classifier(spatialfeats)
        return cm

class PlosOne(nn.Module):
    def __init__(self, config, sizes):
        super(PlosOne, self).__init__()
        self.conv_fe = []
        self.conv_depth = 16
        for fe_name in sizes:
            fe_ = Encode(config, sizes[fe_name])
            self.conv_fe.append(fe_)
        self.conv_fe = nn.ModuleList(self.conv_fe)
        self.combobulator = BasicConv2d(self.conv_depth * len(sizes), self.conv_depth, kernel_size=(3, 1), stride=1)
        self.discombobulator = BasicConvTranspose2d(self.conv_depth, self.conv_depth * len(sizes), kernel_size=(3, 1), stride=1)
        self.upconv1 = Decode(config, 1)
        self.upconv2 = Decode(config, 1)
        self.classifier = nn.Sequential(
                BasicConv2d(2, 64, kernel_size=9, stride=1, padding=4),
                BasicConv2d(64, 64, kernel_size=7, stride=1, padding=3),
                BasicConv2d(64, 64, kernel_size=5, stride=1, padding=2),
                BasicConv2d(64, 1, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        features, indices, sizes = [], None, None
        seqlen = x[0].size()[0]
        x, spatialfeats = x[:-2], x[-2:]
        spatialfeats = [feat.squeeze() for feat in spatialfeats]
        spatialfeats = [feat.view(1, 1, *feat.size()) for feat in spatialfeats]
        spatialfeats = torch.cat(tuple(spatialfeats), dim=1)
        spatialfeats = self.classifier(spatialfeats)
        print(seqlen, spatialfeats.size())
        return spatialfeats

class RecurrentProtein(nn.Module):
    def __init__(self, config, sizes):
        super(RecurrentProtein, self).__init__()
        self.recurrent = nn.GRU(
                input_size=sum(sizes.values()),
                hidden_size=16,
                num_layers=3,
                dropout=0.3,
                bidirectional=True
            )
        output_channels = 1 if config.loss_func == 'mse' else 2
        self.classifier = nn.Sequential(
                BasicConv2d(3, 64, kernel_size=9, stride=1, padding=4),
                BasicConv2d(64, 64, kernel_size=7, stride=1, padding=3),
                BasicConv2d(64, 64, kernel_size=5, stride=1, padding=2),
                BasicConv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
            )
        self.softmax = None if config.loss_func == 'mse' else nn.LogSoftmax()

    def forward(self, x):
        features, indices, sizes = [], None, None
        seqlen = x[0].size()[0]
        seqfeats, spatialfeats = x[:-2], x[-2:]
        seqfeats = torch.cat(tuple(seqfeats), dim=1)
        seqfeats = seqfeats.unsqueeze(1)
        seqfeats, _ = self.recurrent(seqfeats)
        seqfeats = seqfeats.squeeze()
        seqfeats = torch.mm(seqfeats, seqfeats.transpose(0, 1)).squeeze()
        seqfeats = seqfeats.view(1, 1, 1, *seqfeats.size()) # <-- what the fuck?????
        spatialfeats = [feat.squeeze() for feat in spatialfeats]
        spatialfeats = [feat.view(1, 1, *feat.size()) for feat in spatialfeats]
        allfeats = torch.cat(tuple(spatialfeats) + tuple(seqfeats), dim=1)
        allfeats = self.classifier(allfeats)
        if self.softmax:
            allfeats = self.softmax(allfeats)
        return allfeats

# class Generator(nn.Module):
#     def __init__(self, config, sizes):
#         super(Generator, self).__init__()
#         self.conv_fe = []
#         self.lstm_fe = []
#         self.hidden_sz = 32
#         for fe_name in sizes:
#             fe_ = nn.Sequential(
#                 BasicConv2d(sizes[fe_name], 16, kernel_size=(7, 1), stride=3),
#                 BasicConv2d(16, 16, kernel_size=(5, 1), stride=2),
#             )
#             self.conv_fe.append(fe_)
#         for fe_name in sizes:
#             fe_ = nn.Sequential(
#                 nn.LSTM(
#                     input_size=16, 
#                     hidden_size=self.hidden_sz,
#                     num_layers=3,
#                     dropout=0.3,
#                     bidirectional=True
#                 )
#             )
#             self.lstm_fe.append(fe_)
#         self.lstm_fe = nn.ModuleList(self.lstm_fe)
#         self.conv_fe = nn.ModuleList(self.conv_fe)
#         self.fc = nn.Linear(len(sizes) * self.hidden_sz * 2, self.hidden_sz)
#         self.cm_gen_1 = nn.LSTMCell(
#             input_size=self.hidden_sz, 
#             hidden_size=self.hidden_sz
#         )
#         self.cm_gen_2 = nn.LSTMCell(
#             input_size=self.hidden_sz, 
#             hidden_size=self.hidden_sz
#         )
#         self.transform = nn.Linear(self.hidden_sz, 1)
#         self.refine_conv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         features = []
#         seqlen = x[0].size()[0]
#         for index, feature in enumerate(x):
#             size = feature.size()
#             feature = feature.view(1, size[-1], size[0], 1)
#             feature = self.conv_fe[index](feature)
#             feature = torch.transpose(torch.squeeze(feature), 1, 0)
#             feature = feature.unsqueeze(1)
#             feature = self.lstm_fe[index](feature)[0][-1, -1, :].view(64, 1)
#             features.append(feature)
#         features = torch.cat(tuple(features))
#         features = features.view(-1, features.size()[0])
#         proj = self.fc(features)
#         vec1, vec2 = [], []
#         hx1, cx1 = Variable(torch.randn(1, self.hidden_sz)), Variable(torch.randn(1, self.hidden_sz))
#         hx2, cx2 = Variable(torch.randn(1, self.hidden_sz)), Variable(torch.randn(1, self.hidden_sz))
#         if torch.cuda.is_available():
#             hx1 = hx1.cuda()
#             hx2 = hx2.cuda()
#             cx1 = cx1.cuda()
#             cx2 = cx2.cuda()
#         input1, input2 = proj, proj
#         for i in range(seqlen):
#             hx1, cx1 = self.cm_gen_1(input1, (hx1, cx1))
#             hx2, cx2 = self.cm_gen_2(input2, (hx2, cx2))
#             input1, input2 = hx1, hx2
#             vec1.append(self.transform(hx1))
#             vec2.append(self.transform(hx2))
#         vec1 = torch.cat(vec1)
#         vec2 = torch.cat(vec2)
#         cm = torch.mm(vec1, vec2.transpose(0, 1))
#         cm = self.refine_conv(cm.view(1, 1, *cm.size()))
#         cm = F.sigmoid(cm)
#         return cm

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.hidden_sz = 32
        self.conv_fe = nn.Sequential(
                BasicConv2d(1, 64, kernel_size=7, stride=3, padding=1),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                BasicConv2d(64, 64, kernel_size=5, stride=2, padding=1),
                #nn.MaxPool2d(kernel_size=2, stride=2),
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
        'PSFM': 20,
        'sequence': 20
    }
    G = RecurrentProtein(config, sizes)
    #G = NonShitGenerator(config, sizes)
    #D = Discriminator(config)

    if torch.cuda.is_available():
        G = G.cuda()
        #D = D.cuda()

    config.G_optim = optim.Adam(G.parameters(), lr=config.lr)
    #config.D_optim = optim.RMSprop(D.parameters(), lr=config.lr)

    if config.loss_func == 'mse':
        config.loss = nn.MSELoss()
    elif config.loss_func == 'ce':
        config.loss = nn.NLLLoss2d()

    train_data = gen_dataset('train')
    test_data = gen_dataset('test')

    if config.mode == 0:
        G_losses, D_losses = [], []
        for i in range(config.epochs):
            G_loss, D_loss = run_epoch(G, D, batcher, i, config)
            save_checkpoint(G)
            G_losses.append(G_loss)
            D_losses.append(D_loss)
            np.save('outputs/G_losses', np.array(G_losses))
            np.save('outputs/D_losses', np.array(D_losses))

    if config.mode == 1:
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []
        best_acc = 0.0
        for i in range(config.epochs):
            train_loss, train_acc = run_epoch_(G, config, train_data, i, mode='Train')
            test_loss, test_acc = run_epoch_(G, config, test_data, i, mode='Test')
            if test_acc >= best_acc:
                save_checkpoint(G)
                best_acc = test_acc
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            np.save('outputs/train_loss_{}'.format(config.loss_func), np.array(train_losses))
            np.save('outputs/train_acc_{}'.format(config.loss_func), np.array(train_accs))
            np.save('outputs/test_loss_{}'.format(config.loss_func), np.array(test_losses))
            np.save('outputs/test_acc_{}'.format(config.loss_func), np.array(test_accs))

def generate_data(config):
    z = Variable(torch.randn(config.batch_sz, config.z_dim, 2, 2))
    if torch.cuda.is_available():
        z = z.cuda()
    return z

def one_hot_aa_matrix(mat):
    amino_acids = 'ARNDCEQGHILKMFPSTWYV'
    output_mat = np.zeros((len(mat), 20))
    for i in range(len(mat)):
        output_mat[i,amino_acids.index(mat[i])] = 1.
    return output_mat

def process_data(XX, mode, config):
    XX = XX[-1]
    X = copy.copy(XX)
    X['sequence'] = one_hot_aa_matrix(X['sequence'])
    #data = [Variable(torch.from_numpy(np.expand_dims(X[key], 1).astype(np.float32)), volatile=mode is not 'Train').float() for key in ['ACC', 'DISO', 'SS3', 'SS8', 'PSSM', 'PSFM', 'sequence', 'ccmpredZ', 'psicovZ']]
    data = [Variable(torch.from_numpy(X[key].astype(np.float32)), volatile=mode is not 'Train').float() for key in ['ACC', 'DISO', 'SS3', 'SS8', 'PSSM', 'PSFM', 'sequence', 'ccmpredZ', 'psicovZ']]
    cm = Variable(torch.from_numpy(X['contactMatrix'].astype(np.float32)), volatile=mode is not 'Train')
    if config.loss_func == 'mse':
        cm = cm.view(1, 1, *cm.size()) * 100
    elif config.loss_func == 'ce':
        cm = cm.view(1, 1, *cm.size())
    if torch.cuda.is_available():
        data = [d.cuda() for d in data]
        cm = cm.cuda()
    return data, cm

def run_epoch_(model, config, data, epoch, mode='Train'):
    total_acc, total_loss = 0.0, 0.0
    all_preds, all_labels, all_logits = [], [], []
    it = 0
    # you need to turn off batch-norm and dropout during Test
    if mode == 'Train': model.train()
    else: model.eval()
    # iterate over the dataset
    fold = get_batch(data, 1)
    for data in fold:
        X, labels = process_data(data, mode, config)
        logits = model(X)
        loss, acc = loss_and_acc(config, logits, labels)
        loss_num = loss.data[0]
        total_loss += loss_num
        total_acc += acc
        if mode == 'Train':
            # perform backprop
            config.G_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), config.clip_grad)
            config.G_optim.step()    
            if it % 500 == 0:
                print('Epoch {} | Iteration {} | Loss {} | Accuracy {} | LR {}'.
                    format(epoch, it, loss_num, acc, config.lr))
                print('Sample logits: {}, {}'.format(np.max(logits.data.cpu().numpy()), np.min(logits.data.cpu().numpy())))
                sys.stdout.flush()
        else:
            # record data for precision/recall calculations
            if it % 100 == 0:
                np.save('outputs/sample_{}_{}'.format(it // 100, config.loss_func), logits.data.cpu().numpy())
                np.save('outputs/reference_{}_{}'.format(it // 100, config.loss_func), labels.data.cpu().numpy())
                print('Test Sample logits: {}, {}'.format(np.max(logits.data.cpu().numpy()), np.min(logits.data.cpu().numpy())))
        it += 1
    total_loss /= it
    total_acc /= it
    print('{} loss:      {}'.format(mode, total_loss))
    print('{} accuracy:  {}'.format(mode, total_acc))
    return total_loss, total_acc

# this code adapted from wiseodd's WGAN tutorial
#   https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py
def run_epoch(G, D, batcher, epoch, config):
    G_losses, D_losses = [], []
    for it in range(300):
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

            G_losses.append(G_loss.data.cpu().numpy()[0])
            D_losses.append(D_loss.data.cpu().numpy()[0])

            G.eval()
            file = open('metadata.txt', 'w')
            for i in range(8):
                X_sample = next(batcher)
                X, y = process_data(X_sample)
                sample = G(X).data.cpu().numpy()
                reference = y.data.cpu().numpy()
                file.write('{}\n'.format(X_sample[0]['sequence']))
                np.save('outputs/sample_{}'.format(i), sample)
                np.save('outputs/reference_{}'.format(i), reference)

            file.close()

    return np.mean(G_losses), np.mean(D_losses)
