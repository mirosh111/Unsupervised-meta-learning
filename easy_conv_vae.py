import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import utils


class VAE(nn.Module):
    def __init__(self, z_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)),
            ('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            ('bn2', nn.BatchNorm2d(128, momentum=1, affine=True)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            ('bn3', nn.BatchNorm2d(256, momentum=1, affine=True)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0)),
            ('bn4', nn.BatchNorm2d(512, momentum=1, affine=True)),
            ('relu4', nn.ReLU())
        ]))
        self.add_module('fc21', nn.Linear(512, z_dim))
        self.add_module('fc22', nn.Linear(512, z_dim))
        self.add_module('fc4', nn.Linear(z_dim, 512))

        self.decoder = nn.Sequential(OrderedDict([
            ('convTran1', nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0)),
            ('bn5', nn.BatchNorm2d(256, momentum=1, affine=True)),
            ('relu5', nn.ReLU()),
            ('convTran2', nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)),
            ('bn6', nn.BatchNorm2d(128, momentum=1, affine=True)),
            ('relu6', nn.ReLU()),
            ('convTran3', nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)),
            ('bn7', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu7', nn.ReLU()),
            ('convTran4', nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    # def reparameterize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
        # eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        # return eps.mul(std).add_(mu)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, weights=None):
        if weights==None:
            conv = self.encoder(x)
            mu, logvar = self.fc21(conv.view(-1,512)), self.fc22(conv.view(-1,512))
            z = self.reparameterize(mu, logvar)
        else:
            x = utils.conv2d(x, weights[0], weights[1], stride=2, padding=1)
            x = utils.batch_norm(x, weights[2], weights[3], momentum=1)
            x = F.relu(x)
            x = utils.conv2d(x, weights[4], weights[5], stride=2, padding=1)
            x = utils.batch_norm(x, weights[6], weights[7], momentum=1)
            x = F.relu(x)
            x = utils.conv2d(x, weights[8], weights[9], stride=2, padding=1)
            x = utils.batch_norm(x, weights[10], weights[11], momentum=1)
            x = F.relu(x)
            x = utils.conv2d(x, weights[12], weights[13], stride=1, padding=0)
            x = utils.batch_norm(x, weights[14], weights[15], momentum=1)
            x = F.relu(x)
            x = x.view(-1, 512)
            mu = utils.linear(x, weights[16], weights[17])
            logvar = utils.linear(x, weights[18], weights[19])
            z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, weights=None):
        if weights==None:
            z = self.fc4(z)
            deconv_input = z.view(-1, 512, 1, 1)
            z = self.decoder(deconv_input)
        else:
            z = utils.linear(z, weights[20], weights[21])
            z = z.view(-1, 512, 1, 1)
            z = utils.conv_transpose2d(z, weights[22], weights[23], stride=1, padding=0)
            z = utils.batch_norm(z, weights[24], weights[25], momentum=1)
            z = F.relu(z)
            z = utils.conv_transpose2d(z, weights[26], weights[27], stride=2, padding=1)
            z = utils.batch_norm(z, weights[28], weights[29], momentum=1)
            z = F.relu(z)
            z = utils.conv_transpose2d(z, weights[30], weights[31], stride=2, padding=1)
            z = utils.batch_norm(z, weights[32], weights[33], momentum=1)
            z = F.relu(z)
            z = utils.conv_transpose2d(z, weights[34], weights[35], stride=2, padding=1)
            z = torch.sigmoid(z)
        return z

    def forward(self, x, weights=None):
        z, mu, logvar = self.encode(x, weights)
        z = self.decode(z, weights)
        return z, mu, logvar
