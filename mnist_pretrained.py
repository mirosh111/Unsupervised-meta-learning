from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import utils
import easy_conv_vae as conv_vae
import pickle
import matplotlib.pyplot as plt
import numpy as np



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = datasets.MNIST(root='./data')

task_size = 500
beta = 2.5
train_set = utils.sample_task(data,task_size,str(0))
for i in range(1,8):
    task = utils.sample_task(data, task_size, str(i))
    train_set = torch.cat((task, train_set), dim=0)

test_set = utils.sample_task(data,task_size,'7')
train_loader = torch.utils.data.DataLoader(train_set,batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=20, shuffle=True)

model = conv_vae.VAE()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(recon_x, x, mu, logvar, beta=1.):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta=beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, beta=beta).item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))


train_losses = []
test_losses = []
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

plt.figure(figsize = (20,8))

torch.save(model,'results/MNIST/models/pretrained_VAE_mnist_'+str(task_size)+'.pt')
