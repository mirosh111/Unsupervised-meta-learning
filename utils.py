import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta=1.):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')/ x.shape[0]

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/ x.shape[0]

    return BCE + beta * KLD, BCE, KLD

# Sampling data for 1 task in MAML training
def sample_task(data, task_size, task_label, image_resize=28):
    task_to_examples = {}
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((image_resize,image_resize))
    for i, (_, digit) in enumerate(data):
        if str(digit) not in task_to_examples:
            task_to_examples[str(digit)] = []
        task_to_examples[str(digit)].append(i)
    task_idx = random.sample(task_to_examples[task_label], task_size)

    task = torch.tensor(np.array([to_tensor(resize(data[idx][0])).numpy() for idx in task_idx]),
                             dtype=torch.float)
    return task

# Creating dataset for MAML training
def get_dataset(data, batch_size, task_size, batch_number, image_resize=28):
    data_set = []
    for i in tqdm(range(batch_number)):
        batch = []
        for j in tqdm(range(batch_size)):
            task = sample_task(data, task_size, str(i*batch_size + j), image_resize)
            batch.append(task)
        data_set.append(batch)
    return data_set


def get_shuffled_set(data, size, image_resize=28):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((image_resize, image_resize))
    digit_numbers = [i for i in range(size)]
    random.shuffle(digit_numbers)

    batch = torch.tensor(np.array([to_tensor(resize(data[idx][0])).numpy() for idx in digit_numbers]),
                         dtype=torch.float)
    return batch

# Plotting resulting images for MAML finetuning
def plot_images(model, weights, x, loss, z_dim=128, name='image'):
    x_sample,_,_ = model(x, weights)
    images = []
    # n = x.shape[0]
    n=5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for _ in range(n):
        z = torch.randn(1, z_dim)
        z = z.to(device)
        img = model.decode(z, weights)
        img = img.view(28, 28).data
        images.append(img)
    fig = plt.figure(figsize=(20,8))
    for idx in range(n):
        ax = fig.add_subplot(3, n, idx+1, xticks=[], yticks=[])
        ax.imshow(torch.squeeze(x[idx].cpu()), cmap='gray')
        ax2 = fig.add_subplot(3, n, idx + 1 + n, xticks=[], yticks=[])
        ax2.imshow(torch.squeeze(x_sample[idx].cpu().view(28, 28).data), cmap='gray')
        ax3 = fig.add_subplot(3, n, idx + 2*n + 1, xticks=[], yticks=[])
        ax3.imshow(torch.squeeze(images[idx].cpu()),cmap='gray')
    fig.suptitle('Reconstruction loss = ' + str(round(loss.item(),1)), fontsize=16)
    plt.savefig(name)

# Plotting resulting images for pretarained VAE finetuning
def plot_images_2(model, x, loss, z_dim, name):
    x_sample,_,_ = model(x)
    images = []
    n=5
    for _ in range(n):
        z = torch.randn(1, z_dim)
        img = model.decode(z)
        img = img.view(28, 28).data
        images.append(img)
    fig = plt.figure(figsize=(20,8))
    for idx in range(n):
        ax = fig.add_subplot(3, n, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x[idx]), cmap='gray')
        ax2 = fig.add_subplot(3, n, idx + 1 + n, xticks=[], yticks=[])
        ax2.imshow(np.squeeze(x_sample[idx].view(28, 28).data), cmap='gray')
        ax3 = fig.add_subplot(3, n, idx + 2*n + 1, xticks=[], yticks=[])
        ax3.imshow(np.squeeze(images[idx]),cmap='gray')
    fig.suptitle('Reconstruction loss = ' + str(round(loss.item(),1)), fontsize=16)
    plt.savefig(name + '.pdf')


# Plotting STD graph
def plot_std(xaxis,mean_loss, std_loss,title, name):
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    line_style = {"linestyle": "--", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 3}
    ax.errorbar(xaxis, mean_loss, yerr=std_loss, **line_style, color='red', label='Target loss')
    plt.xticks(xaxis)
    plt.yticks(np.arange(10, 250, 10))
    for i, txt in enumerate(std_loss):
        ax.annotate(int(txt), xy=(xaxis[i], int(mean_loss[i])),
                xytext=(xaxis[i] + 0.03,int(mean_loss[i]) + 0.3), color='red')
        ax.grid(color='lightgrey', linestyle='-')
        ax.set_facecolor('w')
        ax.set_xlabel('Number of target tasks', fontsize=18)
        ax.set_ylabel('Test loss', fontsize=18)
        plt.title(title)
    plt.savefig(name+'.pdf')


# Convolutional network layers for MAML realization
def linear(input, weight, bias=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bias is None:
        return F.linear(input, weight.to(device))
    else:
        return F.linear(input, weight.to(device), bias.to(device))

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return F.conv2d(input, weight.to(device), bias.to(device), stride, padding, dilation, groups)

def batch_norm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    # This hack only works when momentum is 1 and avoids needing to track running stats
    # by substuting dummy variables
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).to(device)
    running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).to(device)
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return F.conv_transpose2d(input, weight.to(device), bias.to(device), stride, padding, output_padding, groups, dilation)
