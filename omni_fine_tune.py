import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import VAE_model as vae
import utils
import statistics
import easy_conv_vae as conv_vae
import copy


def train_model(model, dataset, class_numbers, epochs=50, lr=1e-2, image_name='sample', beta=1., number = 500):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    test_losses=[]
    train_x = dataset
    test_x = dataset
    for e in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_sample, z_mu, z_var = model(train_x)
        train_loss, _, _ = utils.loss_function(x_sample, train_x, z_mu, z_var, beta=beta)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            x_sample, z_mu, z_var = model(test_x)
            test_loss, BCE, KLD = utils.loss_function(x_sample, test_x, z_mu, z_var, beta=beta)
            test_losses.append(test_loss.item())
    if number in class_numbers:
        utils.plot_images_2(model, test_x, BCE, z_dim=128, name=image_name)
    return train_loss, test_loss, BCE, KLD


def fine_tune(data, model, source_task_number, target_task_number, epochs, class_numbers, lr=1e-2, image_name='image', beta=1.):
    KLD_losses = []
    BCE_losses = []
    test_losses = []
    mean_test_losses=[]
    std_test_losses = []
    for i in range(1, target_task_number+1):
        plt.close('all')
        number = source_task_number + i
        task = utils.sample_task(data, 20, str(number))
        model_copy = copy.deepcopy(model)
        name = image_name + str(number)
        train_loss, test_loss, BCE, KLD = train_model(model_copy, task, class_numbers=class_numbers, epochs=epochs, lr=lr,
                                                          image_name=name, beta=beta, number = number)
        test_losses.append(test_loss.item())
        KLD_losses.append(KLD.item())
        BCE_losses.append(BCE.item())

        if i%10 ==0:
            mean_test_losses.append(np.mean(test_losses))
            std_test_losses.append(np.std(test_losses))


        print("Task: {}, Train_loss: {}, Test_loss: {}".format(source_task_number+i, train_loss.item(), test_loss.item()))

    return test_losses, BCE_losses, KLD_losses, mean_test_losses, std_test_losses






data = datasets.Omniglot(root = './data', download=True)


source_task_number = 500
target_task_number = 200
class_numbers=[550,600,605,610,611,612,615,619,650]
beta = 1.
model = torch.load('results/Omniglot/models/pretrained_VAE_'+str(source_task_number)+'tasks.pt')


test_losses, BCE_losses, KLD_losses, mean_test_losses, std_test_losses = fine_tune(data,
                                                           model,
                                                           source_task_number,
                                                           target_task_number,
                                                           class_numbers=class_numbers,
                                                           epochs=10,
                                                           lr=1e-3,
                                                           image_name='results/Omniglot/images/pretrained_'+str(source_task_number)+'tasks',
                                                           beta=beta)



title = 'Pretrained VAE Omniglot beta='+str(beta)+str(source_task_number)+'  tasks'
xaxis = np.array(range(1, int(target_task_number/10) + 1))
utils.plot_std(xaxis,mean_test_losses, std_test_losses, title=title, name='Omniglot/images/STD_pretrained_'+str(source_task_number)+'tasks')

with open('transfer/pretrained_omni_'+str(source_task_number)+'task_losses.txt', "wb") as f:
    pickle.dump(test_losses, f)




