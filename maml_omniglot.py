import torch
import torch.nn.functional as F
import utils
from torchvision import datasets, transforms
import VAE_model as vae
import easy_conv_vae as conv_vae
import maml_class
import numpy as np
import pickle
from torch import optim
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import seaborn as sns

data = datasets.Omniglot(root='./data', download=True)
source_task_number=500
task_set_number = 200
data_set = utils.get_dataset(data,50, 20, 10)
random.shuffle(data_set)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs=200


betas = [1.]
for j,beta in enumerate(betas):
    model = conv_vae.VAE(z_dim=128)
    model.to(device)
    maml = maml_class.MAML(model=model, data=data_set, inner_lr=1e-2,
            meta_lr=1e-3,
            input_dim=784,
            inner_steps=5,
            task_size=20,
            val_size=15,
            number_of_tasks=50,
            meta_batch_number=10,
            beta=beta)
    with open('results/logs/log.txt', 'w') as f:
        print('Omniglot training. '+str(source_task_number)+' tasks with 20 pictures, beta='+ str(j), file=f)

    maml.main_loop(epochs-1)
    sns.set()
    plt.figure(figsize = (20,8))
    plt.plot(maml.meta_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Meta Loss')
    plt.savefig("results/Omniglot/images/omni_maml_loss_500tasks.pdf")

    with open('results/logs/tune_log.txt', 'w') as f:
        print('Finetuning on '+str(task_set_number)+' test tasks', file=f)
    numbers = [i for i in range(source_task_number,source_task_number+task_set_number+1)]
    task_losses = []
    mean_task_losses = []
    std_task_losses = []
    for i in tqdm(range(task_set_number)):
        plt.close('all')
        number = source_task_number + i
        task = utils.sample_task(data, 20, str(number))
        task_loss = maml.finetuning(task, 10, number=number, numbers=numbers,
                                    name='results/images/omni_'+str(task_set_number)+'tasks_beta'+str(j+1)+'_'+str(number)+'.pdf')
        task_losses.append(task_loss)
        if i%10==0:
            mean_task_losses.append(np.mean(task_losses))
            std_task_losses.append(np.std(task_losses))

    title = 'MAML VAE beta='+str(betas[j])+' '+str(source_task_number)+' tasks'
    xaxis = np.array(range(1, int(task_set_number/10) + 1))
    utils.plot_std(xaxis,mean_task_losses,std_task_losses, title=title,
                   name = 'results/Omniglot/images/STD_plot_'+str(source_task_number)+'tasks')

    with open('results/Omniglot/txt/maml_task_losses_'+str(source_task_number)+'tasks_beta_'+str(j+1)+'.txt', 'wb') as f:
        pickle.dump(task_losses,f)

    model.to("cpu")
    torch.save(model, 'results/Omniglot/models/maml_conv_'+str(source_task_number)+'tasks_beta_'+str(j+1)+'.pt')







