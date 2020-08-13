import torch
import torch.nn.functional as F
import utils
from torchvision import datasets, transforms
import easy_conv_vae as vae
import maml_class
import numpy as np
import pickle
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

data = datasets.MNIST(root='./data', download=True)

source_task_number = 8
task_size = 500
beta = 2.5
data_set = utils.get_dataset(data, source_task_number, task_size, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = vae.VAE(z_dim=128)
model.to(device)

maml = maml_class.MAML(model=model,
                       data=data_set,
                       inner_lr=1e-2,
                       meta_lr=1e-3,
                       input_dim=784,
                       inner_steps=5,
                       task_size=task_size,
                       val_size=5,
                       number_of_tasks=source_task_number,
                       meta_batch_number=1,
                       beta = beta)

with open('results/logs/log.txt', 'w') as f:
    print('MNIST training. 8 tasks with '+str(task_size)+' pictures', file=f)

maml.main_loop(epochs=100)

sns.set()
plt.figure(figsize = (20, 8))
plt.plot(maml.meta_losses)
plt.xlabel('Epochs')
plt.ylabel('Meta loss')
plt.savefig('results/MNIST/mnist_maml_loss_'+str(task_size)+'.pdf')

with open('results/logs/tune_log.txt', 'w') as f:
    print('Finetuning 2 test tasks', file=f)
task_set_number = 2
numbers = [8,9]
all_tasks_losses = []
for i in tqdm(range(1, task_set_number+1)):
    plt.close('all')
    number = source_task_number+i-1
    task = utils.sample_task(data, 20, str(number))
    losses, task_loss = maml.finetuning(task,10,number=number, numbers=numbers,
                                        name='results/MNIST/images/mnist_'+str(task_size)+'_'+str(number)+'.pdf')
    all_tasks_losses.append(losses)

with open('results/MNIST/txt/mnist_maml_test_losses_'+str(task_size)+'.txt', 'wb') as f:
    pickle.dump(all_tasks_losses, f)
torch.save(maml.model, 'results/MNIST/models/mnist_maml_model_'+str(task_size)+'.pt')






