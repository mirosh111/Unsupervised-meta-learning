import torch
import torch.nn.functional as F
import utils
import statistics
from tqdm import tqdm
import copy

class MAML():
    def __init__(self, model, data, inner_lr, meta_lr, input_dim, inner_steps=3, task_size=20, val_size=5, meta_batch_number=1, number_of_tasks=10, beta=1):
        self.data = data
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model.to(self.device)
        self.weights = list(model.parameters())
        self.meta_optimizer = torch.optim.Adam(self.weights, meta_lr)

        self.input_dim = input_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.task_size = task_size
        self.number_of_tasks = number_of_tasks
        self.meta_batch_number = meta_batch_number
        self.val_size = val_size
        self.beta = beta


        # additional
        self.print_every = 1
        self.plot_every = 1
        self.clip_value = 5
        self.meta_losses = []
        self.BCE_losses = []
        self.KLD_losses = []
        self.tune_losses = []


    def inner_loop(self, task):
        # x = task.view(-1,  self.input_dim)
        # train_x = task[self.val_size:, :].to(self.device)
        # test_x = task[:self.val_size, :].to(self.device)
        train_x = task.to(self.device)
        test_x = task.to(self.device)
        for step in range(self.inner_steps):
            if step==0:
                x_sample, z_mu, z_var = self.model(train_x, self.weights)
                train_loss, BCE, KLD = utils.loss_function(x_sample, train_x, z_mu, z_var, self.beta)

                grad = torch.autograd.grad(train_loss, self.weights)

                # gradient clipping
                for w, g in zip(self.weights, grad):
                    w.grad = g
                torch.nn.utils.clip_grad_norm_(self.weights, self.clip_value)

                temp_weights = [w - self.inner_lr * g for w,g in zip(self.weights, grad)]
                

            else:
                x_sample, z_mu, z_var = self.model(train_x, temp_weights)
                train_loss, BCE, KLD = utils.loss_function(x_sample, train_x, z_mu, z_var, self.beta)
                grad = torch.autograd.grad(train_loss, temp_weights)

                # gradient clipping
                for w, g in zip(temp_weights, grad):
                    w.grad = g
                torch.nn.utils.clip_grad_norm_(temp_weights, self.clip_value)

                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
                
            with open('results/logs/log.txt', 'a+') as f:
                print("inner iteration: {}, Recon loss: {}, KLD Loss: {}".format(step, BCE, KLD), file=f)

        x_sample, z_mu, z_var = self.model(test_x, temp_weights)
        task_loss, BCE, KLD = utils.loss_function(x_sample, test_x, z_mu, z_var, self.beta)
        self.BCE_losses.append(BCE)
        self.KLD_losses.append(KLD)
        with open('results/logs/log.txt','a+') as f:
            print("Task Recon loss: {}, Task KLD loss: {}".format(BCE, KLD),file=f)

        return task_loss

    def main_loop(self, epochs):
        for epoch in tqdm(range(epochs + 1)):
            epoch_loss = 0
            # print("Epoch number: {}/{}".format(epoch, epochs))

            for batch in self.data:
                meta_loss = 0
                for i, task in enumerate(batch):
                    task_loss = self.inner_loop(task)
                    # print("Task: {},  Loss: {}".format(i,task_loss))
                    meta_loss += task_loss

                meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
                for w, g in zip(self.weights, meta_grads):
                    w.grad = g
                torch.nn.utils.clip_grad_norm_(self.weights, self.clip_value)
                self.meta_optimizer.step()

                epoch_loss = meta_loss/len(batch)


            if epoch % self.print_every == 0:
                with open('results/logs/log.txt', 'a+') as f:
                    print("{}/{}. Train loss: {}".format(epoch, epochs, epoch_loss/len(self.data)), file=f)


            if epoch % self.plot_every == 0:
                self.meta_losses.append(epoch_loss/len(self.data))
    
    
    def finetuning(self, task, tune_epochs, number, numbers, name):
        train_x = task.to(self.device)
        test_x = task.to(self.device)
        model_copy = copy.deepcopy(self.model)
        weights = list(model_copy.parameters())
        for epoch in range(tune_epochs):
            if epoch==0:
                
                x_sample, z_mu, z_var = model_copy(train_x, weights)
                train_loss, BCE, KLD = utils.loss_function(x_sample, train_x, z_mu, z_var, self.beta)

                grad = torch.autograd.grad(train_loss, weights)

                # gradient clipping
                for w, g in zip(weights, grad):
                    w.grad = g
                torch.nn.utils.clip_grad_norm_(weights, self.clip_value)

                temp_weights = [w - self.inner_lr * g for w, g in zip(weights, grad)]
            else:
                x_sample, z_mu, z_var = model_copy(train_x, temp_weights)
                train_loss, BCE, KLD = utils.loss_function(x_sample, train_x, z_mu, z_var, self.beta)
                grad = torch.autograd.grad(train_loss, temp_weights)

                # gradient clipping
                for w, g in zip(temp_weights, grad):
                    w.grad = g
                torch.nn.utils.clip_grad_norm_(temp_weights, self.clip_value)

                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
            with open("results/logs//tune_log.txt", 'a+') as f:
                print("inner iteration: {}, Recon_loss = {}, KLD_loss = {}".format(epoch, BCE, KLD), file=f)

        x_sample, z_mu, z_var = model_copy(test_x, temp_weights)
        task_loss, BCE, KLD = utils.loss_function(x_sample, test_x, z_mu, z_var, self.beta)
        if number in numbers:
            utils.plot_images(model_copy, temp_weights, test_x, BCE, name=name)
        with open('transfer/tune_log.txt', 'a+') as f:
            print('Task test loss: {}'.format(task_loss), file=f)
        
        return task_loss.item() 


