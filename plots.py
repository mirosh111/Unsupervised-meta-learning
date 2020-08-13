import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle



with open("results/MNIST/txt/mnist_maml_losses.txt", "rb") as f:
    mnist_meta_losses = pickle.load(f)

with open("results/MNIST/txt/mnist_pretrained_losses.txt", "rb") as f:
    mnist_pretrained_losses = pickle.load(f)


with open("results/Omniglot/txt/maml_conv_100_test_losses.txt", "rb") as f:
    maml_test_losses = pickle.load(f)

with open("results/Omniglot/txt/pretrained_128_test_losses_100TT.txt", "rb") as f:
    pretrained_test_losses = pickle.load(f)



sns.set()
plt.figure(figsize=(10,8))
sns.kdeplot(pretrained_test_losses, label='Pretrained VAE')
sns.kdeplot(maml_test_losses, label = 'MAML VAE')
plt.xlabel('Loss')
plt.ylabel('Density')
plt.legend()
plt.savefig('results/Omniglot/images/kde_plot.pdf')



plt.figure(figsize=(10,8), dpi=80)
plt.plot(mnist_meta_losses, label='MAML VAE ')
plt.plot(mnist_pretrained_losses, label='Pretrained VAE')
plt.title('MNIST Test loss')
plt.xlabel('Epochs')
plt.ylabel('Test loss')
plt.legend()
plt.savefig("results/MNIST/images/mnist_test_losses.pdf")

