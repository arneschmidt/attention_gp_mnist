import matplotlib.pyplot as plt
import numpy as np


in_dir = '/home/arne/projects/attention_gp_mnist/out_val_cifar10/'
train_file = 'sparse_categorical_accuracy.npy'
val_file = 'val_sparse_categorical_accuracy.npy'
out_path = './train_curve.png'

train = np.load(in_dir + train_file)
val = np.load(in_dir + val_file)

x = np.arange(0, train.shape[0])


plt.rcParams.update({'font.size': 16})
plt.plot(x, train, 'k', label="training")
plt.plot(x, val, 'r', label="validation")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.savefig(out_path)