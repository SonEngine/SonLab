import numpy as np
import sys,os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHAP5_DIR = os.path.join(BASE_DIR, "Chap5")

sys.path.append(BASE_DIR)
sys.path.append(MODEL_DIR)
sys.path.append(CHAP5_DIR)

from network import network
import matplotlib.pyplot as plt
from optimizer import *

save_path = os.path.join(MODEL_DIR, "params.npz")

from dataset.mnist import load_mnist

if __name__ == "__main__":

    (x_train, t_train), (xtest, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list = []
    train_acc_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    image_size = x_train.shape[1]
    batch_size = 100
    learning_rate = 0.1

    n = network(image_size, 50, 10, "ReLU")
    iter_per_epoch = max(train_size / batch_size, 1)
    optimizer = Momentum(0.01, 0.9)
    #optimizer = AdaGrad(0.01)

    for i in range(iters_num):

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        loss = n.predict(x_batch, t_batch)
        train_loss_list.append(loss)

        grads = n.gradient(x_batch, t_batch)
        params = n.w
        optimizer.update(params, grads)

        if i % iter_per_epoch == 0:
            train_acc = n.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            percent = 100*(i+1) / iters_num
            print(f"train_acc | {train_acc:.3f} - {int(percent)} %")


    test_acc = n.accuracy(xtest, t_test)
    
    print(f"test acc | {test_acc:.3f}")

    np.savez_compressed(save_path, **n.w)
