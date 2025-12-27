from network import network
import numpy as np
import matplotlib.pyplot as plt
import os, sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

sys.path.append(BASE_DIR)
sys.path.append(MODEL_DIR)

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

    n = network(image_size, 50, 10)
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        loss = n.predict(x_batch, t_batch)
        train_loss_list.append(loss)

        grad = n.gradient(x_batch, t_batch)

        for key in {"w1", "b1", "w2", "b2"}:
            n.w[key] -= learning_rate * grad[key]

        if i % iter_per_epoch == 0:
            train_acc = n.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            print("train acc | " + str(train_acc))

    test_acc = n.accuracy(xtest, t_test)
    print("test acc | " + str(test_acc))

    np.savez_compressed(save_path, **n.w)
