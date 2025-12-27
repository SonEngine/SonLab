import numpy as np


def add(x):
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        x[idx] += 1
        it.iternext()


class test:
    def __init__(self):
        self.x = {}
        self.x["test"] = np.array([1.0, 2.0, 3.0])

    def t(self):
        add(self.x["test"])
        print(self.x["test"])


if __name__ == "__main__":
    # t = test()
    # t.t()

    arr = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
        ]
    )
    batch_size = 2
    train_size = arr.shape[0]

    for i in range(10):

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = arr[batch_mask]
        print(f"{train_size}, {batch_mask}, {x_batch}")
