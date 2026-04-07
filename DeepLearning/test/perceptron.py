import numpy as np


class perceptron:
    def __init__(self):
        self.andW0 = 1.0
        self.andW1 = 1.0
        self.andTheta = 1.0
        self.nandW0 = -1.0
        self.nandW1 = -1.0
        self.nandTheta = -1.5
        self.orW0 = 1.0
        self.orW1 = 1.0
        self.orTheta = 0.5

    def AND(self, x: float, y: float) -> int:
        print(f"AND({x}, {y}) -> ", end="")
        return 1 if ((x * self.andW0 + y * self.andW1) > self.andTheta) else 0

    def NAND(self, x: float, y: float) -> int:
        print(f"NAND({x}, {y}) -> ", end="")
        return 1 if ((x * self.nandW0 + y * self.nandW1) > self.nandTheta) else 0

    def OR(self, x: float, y: float) -> int:
        print(f"OR({x}, {y}) -> ", end="")
        return 1 if ((x * self.orW0 + y * self.orW1) > self.orTheta) else 0

    def XOR(self, x: float, y: float) -> int:
        print(f"XOR({x}, {y}) -> ", end="")

        orRet = self.OR(x, y)
        nandRet = self.NAND(x, y)

        return self.AND(orRet, nandRet)


def perceptron_test(func):
    print(func(0, 0))
    print(func(1, 0))
    print(func(0, 1))
    print(func(1, 1))


if __name__ == "__main__":
    p = perceptron()

    perceptron_test(p.XOR)
