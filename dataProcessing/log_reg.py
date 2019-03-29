import numpy as np
import math
import random
import matplotlib.pyplot as plt


class LogReg:
    def __init__(self, user_size, group_size, c, step_size):
        self.W = np.random.normal(0, c ** 3, size=(user_size, group_size))
        self.user_size = user_size
        self.group_size = group_size
        self.step_size = step_size
        self.c = c
        self.n_steps = 1
        self.sample = None
        self.norm_array = []

    def calc_s(self, x, z):
        """x - контекст пользователя
        z - контекст группы
        self.W - матрица весов
        """
        return np.dot(np.dot(x.transpose(), self.W), z)

    def sigmoid(self, arg):
        return 1.0 / (1 + np.exp(-arg))

    def step(self, x, z, r):
        s = self.sigmoid(r * self.calc_s(x, z))
        for i in range(self.user_size):
            for j in range(self.group_size):
                self.W[i][j] = self.W[i][j] - self.step_size * (1 / np.sqrt(self.n_steps)) * \
                               ((1 / self.c) * self.W[i][j] - r * x[i] * z[j] *
                                (1 - s))
        self.n_steps += 1

    def norma(self, A):
        return np.linalg.norm(A)

    def fit(self, sample):
        """sample - список кортежей вида (x_i, z_j, r_ij)
        """
        self.sample = sample
        n_steps = 0
        while True and n_steps < 1_000_000:
            x, z, r = random.choice(self.sample)
            print(n_steps)
            prom = self.W.copy()
            self.step(x, z, r)
            tmp = self.norma(prom - self.W)
            self.norm_array.append(tmp)
            if tmp < 1e-5:
                break
            n_steps += 1

        x = list(range(len(self.norm_array)))
        plt.plot(x, self.norm_array)
        plt.grid()
        plt.show()

    def predict(self, x, z):
        return self.sigmoid(self.calc_s(x, z))

    def export_model(self):
        pass

    def import_model(self):
        pass

