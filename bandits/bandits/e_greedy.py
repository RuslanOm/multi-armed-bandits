# сам алгоритм взят отсюда
# https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

import math
import random
from base_bandit import BaseBandit


class EGreedy(BaseBandit):

    def __init__(self, c, d):
        super().__init__()
        self.n = 1
        self.k = 0
        self.curr_max = None
        self.rewards = {}
        self.n_plays = {}

        # параметры алгоритма
        self.c = c
        self.d = d

    def predict_arm(self, event):
        arm, arms, reward, user_context, group_context = event
        # если руку ещё не видели, то инициализируем ее
        for item in arms:
            if item not in self.arms:
                self.init_arm(item)

        # считаем e_n по формуле из статьи
        e_n = min(1, self.c * self.k / (self.d ** 2 * self.n))
        r = random.random()

        # обновляем текущую максимальную руку
        # e_n меньше единицы, то выполняем условия, иначе просто играем случайную руку
        self.get_max_hand(arms)
        if e_n < 1:
            if 0 < r < 1 - e_n:
                return self.curr_max
            else:
                random_int = random.randint(0, len(arms) - 1)
                return list(arms)[random_int]
        random_int = random.randint(0, len(arms) - 1)
        return list(arms)[random_int]

    def init_arm(self, arm):
        self.arms.add(arm)
        self.rewards.setdefault(arm, 0)
        self.n_plays.setdefault(arm, 1)

    def get_max_hand(self, arms):
        # список текущих средних наград рук
        ls_tmp = [self.rewards[item] / self.n_plays[item] for item in arms]

        curr_max = max(ls_tmp)
        ind = ls_tmp.index(curr_max)

        # обновляем текущую максимальную руку
        self.curr_max = arms[ind]

    def update(self, event):
        arm, arms, reward, user_context, group_context = event
        self.rewards[arm] += reward
        self.n_plays[arm] += 1

