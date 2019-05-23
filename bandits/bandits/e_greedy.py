"""This class is an implementation E-Greedy algorithm from https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

"""

import random
from bandits.bandits.base_bandit import BaseBandit


class EGreedy(BaseBandit):

    def __init__(self, epsilon, average_reward=0):
        super().__init__()
        self.epsilon = epsilon
        self.n = 1
        self.k = 0
        self.curr_max = None
        self.rewards_d = {}
        self.n_plays = {}

        self.average_reward = average_reward

    def predict_arm(self, event):
        arm, arms, reward, user_context, groups = event
        # если руку ещё не видели, то инициализируем ее
        for item in arms:
            if item not in self.arms:
                self.init_arm(item)

        # считаем e_n по формуле из статьи
        e_n = self.epsilon
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
        self.rewards_d.setdefault(arm, 0)
        self.n_plays.setdefault(arm, 1)

        self.n_clicks_b.setdefault(arm, 0)
        self.n_clicks_r.setdefault(arm, 0)
        self.n_shows_b.setdefault(arm, 0)
        self.n_shows_r.setdefault(arm, 0)

    def get_max_hand(self, arms):
        # список текущих средних наград рук
        ls_tmp = [self.rewards_d[item] / self.n_plays[item] for item in arms]

        curr_max = max(ls_tmp)
        ind = ls_tmp.index(curr_max)

        # обновляем текущую максимальную руку
        self.curr_max = arms[ind]

    def update(self, event):
        arm, arms, reward, user_context, group_context = event
        self.rewards_d[arm] += reward
        self.n_plays[arm] += 1
        self.n += 1

        self.n_shows_b[arm] += 1
        self.n_clicks_b[arm] += reward

        self.n_steps += 1
        self.rewards += reward

        self.regret.append(self.n_steps * self.average_reward - self.rewards)

