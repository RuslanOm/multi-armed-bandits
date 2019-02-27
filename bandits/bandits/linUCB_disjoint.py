import numpy as np
import pandas as pd
from base_bandit import BaseBandit


class DisjointBandit(BaseBandit):
    def __init__(self,  alpha, size_of_user_contex, r1, r0):
        super().__init__()
        self.r1 = r1
        self.r0 = r0
        self.size_of_user_context = size_of_user_contex
        self.alpha = alpha

        self.Aa = {}
        self.Aa_inv = {}
        self.ba = {}
        self.theta_hat = {}

    def init_arm(self, key):
        self.arms.add(key)
        self.Aa[key] = np.identity(self.size_of_user_context)
        self.Aa_inv[key] = np.identity(self.size_of_user_context)
        self.ba[key] = np.zeros((self.size_of_user_context, 1))

        self.n_shows_b[key] = 0
        self.n_shows_r[key] = 0

        self.n_clicks_b[key] = 0
        self.n_clicks_r[key] = 0

    def predict_arm(self, event):

        arm, arms, reward, user_context, group_context = event

        self.n_shows_r[arm] += 1
        self.n_clicks_r[arm] += reward

        # если руку ещё не видели - инициализируем ее
        if arm not in self.Aa:
            self.init_arm(arm)

        payoffs = {}

        for key in arms:
            self.theta_hat[key] = np.dot(self.Aa_inv[key], self.ba[key])
            payoffs[key] = np.dot(user_context, self.theta_hat[key]) + \
                           self.alpha * np.sqrt(np.dot(np.dot(user_context, self.Aa_inv[key]),
                                                       user_context.transpose()))
        v = list(payoffs.values())
        k = list(payoffs.keys())
        return k[v.index(max(v))], max(v)

    def update(self, event):
        arm, arms, reward, user_context, group_context = event

        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0

            self.Aa[arm] += np.outer(user_context, user_context)  
            self.Aa_inv[arm] = np.linalg.inv(self.Aa[arm])
            self.ba[arm] += r * user_context.transpose()
            self.theta_hat[arm] = self.Aa_inv[arm].dot(self.ba[arm])

            self.n_shows_b[arm] += 1
            self.n_clicks_b[arm] += r


