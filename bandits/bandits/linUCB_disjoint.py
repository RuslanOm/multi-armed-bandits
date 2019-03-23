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

        self.n_clicks_b.setdefault(key, 0)
        self.n_clicks_r.setdefault(key, 0)
        self.n_shows_b.setdefault(key, 0)
        self.n_shows_r.setdefault(key, 0)

    def predict_arm(self, event):

        arm, arms, reward, user_context, group_context = event

        for item in arms:
            if item not in self.arms:
                self.init_arm(item)

        payoffs = {}

        for key in arms:
            self.theta_hat[key] = np.dot(self.Aa_inv[key], self.ba[key])
            payoffs[key] = np.dot(self.theta_hat[key].transpose(), user_context) + \
                           self.alpha * np.sqrt(np.dot(np.dot(user_context.transpose(), self.Aa_inv[key]),
                                                       user_context))
        v = list(payoffs.values())
        k = list(payoffs.keys())
        return k[v.index(max(v))]

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
            self.ba[arm] += r * user_context
            self.theta_hat[arm] = self.Aa_inv[arm].dot(self.ba[arm])

            self.n_shows_b[arm] += 1
            self.n_clicks_b[arm] += reward


