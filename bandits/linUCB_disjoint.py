"""This class it is an implementation of Contextual Bandit Algorithm with Disjoint Linear Model

"""
import numpy as np
import pandas as pd


class ContextBandit:

    def __init__(self, alpha, context_size, set_of_arms):
        """Initialization of our bandit class
        alpha: float, parameter of algorithm
        """
        self.alpha = alpha
        self.arms = set_of_arms
        self.n_context = context_size

        self.Aa = {key: np.identity(self.n_context) for key in self.arms}
        self.Aa_inv = {key: np.identity(self.n_context) for key in self.arms}
        self.b_a = {key: np.zeros((self.n_context, 1)) for key in self.arms}
        self.theta_a = {key: np.zeros((self.n_context, 1)) for key in self.arms}

        self.n_shows_b = {key: 0 for key in self.arms}
        self.n_shows_r = {key: 0 for key in self.arms}

        self.n_clicks_b = {key: 0 for key in self.arms}
        self.n_clicks_r = {key: 0 for key in self.arms}

    def calc_payoffs(self, context):
        payoffs = {key: -1 for key in self.Aa}

        mx = -1000000
        ind = -1

        for key in self.Aa:
            self.theta_a[key] = np.dot(self.Aa_inv[key], self.b_a[key])
            payoffs[key] = np.dot(context, self.theta_a[key]) + \
                           self.alpha * np.sqrt(np.dot(np.dot(context, self.Aa_inv[key]), context.transpose()))
            ind = ind if mx > payoffs[key] else key
            mx = mx if mx > payoffs[key] else payoffs[key]

        return ind, mx, payoffs

    def update(self, context, reward, arm):
        self.Aa[arm] += np.outer(context, context)
        self.Aa_inv[arm] = np.linalg.inv(self.Aa[arm])
        self.b_a[arm] += reward * context.transpose()
        self.theta_a[arm] = self.Aa_inv[arm].dot(self.b_a[arm])

        self.n_shows_b[arm] += 1
        self.n_clicks_b[arm] += reward

    def get_results_csv(self, file_name):
        data = {
            "arms": [],
            "n_clicks_b": [],
            "n_shows_b": [],
            "n_clicks_r": [],
            "n_shows_r": []
        }
        for item in self.arms:
            data["arms"].append(item)
            data["n_clicks_b"].append(self.n_clicks_b[item])
            data["n_shows_b"].append(self.n_shows_b[item])
            data["n_clicks_r"].append(self.n_clicks_r[item])
            data["n_shows_r"].append(self.n_shows_r[item])

        data = pd.DataFrame(data=data)
        data.to_csv(file_name, index=False)

