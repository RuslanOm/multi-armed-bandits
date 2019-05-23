"""This class it is an implementation of Contextual Bandit Algorithm with Hybrid Linear Model

"""
import numpy as np
from bandits.bandits.base_bandit import BaseBandit
import random


def get_all_max(d, epsilon=10 ** (-6)):
    # необходимо в случае, когда сразу несколько ожидаемых выплат имеют почти равные значения
    res = []

    ls = d.values()

    my_max = max(ls)

    for item in d:
        if abs(d[item] - my_max) < epsilon:
            res.append(item)
    return res


class HybridBandit(BaseBandit):

    def __init__(self, alpha, size_of_user_context, size_of_group_context, average_reward=0, r1=1, r0=0):
        """Initialization of our bandit class
        alpha: float, parameter of algorithm
        r1: float, reward, if user clicks on our article
        r0: float, reward, if user doesn't click on our article
        size_of_user_context: int, len of user context vector
        size_of_group_context: int, len of group context vector
        """
        super(HybridBandit, self).__init__()
        self.alpha = alpha
        self.size_of_user_context = size_of_user_context
        self.size_of_group_context = size_of_group_context
        self.r1 = r1
        self.r0 = r0

        # словари для хранения параметров, используемых в алгоритме; ключи - id-шники статей/групп
        self.Aa = {}
        self.Aa_inv = {}
        self.Ba = {}
        self.ba = {}
        self.za = {}
        self.BaT = {}

        self.A0inv_BaT_Aainv = {}
        self.Aainv_Ba_A0inv_BaT_Aainv = {}

        self.A0 = np.identity(self.size_of_group_context * self.size_of_user_context)
        self.b0 = np.zeros((self.size_of_group_context * self.size_of_user_context, 1))
        self.A0_inv = np.identity(self.size_of_group_context * self.size_of_user_context)

        self.beta_hat = None
        self.theta_hat = {}

        self.average_reward = average_reward
        self.regret = []
        self.rewards = 0.0
        self.n_steps = 0

    def init_arm(self, key):
        self.arms.add(key)
        self.Aa[key] = np.identity(self.size_of_user_context)
        self.Aa_inv[key] = np.identity(self.size_of_user_context)
        self.Ba[key] = np.zeros((self.size_of_user_context, self.size_of_group_context * self.size_of_user_context))
        self.ba[key] = np.zeros((self.size_of_user_context, 1))
        self.BaT[key] = self.Ba[key].T

        self.n_clicks_b.setdefault(key, 0)
        self.n_clicks_r.setdefault(key, 0)
        self.n_shows_b.setdefault(key, 0)
        self.n_shows_r.setdefault(key, 0)

    def predict_arm(self, event):
        self.calc_beta_hat()
        arm, arms, reward, user_context, groups = event

        for item in arms:
            if item not in self.arms:
                self.init_arm(item)

        # если руку ещё не видели - инициализируем ее
        if arm not in self.Aa:
            self.init_arm(arm)

        s_tmp = {}
        p_tmp = {}

        # считаем ожидаемые награды для каждой руки из пула актуальных рук на данном шаге
        for key in arms:
            self.theta_hat[key] = np.dot(
                self.Aa_inv[key], self.ba[key] - np.dot(self.Ba[key], self.beta_hat)
            )
            self.za[key] = np.outer(user_context, groups[key]).reshape(-1)
            self.za[key] = np.array([list(self.za[key])]).T

            self.A0inv_BaT_Aainv[key] = np.dot(self.A0_inv, np.dot(self.BaT[key], self.Aa_inv[key]))
            self.Aainv_Ba_A0inv_BaT_Aainv[key] = np.dot(self.Aa_inv[key],
                                                        np.dot(self.Ba[key], self.A0inv_BaT_Aainv[key]))

            s_tmp[key] = np.dot(self.za[key].T, np.dot(self.A0_inv, self.za[key])) - \
                         2 * np.dot(self.za[key].T, np.dot(self.A0inv_BaT_Aainv[key], user_context)) + \
                         np.dot(user_context.T, np.dot(self.Aa_inv[key], user_context)) + \
                         np.dot(user_context.T, np.dot(self.Aainv_Ba_A0inv_BaT_Aainv[key], user_context))

            p_tmp[key] = np.dot(self.za[key].T, self.beta_hat) + \
                         np.dot(user_context.T, self.theta_hat[key]) + self.alpha * np.sqrt(s_tmp[key])

        # находим множество рук, которые мог бы показать алгоритм на данном шаге (их p_tmp отличается не больше, чем на
        # epsilon
        valid_arms = get_all_max(p_tmp)

        return random.choice(valid_arms)

    def calc_beta_hat(self):
        self.beta_hat = np.dot(self.A0_inv, self.b0)

    def update(self, event):
        arm, reward, user_context = event

        if reward == -1:
            pass
        else:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0

            self.A0 = self.A0 + np.dot(self.Ba[arm].T, np.dot(self.Aa_inv[arm], self.Ba[arm]))
            self.b0 = self.b0 + np.dot(self.Ba[arm].T, np.dot(self.Aa_inv[arm], self.ba[arm]))

            self.Aa[arm] = self.Aa[arm] + np.outer(user_context, user_context)
            self.Ba[arm] = self.Ba[arm] + np.dot(user_context, self.za[arm].T)
            self.ba[arm] = self.ba[arm] + r * user_context
            self.BaT[arm] = self.Ba[arm].T
            self.Aa_inv[arm] = np.linalg.inv(self.Aa[arm])

            self.A0 = self.A0 + np.outer(self.za[arm], self.za[arm]) - \
                      np.dot(self.Ba[arm].T, np.dot(self.Aa_inv[arm], self.Ba[arm]))
            self.b0 = self.b0 + r * self.za[arm] - \
                      np.dot(self.Ba[arm].T, np.dot(self.Aa_inv[arm], self.ba[arm]))

            self.A0_inv = np.linalg.inv(self.A0)

            self.n_shows_b[arm] += 1
            self.n_clicks_b[arm] += reward

            self.n_steps += 1
            self.rewards += reward

            self.regret.append(self.n_steps * self.average_reward - self.rewards)






