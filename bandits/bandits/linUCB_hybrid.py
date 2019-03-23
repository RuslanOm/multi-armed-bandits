"""This class it is an implementation of Contextual Bandit Algorithm with Hybrid Linear Model

"""
import numpy as np
from base_bandit import BaseBandit


class HybridBandit(BaseBandit):

    def __init__(self, alpha, size_of_user_context, size_of_group_context, r1, r0):
        """Initialization of our bandit class
        alpha: float, parameter of algorithm
        r1: float, reward, if user clicks on our article
        r0: float, reward, if user doesn't click on our article
        size_of_user_context: int, len of user context vector
        size_of_group_context: int, len of group context vector
        """
        super().__init__()
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

    def init_arm(self, key):
        self.arms.add(key)
        self.Aa[key] = np.identity(self.size_of_user_context)
        self.Aa_inv[key] = np.identity(self.size_of_user_context)
        self.Ba[key] = np.zeros((self.size_of_user_context, self.size_of_group_context * self.size_of_user_context))
        self.ba[key] = np.zeros((self.size_of_user_context, 1))
        self.BaT[key] = self.Ba[key].transpose()

        self.n_clicks_b.setdefault(key, 0)
        self.n_clicks_r.setdefault(key, 0)
        self.n_shows_b.setdefault(key, 0)
        self.n_shows_r.setdefault(key, 0)

    def predict_arm(self, event):
        self.calc_beta_hat()
        arm, arms, reward, user_context, group_context = event
        # print(arm, arms, reward, user_context, group_context)

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
            self.za[key] = np.outer(user_context, group_context).reshape(-1)
            self.za[key] = np.array([list(self.za[key])]).transpose()
            # print(self.za)
            # print(user_context.transpose())
            # print( np.dot(self.Aa_inv[key], user_context.transpose()))
            # print(np.dot(user_context.transpose(), np.dot(self.Aa_inv[key], user_context)))

            self.A0inv_BaT_Aainv[key] = np.dot(self.A0_inv, np.dot(self.BaT[key], self.Aa_inv[key]))
            self.Aainv_Ba_A0inv_BaT_Aainv[key] = np.dot(self.Aa_inv[key],
                                                        np.dot(self.Ba[key], self.A0inv_BaT_Aainv[key]))

            # print(np.dot(self.Aa_inv[key], user_context))
            # print(np.dot(np.dot(user_context.transpose(), self.Aa_inv[key]), user_context))
            s_tmp[key] = np.dot(self.za[key].transpose(), np.dot(self.A0_inv, self.za[key])) - \
                         2 * np.dot(self.za[key].transpose(), np.dot(self.A0inv_BaT_Aainv[key], user_context)) + \
                         np.dot(user_context.transpose(), np.dot(self.Aa_inv[key], user_context)) + \
                         np.dot(user_context.transpose(), np.dot(self.Aainv_Ba_A0inv_BaT_Aainv[key], user_context))

            p_tmp[key] = np.dot(self.za[key].transpose(), self.beta_hat) + \
                         np.dot(user_context.transpose(), self.theta_hat[key]) + self.alpha * np.sqrt(s_tmp[key])

        # находи максимум в этом словаре
        v = list(p_tmp.values())
        k = list(p_tmp.keys())
        return k[v.index(max(v))]

    def calc_beta_hat(self):
        self.beta_hat = np.dot(self.A0_inv, self.b0)

    def update(self, event):

        arm, arms, reward, user_context, group_context = event

        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0

            self.A0 = self.A0 + np.dot(self.Ba[arm].transpose(), np.dot(self.Aa_inv[arm], self.Ba[arm]))
            self.b0 = self.b0 + np.dot(self.Ba[arm].transpose(), np.dot(self.Aa_inv[arm], self.ba[arm]))

            self.Aa[arm] = self.Aa[arm] + np.outer(user_context, user_context)
            self.Ba[arm] = self.Ba[arm] + np.dot(user_context, self.za[arm].transpose())
            self.ba[arm] = self.ba[arm] + r * user_context
            self.BaT[arm] = self.Ba[arm].transpose()
            self.Aa_inv[arm] = np.linalg.inv(self.Aa[arm])

            self.A0 = self.A0 + np.outer(self.za[arm], self.za[arm]) -\
                  np.dot(self.Ba[arm].transpose(), np.dot(self.Aa_inv[arm], self.Ba[arm]))
            self.b0 = self.b0 + r * self.za[arm] -\
                  np.dot(self.Ba[arm].transpose(), np.dot(self.Aa_inv[arm], self.ba[arm]))
            self.A0_inv = np.linalg.inv(self.A0)

            self.n_shows_b[arm] += 1
            self.n_clicks_b[arm] += reward




