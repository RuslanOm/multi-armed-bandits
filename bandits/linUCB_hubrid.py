"""This class it is an implementation of Contextual Bandit Algorithm with Hybrid Linear Model

"""
import numpy as np
import pandas as pd


class HybridBandit:

    def __init__(self, alpha, size_of_user_context, size_of_group_context, r1, r0):
        """Initialization of our bandit class
        alpha: float, parameter of algorithm
        r1: float, reward, if user clicks on our article
        r0: float, reward, if user doesn't click on our article
        size_of_user_context: int, len of user context vector
        size_of_group_context: int, len of group context vector
        """
        self.alpha = alpha
        self.size_of_user_context = size_of_user_context
        self.size_of_group_context = size_of_group_context
        self.r1 = r1
        self.r0 = r0,

        # словари для хранения параметров, используемых в алгоритме; ключи - id-шники статей/групп
        self.Aa = {}
        self.Aa_inv = {}
        self.Ba = {}
        self.ba = {}
        self.za = {}

        self.A0 = np.identity(self.size_of_group_context * self.size_of_user_context)
        self.b0 = np.zeros((self.size_of_group_context * self.size_of_user_context, 1))

        self.beta_hat = None
        self.theta_hat = {}

        # словари для подсчета общей награды каждой руки; ключи - id-шники статей/групп
        self.n_shows_b = {}
        self.n_shows_r = {}

        self.n_clicks_b = {}
        self.n_clicks_r = {}

        # специальное множество для id-шников плохих групп
        self.black_list = set()

    def init_arm(self, key):
        self.Aa[key] = np.identity(self.size_of_user_context)
        self.Aa_inv[key] = np.identity(self.size_of_user_context)
        self.Ba[key] = np.zeros((self.size_of_user_context, self.size_of_group_context * self.size_of_user_context))
        self.ba[key] = np.zeros((self.size_of_user_context, 1))

        self.n_shows_b[key] = 0
        self.n_shows_r[key] = 0

        self.n_clicks_b[key] = 0
        self.n_clicks_r[key] = 0

    def get_max_hand(self, event):
        arm, arms, reward, user_context, group_context = event

        # если руку ещё не видели - инициализируем ее
        if arm not in self.Aa:
            self.init_arm(arm)

        s_tmp = {}
        p_tmp = {}

        # считаем ожидаемые награды для каждой руки из пула актуальных рук на данном шаге
        for key in arms:
            if key not in self.black_list:
                self.theta_hat[key] = np.dot(
                    np.linalg.inv(self.Aa[key]), self.ba[key] - np.dot(self.Ba[key], self.beta_hat)
                )
                self.za[key] = np.outer(user_context, group_context).reshape(-1)
                self.za[key] = np.array([list(self.za[key])]).transpose()
                # print(user_context.transpose())
                # print( np.dot(self.Aa_inv[key], user_context.transpose()))
                # print(np.dot(user_context.transpose(), np.dot(self.Aa_inv[key], user_context)))

                s_tmp[key] = np.dot(self.za[key].transpose(), np.dot(np.linalg.inv(self.A0), self.za[key])) - \
                             2 * np.dot(self.za[key].transpose(),
                                        np.dot(np.linalg.inv(self.A0),
                                               np.dot(self.Ba[key].transpose(),
                                                      np.dot(self.Aa_inv[key], user_context)))) + \
                             np.dot(user_context.transpose(), np.dot(self.Aa_inv[key], user_context)) + \
                             np.dot(user_context.transpose(),
                                    np.dot(self.Aa_inv[key],
                                           np.dot(self.Ba[key],
                                                  np.dot(np.linalg.inv(self.A0),
                                                         np.dot(self.Ba[key].transpose(),
                                                                np.dot(self.Aa_inv[key], user_context))))))
                p_tmp[key] = np.dot(self.za[key].transpose(), self.beta_hat) + \
                             np.dot(user_context.transpose(), self.theta_hat[key]) + self.alpha * np.sqrt(s_tmp[key])

        # находи максимум в этом словаре
        v = list(p_tmp.values())
        k = list(p_tmp.keys())
        return k[v.index(max(v))], max(v)

    def calc_beta_hat(self):
        self.beta_hat = np.dot(np.linalg.inv(self.A0), self.b0)

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

            self.A0 = self.A0 + np.outer(self.za[arm], self.za[arm]) -\
                  np.dot(self.Ba[arm].transpose(), np.dot(self.Aa_inv[arm], self.Ba[arm]))
            self.b0 = self.b0 + r * self.za[arm] -\
                  np.dot(self.Ba[arm].transpose(), np.dot(self.Aa_inv[arm], self.ba[arm]))

            self.n_shows_b[arm] += 1
            self.n_clicks_b[arm] += reward

    def reboot(self):
        for key in self.Aa:
            self.n_shows_b[key] = 0
            self.n_shows_r[key] = 0

            self.n_clicks_b[key] = 0
            self.n_clicks_r[key] = 0

    def get_results_csv(self, file_name):
        """All kinds of rewards
        """
        data = {
            "arms": [],
            "n_clicks_b": [],
            "n_shows_b": [],
            "n_clicks_r": [],
            "n_shows_r": [],
            "ctr_bandit": [],
            "ctr_policy": [],
            "diff": []
        }
        for item in self.Aa:
            data["arms"].append(item)
            data["n_clicks_b"].append(self.n_clicks_b[item])
            data["n_shows_b"].append(self.n_shows_b[item])
            data["n_clicks_r"].append(self.n_clicks_r[item])
            data["n_shows_r"].append(self.n_shows_r[item])
            data["ctr_bandit"].append(self.n_clicks_b[item] / self.n_shows_b[item]) if self.n_shows_b[item] != 0 \
                else data["ctr_bandit"].append(None)
            data["ctr_policy"].append(self.n_clicks_r[item] / self.n_shows_r[item]) if self.n_shows_r[item] != 0 \
                else data["ctr_policy"].append(None)
            data["diff"].append(data["ctr_bandit"][-1] - data["ctr_policy"][-1]) \
                if data["ctr_policy"][-1] is not None and data["ctr_bandit"][-1] is not None \
                else data["diff"].append(None)

        data["arms"].append("total")
        data["n_clicks_b"].append(sum(data["n_clicks_b"]))
        data["n_shows_b"].append(sum(data["n_shows_b"]))
        data["n_clicks_r"].append(sum(data["n_clicks_r"]))
        data["n_shows_r"].append(sum(data["n_shows_r"]))

        ctr_b = 0
        ctr_p = 0

        diff = 0
        counter_diff = 0

        counter_b = 0
        counter_p = 0
        for i in range(len(data["ctr_policy"])):
            ctr_b += data["ctr_bandit"][i] if data["ctr_bandit"][i] is not None else 0
            counter_b += 1 if data["ctr_bandit"][i] is not None else 0
            ctr_p += data["ctr_policy"][i] if data["ctr_policy"][i] is not None else 0
            counter_p += 1 if data["ctr_policy"][i] is not None else 0

            diff += data["diff"][i] if data["diff"][i] is not None else 0
            counter_diff += 1 if data["diff"][i] is not None else 0

        ctr_b /= counter_b if counter_b != 0 else 1
        ctr_p /= counter_p if counter_p != 0 else 1
        diff /= counter_diff if counter_diff != 0 else 1

        data["ctr_bandit"].append(ctr_b)
        data["ctr_policy"].append(ctr_p)
        data["diff"].append(diff)

        data = pd.DataFrame(data=data)
        data.to_csv(file_name, index=False)

