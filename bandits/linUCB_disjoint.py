"""This class it is an implementation of Contextual Bandit Algorithm with Disjoint Linear Model

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ContextBandit:

    def __init__(self, alpha):
        """Initialization of our bandit class
        alpha: float, parameter of algorithm
        """
        self.alpha = alpha

    def read_txt(self, directory):
        """Format of txt file: id of group/article/arm, reward, user context
        """
        with open(directory, "r") as f:
            data = f.readlines()

        arms = set()
        data = [item.split() for item in data]

        user_log = []
        for item in data:
            arms.add(item[0])
            user_log.append([item[0], int(item[1]), np.array([list(map(int, item[2:]))])])

        np.random.shuffle(user_log)
        self.arms = arms
        self.user_log = user_log
        self.n_context = len(item[2:])

        self.CTR = {key: {} for key in self.arms}
        self.CTR_relative = {key: {} for key in self.arms}

        self.ctr_num = {key: 0 for key in self.arms}
        self.ctr_den = {key: 0 for key in self.arms}

    def read_csv(self, directory):
        """Format of csv file: id of group/article/arm, reward, user context(ndarray (1, n_context))
        """
        data = pd.read_csv(directory)

        self.user_log = list(zip(data["group"], data["reward"], data["context"]))
        np.random.shuffle(self.user_log)

        self.arms = set(data["group"])
        self.n_context = len(data["context"][0])

        self.CTR = {key: {} for key in self.arms}
        self.CTR_relative = {key: {} for key in self.arms}

        self.ctr_num = {key: 0 for key in self.arms}
        self.ctr_den = {key: 0 for key in self.arms}

    def fit(self):
        time = 0
        self.Aa = {key: np.identity(self.n_context) for key in self.arms}
        self.Aa_inv = {key: np.identity(self.n_context) for key in self.arms}
        self.b_a = {key: np.zeros((self.n_context, 1)) for key in self.arms}
        self.theta_a = {key: np.zeros((self.n_context, 1)) for key in self.arms}

        for arm, rew, us in self.user_log:
            payoffs = {key: -1 for key in self.Aa}

            mx = -1000000
            ind = -1

            for key in self.Aa:
                self.theta_a[key] = np.dot(self.Aa_inv[key], self.b_a[key])
                payoffs[key] = np.dot(us, self.theta_a[key]) + \
                               self.alpha * np.sqrt(np.dot(np.dot(us, self.Aa_inv[key]), us.transpose()))
                ind = ind if mx > payoffs[key] else key
                mx = mx if mx > payoffs[key] else payoffs[key]

            pred_arm = ind

            if arm == pred_arm:
                self.Aa[pred_arm] += np.outer(us, us)
                self.Aa_inv[pred_arm] = np.linalg.inv(self.Aa[pred_arm])
                self.b_a[pred_arm] += rew * us.transpose()
                self.theta_a[pred_arm] = self.Aa_inv[pred_arm].dot(self.b_a[pred_arm])

                curr_CTR, self.ctr_num[arm], self.ctr_den[arm] = \
                    self.calc_CTR(rew, self.ctr_num[arm], self.ctr_den[arm])

                self.CTR[arm][time] = curr_CTR
                self.CTR_relative[arm][time] = curr_CTR / self.CTR_random[arm]\
                    if self.CTR_random[arm] > 0 else curr_CTR / 1
            time += 1

    def calc_CTR(self, reward, ctr_num, ctr_den):
        ctr_num = ctr_num + reward
        ctr_den = ctr_den + 1

        ctr = ctr_num / ctr_den

        return ctr, ctr_num, ctr_den

    def calc_random_ctr(self):
        # CTR of random policy
        n_shows = {key: 0 for key in self.arms}
        rews = {key: 0 for key in self.arms}

        for arm, rew, _ in self.user_log:
            n_shows[arm] += 1
            rews[arm] += rew

        self.CTR_random = {arm: rews[arm] / n_shows[arm] for arm in self.arms}

    def plot_graphs(self, ls=None):
        ls = self.arms if ls is None else ls
        plt.grid()
        for item in ls:
            lists = sorted(self.CTR_relative[item].items())
            x, y = zip(*lists)
            plt.plot(x, y)
        plt.show()
