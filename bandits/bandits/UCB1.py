"""This class is an implementation UCB1 algorithm from https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

"""
import math
from bandits.bandits.base_bandit import BaseBandit


class UCB1(BaseBandit):

    def __init__(self, alpha, average_reward=0):
        super().__init__()
        self.alpha = alpha
        self.n_plays = {}
        self.n = 0
        self.cumulative_reward = {}

        self.average_reward = average_reward

    def predict_arm(self, event):
        # насчитываем оценки только для рук из actual_arms
        arm, arms, reward, user_context, groups = event
        for item in arms:
            if item not in self.arms:
                self.init_arm(item)
        for item in arms:
            if item not in self.arms:
                return item
        else:
            payoffs = [self.upper_bound(arm) for arm in arms]

        return arms[payoffs.index(max(payoffs))]

    def upper_bound(self, arm):
        return self.cumulative_reward[arm] / self.n_plays[arm] + self.alpha * \
               math.sqrt(2 * math.log(self.n + 1) / self.n_plays[arm])

    def init_arm(self, arm):
        self.arms.add(arm)
        self.n_plays.setdefault(arm, 1)
        self.cumulative_reward.setdefault(arm, 0)

        self.n_clicks_b.setdefault(arm, 0)
        self.n_clicks_r.setdefault(arm, 0)
        self.n_shows_b.setdefault(arm, 0)
        self.n_shows_r.setdefault(arm, 0)

    def update(self, event):
        arm, arms, reward, user_context, group_context = event

        self.n_plays[arm] += 1
        self.n += 1
        self.cumulative_reward[arm] += reward

        self.n_shows_b[arm] += 1
        self.n_clicks_b[arm] += reward

        self.n_steps += 1
        self.rewards += reward

        self.regret.append(self.n_steps * self.average_reward - self.rewards)
