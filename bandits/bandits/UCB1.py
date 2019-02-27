import math
from base_bandit import BaseBandit


class UCB1(BaseBandit):

    def __init__(self):
        super().__init__()
        self.n_plays = {}
        self.n = 0
        self.cumulative_reward = {}

    def predict_arm(self, event):
        # насчитываем оценки только для рук из actual_arms
        arm, arms, reward, user_context, group_context = event

        for item in arms:
            if item not in self.arms:
                self.init_arm(item)
                return item
        else:
            payoffs = [self.upper_bound(arm) for arm in arms]

        return arms[payoffs.index(max(payoffs))]

    def upper_bound(self, arm):
        return self.cumulative_reward[arm] / self.n_plays[arm] + math.sqrt(2 * math.log(self.n + 1) / self.n_plays[arm])

    def init_arm(self, arm):
        self.arms.add(arm)
        self.n_plays.setdefault(arm, 0)
        self.cumulative_reward.setdefault(arm, 0)

    def update(self, event):
        arm, arms, reward, user_context, group_context = event
        self.n_plays[arm] += 1
        self.cumulative_reward[arm] += reward


