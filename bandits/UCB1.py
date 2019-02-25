import math


class UCB1:

    def __init__(self):
        self.hands = {}
        self.n_plays = {}
        self.n = 0
        self.cumulative_reward = {}

    def predict(self, actual_arms):
        # насчитываем оценки только для рук из actual_arms
        payoffs = [self.upper_bound(arm) for arm in actual_arms]

        return actual_arms[payoffs.index(max(payoffs))]

    def upper_bound(self, arm):
        return self.cumulative_reward[arm] / self.n_plays[arm] + math.sqrt(2 * math.log(self.n + 1) / self.n_plays[arm])

    def update(self, arm, reward):
        # если нету ещё руки, то устанавливаем значения счетчиков на 0
        self.n_plays.setdefault(arm, 0)
        self.cumulative_reward.setdefault(arm, 0)

        self.n_plays[arm] += 1
        self.cumulative_reward[arm] += reward
