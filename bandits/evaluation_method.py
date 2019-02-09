from linUCB_disjiont import ContextBandit
import pandas as pd
import numpy as np


def read_txt(path):
    """Format of txt file: id of group/article/arm, reward, user context
    """
    with open(path, "r") as f:
        data = f.readlines()

    arms = set()
    data = [item.split() for item in data]

    stream = []
    for item in data:
        arms.add(item[0])
        stream.append([item[0], int(item[1]), np.array([list(map(int, item[3:]))])])

    return arms, len(item[3:]), stream


def read_csv(path):
    """Format of csv file: id of group/article/arm, reward, user context(ndarray (1, n_context))
    """
    data = pd.read_csv(path)
    stream = list(zip(data["group"], data["reward"], data["context"]))
    return set(data["group"]), len(data["context"][0]), stream


def evaluate(alpha, path, file="txt"):

    if file == "txt":
        arms, n_context, stream = read_txt(path)
    else:
        arms, n_context, stream = read_csv(path)
    bandit = ContextBandit(alpha, n_context, arms)

    h = []
    Ga = 0
    T = 0

    for arm, reward, context in stream:
        pred_arm, rew, _ = bandit.calc_payoffs(context)

        bandit.n_shows_r[arm] += 1
        bandit.n_clicks_r[arm] += reward

        if pred_arm == arm:
            bandit.update(context, reward, arm)
            h.append((arm, reward, context))
            T += 1
            Ga += reward

    bandit.get_results_csv(f"data/alpha{alpha}.csv")
    ans = [str(arm) + " " + str(reward) + " " + " ".join(list(map(str, list(context))))
           + "\n" for arm, reward, context in h]
    f = open(f"data/history{alpha}.txt", "w")
    f.write(" ".join(ans))
    f.close()

    del bandit
    del stream
    return Ga / T


if __name__ == "__main__":
    ls = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.3]
    res = [evaluate(item, "data/top22arms_for2days.txt") for item in ls]
    f = open("total_rewards.txt", "w")
    f.write(" ".join(list(map(str, res))))
    f.close()
