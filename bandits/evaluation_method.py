"""Implementation of offline-evaluation method from http://proceedings.mlr.press/v26/li12a/li12a.pdf"""
import numpy as np
import os
import time
import datetime
import pickle
import pandas as pd
import sklearn.decomposition as sd

from bandits.bandits.linUCB_hybrid import HybridBandit
from bandits.bandits.UCB1 import UCB1
from bandits.bandits.e_greedy import EGreedy
from bandits.bandits.linUCB_disjoint import DisjointBandit
from bandits.bandits.base_bandit import BaseBandit
from sklearn.utils import shuffle


def format_event(line):
    # форматирование событий из лог-файла
    # данный способ форматирования приведен для вида событий, которые представлены в датасете из README
    line = line.strip().split("|")
    arm = line[0].split()[1]
    reward = int(line[0].split()[2])

    # контекст пользователя
    user_prom = line[1].split()[1:-1]
    user_context = [float(item) for _, item in list(map(lambda x: x.split(":"), user_prom))]
    user_context.insert(0, 1.0)
    user_context = np.array([user_context]).transpose()

    groups = {}

    # контексты групп
    group_cont = line[2:]
    for item in group_cont:
        prom = item.split()
        ind = prom[0]
        cont = prom[1:-1]
        group_context = [float(item) for _, item in list(map(lambda x: x.split(":"), cont))]
        group_context.insert(0, 1.0)
        groups[ind] = np.array([group_context])

    # список актуальный рук
    arms = [item.split()[0] for item in line[2:]]

    return arm, arms, reward, user_context, groups


def read_events(f, count=200_000):
    # чтение событий реализовано буфферным способом, чтобы не засорять оперативную память читая весь файл целиком
    # размеры файлов могут достигать нескольких Гб
    line = f.readline()
    res = []
    if not line:
        return None
    res.append(line)
    while len(res) < count and line:
        line = f.readline()
        res.append(line)

    # нужно, чтобы получение очередного события в evaluate занимало не O(count) времени, а O(1)
    res.reverse()
    return res


def evaluate(bandit, kind=None, learning=False, n_learning_files=1):
    assert isinstance(bandit, BaseBandit), "Wrong first argument"

    # награда на обучении
    g_learning = 0
    t_learning = 0

    # награда на тесте
    g_test = 0
    t_test = 0

    path = "/path/to/data"
    ls = os.listdir(path)

    # счетчик шагов
    step = 0

    # время запуска
    start_time = datetime.datetime.now()

    path_for_results = "..."

    for i in range(len(ls)):
        f = open(path + "/" + ls[i], "r")
        buffer = read_events(f)
        while True:
            # работа метода длится до тех пор, пока есть непросмотренные события
            if buffer is None:
                break

            while buffer:
                line = buffer.pop()
                step += 1
                if step % 200_000 == 0:
                    print(step, ls[i])
                    print(f"Время с начала запуска: {datetime.datetime.now() - start_time}", datetime.datetime.now())
                if not line:
                    break
                try:
                    event = format_event(line)
                    arm, arms, reward, user_context, groups = event
                    max_hand = bandit.predict_arm(event)
                    bandit.n_shows_r[arm] += 1
                    bandit.n_clicks_r[arm] += reward

                    # если угадали, то обновляем информацию по руке
                    if max_hand == arm:
                        bandit.update((arm, reward, user_context))

                        # некоторые алгоритмы не подразумевают обучения (для них learning = False)
                        # подразумевается, что обучение происходит на первом файле с событиями (т.е. события за 1 день)
                        # а остальные файлы - для тестирования
                        if learning and i < n_learning_files:
                            t_learning += 1
                            g_learning += reward
                        else:
                            t_test += 1
                            g_test += reward

                except ValueError:
                    print(line)
            buffer = read_events(f)

        f.close()

        bandit.get_results_csv(path_for_results + f"/{kind}_day_{i + 1}.csv")
        f = open(path_for_results + f"/rewards_{kind}_day_{i + 1}.txt", "a")
        if learning and i < n_learning_files:
            f.write(f"total reward " + str(g_learning / t_learning) + " " + str(g_learning) + " " + str(t_learning))

            # сброс счетчиков кликов и показов после обучения
            bandit.reboot()
        else:
            f.write(f"total reward " + str(g_test / t_test) + " " + str(g_test) + " " + str(t_test))

        f.close()
    print(f"Полное время: {datetime.datetime.now() - start_time}")
    return bandit


if __name__ == "__main__":
    print(f"Started at {time.ctime(time.time())}")
    bandit = HybridBandit(1.2, 6, 6, 0.0358)
    evaluate(bandit)
    path_bandit = ".../bandit.pickle"
    with open(path_bandit, 'wb') as f:
        pickle.dump(bandit, f)


