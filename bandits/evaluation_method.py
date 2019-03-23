from linUCB_hybrid import HybridBandit
from UCB1 import UCB1
from e_greedy import EGreedy
from linUCB_disjoint import DisjointBandit
from base_bandit import BaseBandit
import numpy as np
import os
import time
import datetime
import pandas as pd
import sklearn.decomposition as sd
from sklearn.utils import shuffle


def format_event(line):
    # форматирование событий из лог-файла
    line = line.strip().split("|")
    arm = line[0].split()[1]
    reward = int(line[0].split()[2])

    # контекст пользователя
    user_prom = line[1].split()[1:-1]
    user_context = [float(item) for _, item in list(map(lambda x: x.split(":"), user_prom))]
    user_context.insert(0, 1.0)
    user_context = np.array([user_context]).transpose()

    # онтекст группы
    group_cont = line[2:]
    list_of_groups = [item.split()[0] for item in group_cont]
    # print(list_of_groups)
    # print(len(arm))
    group_ind = list_of_groups.index(arm)
    group_cont = group_cont[group_ind].split()[1:-1]
    group_context = [float(item) for _, item in list(map(lambda x: x.split(":"), group_cont))]
    group_context.insert(0, 1.0)
    group_context = np.array([group_context])

    # список актуальный рук
    arms = [item.split()[0] for item in line[2:]]

    return arm, arms, reward, user_context, group_context


def read_events(f):
    line = f.readline()
    res = []
    if not line:
        return None
    res.append(line)
    while len(res) < 200_000 and line:
        line = f.readline()
        res.append(line)
    res.reverse()
    return res


def pca(views, groups):
    views = shuffle(views)
    groups = groups.fillna(0)

    user_ids = list(views["userId"])
    groups_ids = list(views["groupId"])
    label = list(views["label"])
    ls = []
    for item in label:
        ls.append(1) if item == "Liked" else ls.append(0)
    label = ls

    l_view = list(views)[6:]
    l_groups = list(groups)[1:]

    x_views = views[l_view]
    x_groups = groups[l_groups]

    pc = sd.PCA(n_components=12)

    n_views = pc.fit_transform(x_views)
    n_groups = pc.transform(x_groups)

    res_views = [np.array([list(item)]).transpose() for item in n_views]
    res_groups = [np.array([list(item)]) for item in n_groups]

    d_groups = {}
    for i in range(len(list(groups["groupId"]))):
        d_groups[groups["groupId"][i]] = res_groups[i]

    return label, groups_ids, res_views, d_groups


def evaluate_csv(bandit, kind):
    assert isinstance(bandit, BaseBandit), "Wrong first argument"

    # общая награда за весь прогон по всем файлам
    ga = 0
    t = 0

    views = pd.read_csv("dataOKcsv/top100groups_views.csv")
    groups = pd.read_csv("dataOKcsv/groups_formated.csv")

    res, arm, event, arms = pca(views, groups)
    set_arms = set(arm)
    print(len(res), len(set_arms), len(event), len(arms), len(arm))

    # счетчик шагов
    step = 0

    start_time = datetime.datetime.now()
    f_errors = open(f"ok_test/errors_{kind}_in_day_.txt", "a")
    f_history = open(f"ok_test/history_{kind}_in_day_.txt", "a")
    f_shows = open(f"ok_test/bandit_{kind}_shows_in_day_.txt", "a")
    error_buff = []
    history_buff = []
    shows_buff = []

    for i in range(len(res)):

        # f - лог-файлы; f_errors - файл для записи событий, когда произошли ошибки; f_history - файл событий, когда
        # произошло совпадение выдачи политики и бандитов; f_shows - файл для записи истории показов бандитов

        step += 1
        if step % 1000 == 0:
            print(step)
            print(f"Время с начала запуска: {datetime.datetime.now() - start_time}", datetime.datetime.now())
        max_hand = bandit.predict_arm((arm[i], set_arms, res[i], event[i], arms[arm[i]]))
        shows_buff.append(str(max_hand) + "\n")
        if len(shows_buff) > 200_000:
            f_shows.writelines(shows_buff)
            shows_buff.clear()

        bandit.n_shows_r[arm[i]] += 1
        bandit.n_clicks_r[arm[i]] += res[i]

        # если угадали, то обновляем информацию по руке
        if max_hand == arm[i]:
            bandit.update((arm[i], set_arms, res[i], event[i], arms[arm[i]]))
            history_buff.append(str(arm[i]))
            if len(history_buff) > 200_000:
                f_history.writelines(history_buff)
                history_buff.clear()
            t += 1
            ga += res[i]

        # except ValueError as e:
        #     print(e)
        #     error_buff.append(str(arm[i]))
        #     if len(error_buff) > 200_000:
        #         f_errors.writelines(error_buff)
        #         error_buff.clear()

    print(len(shows_buff), len(history_buff), len(error_buff))

    f_shows.writelines(shows_buff)
    f_errors.writelines(error_buff)
    f_history.writelines(history_buff)

    f_shows.close()
    f_history.close()
    f_errors.close()
    bandit.get_results_csv(f"ok_test/{kind}_day_.csv")
    f = open(f"ok_test/rewards_{kind}_day_.txt", "a")
    f.write(f"total reward " + str(ga / t) + " " + str(ga) + " " + str(t))

    f.close()

    print(f"Полное время: {datetime.datetime.now() - start_time}")


def evaluate(bandit, kind=None, learning=False):
    assert isinstance(bandit, BaseBandit), "Wrong first argument"

    # общая награда за весь прогон по всем файлам
    ga = 0
    t = 0

    # награда на обучении
    g_learning = 0
    t_learning = 0

    # награда на тесте
    g_test = 0
    t_test = 0

    path = "/home/ruslan/PycharmProjects/group_recommender/group_recommender/bandittts_zip/bandittts"
    ls = os.listdir(path)
    ls.sort()

    # счетчик шагов
    step = 0

    start_time = datetime.datetime.now()

    for i in range(len(ls)):

        # f - лог-файлы; f_errors - файл для записи событий, когда произошли ошибки; f_history - файл событий, когда
        # произошло совпадение выдачи политики и бандитов; f_shows - файл для записи истории показов бандитов
        f = open(path + "/" + ls[i], "r")
        f_errors = open(f"/home/ruslan/PycharmProjects/group_recommender/group_recommender/data_for_pm"
                        f"/hybrid/invalid_events/errors_{kind}_in_day_{i + 1}.txt", "a")
        f_history = open(f"/home/ruslan/PycharmProjects/group_recommender/"
                         f"group_recommender/data_for_pm//hybrid/history/history_{kind}_in_day_{i + 1}.txt", "a")
        f_shows = open(f"/home/ruslan/PycharmProjects/group_recommender/"
                       f"group_recommender/data_for_pm/hybrid/history_bandit/bandit_{kind}_shows_in_day_{i + 1}.txt",
                       "a")

        buffer = read_events(f)
        error_buff = []
        history_buff = []
        shows_buff = []
        while True:
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
                    arm, arms, reward, user_context, group_context = event
                    max_hand = bandit.predict_arm(event)
                    shows_buff.append(str(max_hand) + "\n")
                    if len(shows_buff) > 200_000:
                        f_shows.writelines(shows_buff)
                        shows_buff.clear()

                    bandit.n_shows_r[arm] += 1
                    bandit.n_clicks_r[arm] += reward

                    # если угадали, то обновляем информацию по руке
                    if max_hand == arm:
                        bandit.update(event)
                        history_buff.append(str(arm) + "\n")
                        if len(history_buff) > 200_000:
                            f_history.writelines(history_buff)
                            history_buff.clear()
                        t += 1
                        ga += reward

                        if learning and i == 0:
                            t_learning += 1
                            g_learning += reward
                        else:
                            t_test += 1
                            g_test += reward

                except ValueError:
                    error_buff.append(line)
                    if len(error_buff) > 200_000:
                        f_errors.writelines(error_buff)
                        error_buff.clear()

            buffer = read_events(f)

        f_shows.writelines(shows_buff)
        f_errors.writelines(error_buff)
        f_history.writelines(history_buff)

        f_shows.close()
        f_history.close()
        f_errors.close()
        f.close()

        bandit.get_results_csv(f"/home/ruslan/PycharmProjects/group_recommender/"
                               f"group_recommender/data_for_pm//hybrid/results/{kind}_day_{i + 1}.csv")
        f = open(f"/home/ruslan/PycharmProjects/group_recommender/"
                 f"group_recommender/data_for_pm/hybrid/all_kinds_of_rewards/rewards_{kind}_day_{i + 1}.txt", "a")
        if learning and i == 0:
            f.write(f"total reward " + str(g_learning / t_learning) + " " + str(g_learning) + " " + str(t_learning))

            # сброс счетчиков кликов и показов после обучения
            bandit.reboot()
        else:
            f.write(f"total reward " + str(g_test / t_test) + " " + str(g_test) + " " + str(t_test))

        f.close()
    print(f"Полное время: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    start = time.ctime(time.time())
    # evaluate(UCB1(0.1), "UCB1")
    # evaluate(HybridBandit(2.1, 6, 6, 0.8, -15), "Hybrid", True)
    # evaluate(EGreedy(0.1), "E-Greedy-const")
    # evaluate(DisjointBandit(2.1, 6, 0.8, -15), "D-Bandit", True)
    # l = np.linspace(1.8, 2, 2)
    # for item in l:
    #     evaluate_csv(HybridBandit(item, 12, 12, 0.8, -15), f"Hybrid_{item}")

    evaluate(HybridBandit(1.6, 6, 6, 0.8, -15), f"Hybrid_{0.8}", True)
    end = time.ctime(time.time())
    print(f"start time: {start}")
    print(f"end time: {end}")
