from linUCB_hybrid import HybridBandit
from UCB1 import UCB1
from e_greedy import EGreedy
import numpy as np
import os
import time
import datetime


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


def evaluate(bandit, kind=None, learning=False):

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
        f_errors = open(f"/home/ruslan/PycharmProjects/group_recommender/group_recommender/big_start"
                  f"/invalid_events/errors_{kind}_in_day_{i + 1}.txt", "a")
        f_history = open(f"/home/ruslan/PycharmProjects/group_recommender/"
                  f"group_recommender/big_start/history/history_{kind}_in_day_{i + 1}.txt", "a")
        f_shows = open(f"/home/ruslan/PycharmProjects/group_recommender/"
                  f"group_recommender/big_start/history_bandit/bandit_{kind}_shows_in_day_{i + 1}.txt", "a")

        while True:
            line = f.readline()
            step += 1
            if step % 200_000 == 0:
                print(step, ls[i])
                print(f"Время с начала запуска: {datetime.datetime.now() - start_time}", datetime.datetime.now())
            if not line:
                break
            try:
                event = format_event(line)
                arm, arms, reward, user_context, group_context = event
                max_hand, rew = bandit.predict_arm(event)

                f_shows.write(str(max_hand) + "\n")

                # если угадали, то обновляем информацию по руке
                if max_hand == arm:
                    bandit.update(event)
                    f_history.write(line)
                    t += 1
                    ga += reward

                    if learning:
                        t_learning += 1
                        g_learning += reward
                    else:
                        t_test += 1
                        g_test += reward

            except ValueError:
                bandit.black_list.add(arm)
                f_errors.write(line)
        f_shows.close()
        f_history.close()
        f_errors.close()
        f.close()

        bandit.get_results_csv(f"/home/ruslan/PycharmProjects/group_recommender/"
                  f"group_recommender/big_start/results/{kind}_day_{i + 1}.csv")
        f = open(f"/home/ruslan/PycharmProjects/group_recommender/"
                  f"group_recommender/big_start/all_kinds_of_rewards/rewards_{kind}_day_{i + 1}.txt", "a")
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
    evaluate(UCB1(), "UCB1")
    evaluate(HybridBandit(2.1, 6, 6, 0.8, -15), "Hybrid")
    evaluate(EGreedy(1, 0.9), "E-Greedy")

    end = time.ctime(time.time())
    print(f"start time: {start}")
    print(f"end time: {end}")

