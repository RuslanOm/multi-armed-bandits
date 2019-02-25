from UCB1 import UCB1


def format_event(line):
    # форматируем строки событий, идущих из лог-файла
    line = line.strip().split("|")
    arm = line[0].split()[1]
    reward = int(line[0].split()[2])

    arms = [item.split()[0] for item in line[2:]]

    return arm, reward, arms


def evaluate():
    # заполнить самостоятельно =)
    path = ""
    other_path = ""

    bandit = UCB1()

    f = open(path, "r")
    f_bandit = open(other_path, "a")

    while True:
        line = f.readline()
        if not line:
            break

        arm, reward, arms = format_event(line)

        # если рука новая, то пытаемся ее дергать до тех пор, пока не совпадет выдача
        # когда совпадает, инициализируем параметры для этой руки

        for item in arms:
            if item not in bandit.hands and item != arm:
                break
            elif item not in bandit.hands and item == arm:
                bandit.update(arm, reward)
                f_bandit.write(arm + "\n")
                break
        else:
            # если же все руки старые и проинициализированы, то по честному считаем upperbound и выдаем максимум
            # если совпало, то обновляем соответствующие счетчики
            res_arm = bandit.predict(arms)
            if res_arm == arm:
                bandit.update(arm, reward)
                f_bandit.write(arm + "\n")



