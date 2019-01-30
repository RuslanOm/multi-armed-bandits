"""This class it is an implementation of Contextual Bandit Algorithm with Disjoint Linear Model

"""
import numpy as np


class ContextBandit:

    def __init__(self, d, groups, users, alpha):
        """Initialization of our bandit class

        d: int, dimension of context vectors of users and groups

        groups: dict(groupId: group_context), the dictionary of group context_vectors

        users: list(real_reward, groupId, user_context), the list of tuples, which imitates real process

        alpha: float, parameter of algorithm
        """
        self.users = users
        self.groups = groups
        self.alpha = alpha
        self._d = d

        # матрицы, прямая и обратная, соответствующие группам

        self._Aa = {key: np.dot(groups[key].transpose, groups[key]) for key in groups}
        self._Aa_inv = {key: np.linalg.inv(self._Aa[key]) for key in self._Aa}

        # вот здесь вот тонкий момент с инициализацией, поскольку у них изначально весь контекст по статьям прям голый
        # и нулевой вектор естественным образом нормально вписывается, но у нас уже есть какая-то стартовая информация
        # по поводу групп и это уже отражено в матрице Aa, но никак не отражается в векторе b_a; так и оставить пока?

        self._b_a = {key: np.zeros(self._d, 1) for key in groups}
        self.theta_a = {key: np.zeros(self._d, 1) for key in groups}

    def recommend(self):
        for reward, group, context in self.users:
            # инициализируем словарь ожидаемых "выплат" для каждой руки
            payoffs = {key: -1 for key in self._Aa}

            # будем считать максимальную выплату сразу же по мере их подсчета
            my_max = -100000000
            max_key = -1

            # просто цикл из алгоритма для каждой руки
            for key in self._Aa_inv:
                self.theta_a[key] = np.dot(self._Aa_inv[key], self._b_a[key])
                payoffs[key] = self.theta_a[key] * context + \
                               self.alpha * np.sqrt(np.dot(np.dot(context.transpose(), self._Aa_inv[key]), context))
                max_key = max_key if payoffs[key] < my_max else key
                my_max = my_max if payoffs[key] < my_max else payoffs[key]

            # если угадали с выбором, то обновляем соотвествующие матрицы и вектора
            if max_key == group:
                self.update(max_key, reward, context)

    def update(self, key, reward, context):
        self._Aa[key] += np.dot(context.transpose(), context)
        self._Aa_inv[key] = np.linalg.inv(self._Aa[key])
        self._b_a += reward * context

    def calc_ctr(self):
        # пока ещё не придумал как подсчитывать
        pass

    # добавление группы в пул групп
    def add_arm(self, group, context=None):
        if context is None:
            self.groups[group] = None
            self._Aa[group] = np.identity(self._d)
            self._Aa_inv[group] = np.identity(self._d)
            self._b_a[group] = np.zeros((self._d, 1))
            self.theta_a[group] = np.zeros((self._d, 1))

    # удаление группы из пула групп
    def delete_arm(self, group):
        del self.groups[group]

