"""Parent class for all bandit algorithms"""
import pandas as pd
import pickle


class BaseBandit:

    def __init__(self):
        # множество рук
        self.arms = set()

        # словари счетчиков для подсчетов CTR
        self.n_shows_b = {}
        self.n_shows_r = {}

        self.n_clicks_b = {}
        self.n_clicks_r = {}

        self.average_reward = 0
        self.regret = []
        self.rewards = 0.0
        self.n_steps = 0

    def predict_arm(self, event):
        """Method for predicting arm for the step"""
        pass

    def update(self, event):
        """Updating internal parameters of algorithm"""
        pass

    def init_arm(self, arm):
        """Initiating parameters of algorithm for a new arm"""
        pass

    def reboot(self):
        """Updating counters after learning period"""
        for key in self.arms:
            self.n_shows_b[key] = 0
            self.n_shows_r[key] = 0

            self.n_clicks_b[key] = 0
            self.n_clicks_r[key] = 0

    def get_results_csv(self, file_name):
        """All kinds of rewards
        """
        data = {
            "arms": [],
            "n_clicks_b": [],
            "n_shows_b": [],
            "n_clicks_r": [],
            "n_shows_r": [],
            "ctr_bandit": [],
            "ctr_policy": [],
            "diff": []
        }
        print(self.n_clicks_b)
        print(self.n_clicks_r)
        print(self.n_shows_b)
        print(self.n_shows_r)
        print(self.arms)
        for item in self.arms:
            data["arms"].append(item)
            data["n_clicks_b"].append(self.n_clicks_b[item])
            data["n_shows_b"].append(self.n_shows_b[item])
            data["n_clicks_r"].append(self.n_clicks_r[item])
            data["n_shows_r"].append(self.n_shows_r[item])
            data["ctr_bandit"].append(self.n_clicks_b[item] / self.n_shows_b[item]) if self.n_shows_b[item] != 0 \
                else data["ctr_bandit"].append(None)
            data["ctr_policy"].append(self.n_clicks_r[item] / self.n_shows_r[item]) if self.n_shows_r[item] != 0 \
                else data["ctr_policy"].append(None)
            data["diff"].append(data["ctr_bandit"][-1] - data["ctr_policy"][-1]) \
                if data["ctr_policy"][-1] is not None and data["ctr_bandit"][-1] is not None \
                else data["diff"].append(None)

        data["arms"].append("total")
        data["n_clicks_b"].append(sum(data["n_clicks_b"]))
        data["n_shows_b"].append(sum(data["n_shows_b"]))
        data["n_clicks_r"].append(sum(data["n_clicks_r"]))
        data["n_shows_r"].append(sum(data["n_shows_r"]))

        ctr_b = 0
        ctr_p = 0

        diff = 0
        counter_diff = 0

        counter_b = 0
        counter_p = 0
        for i in range(len(data["ctr_policy"])):
            ctr_b += data["ctr_bandit"][i] if data["ctr_bandit"][i] is not None else 0
            counter_b += 1 if data["ctr_bandit"][i] is not None else 0
            ctr_p += data["ctr_policy"][i] if data["ctr_policy"][i] is not None else 0
            counter_p += 1 if data["ctr_policy"][i] is not None else 0

            diff += data["diff"][i] if data["diff"][i] is not None else 0
            counter_diff += 1 if data["diff"][i] is not None else 0

        ctr_b /= counter_b if counter_b != 0 else 1
        ctr_p /= counter_p if counter_p != 0 else 1
        diff /= counter_diff if counter_diff != 0 else 1

        data["ctr_bandit"].append(ctr_b)
        data["ctr_policy"].append(ctr_p)
        data["diff"].append(diff)

        data = pd.DataFrame(data=data)
        data.to_csv(file_name, index=False)

    def dump(self, file_name):
        """Saving a bandit like byte file"""
        # file_name должно иметь расширение .pickle

        with open(file_name, "wb") as f:
            pickle.dump(self, f)
