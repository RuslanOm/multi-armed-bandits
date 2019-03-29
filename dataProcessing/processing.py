from log_reg import LogReg
import pandas as pd
import numpy as np
import sklearn.decomposition as sd
import random

#
# def func(path, users):
#     df = pd.read_csv(path)
#     u = pd.read_csv(users)
#     d = {key: value for key, value in zip(u["userId"], u["gender"])}
#     gender = []
#     for item in df["userId"]:
#         gender.append(d[item]) if item in d else gender.append(0.5)
#     label = []
#     for item in df["label"]:
#         label.append(1) if item == "Liked" else label.append(-1)
#     res = {"userId": df["userId"],
#            "groupId": df["groupId"],
#            "label": label
#            }
#     for i in range(300):
#         res[f"lda_{i}"] = df[f"lda_{i}"]
#     for i in range(26):
#         res[f"svd_{i}"] = df[f"svd_{i}"]
#
#     res["gender"] = gender
#     new_data = pd.DataFrame(data=res)
#     new_data.to_csv("/home/ruslan/PycharmProjects/group_recommender/group_recommender/dataOKcsv/views_formated.csv",
#                     index=False)


# def format_data(path_views, path_groups):
#     df_views = pd.read_csv(path_views)
#     df_groups = pd.read_csv(path_groups)
#
#     df_groups = df_groups.fillna(0)
#     df_views = df_views.fillna(0)
#
#     df_views = df_views.get_values()
#     df_groups = df_groups.get_values()
#
#     f = open("dataForReg.txt", "a")
#
#     groups = {key: list(value) for key, value in zip(df_groups[:, 0], df_groups[:, 1:])}
#     print(list(groups[df_views[3][1]]))
#     for i in range(len(df_views[:, 0])):
#         print(i)
#         s = " ".join(map(str, list(df_views[i, 3:]))) + "|" + " ".join(map(str, groups[df_views[i][1]])) + "|" +\
#             str(df_views[i][2]) + "\n"
#         f.write(s)
#     del df_views
#     del df_groups
#     f.close()
#     return "OK"


def pca_data():
    f = open("dataForReg.txt", "r")
    lines = f.readlines()
    f.close()
    X = []
    Z = []
    R = []
    while lines:
        line = lines.pop()
        x, z, r = line.strip().split("|")

        x, z = x.split(), z.split()
        s_user = len(x)
        s_group = len(z)

        r = float(r.strip(" "))
        x = np.array(list(map(float, x)))
        z = np.array(list(map(float, z)))
        X.append(x)
        Z.append(z)
        R.append(r)
        del line
    print("Data reading completed")
    pc = sd.PCA(n_components=0.9)

    n_x = pc.fit_transform(X[:200_000])
    other_x = pc.transform(X[200_000:])

    n_z = pc.transform(Z[: 200_000])
    other_z = pc.transform(Z[200_000:])

    f = open("dataForRegPCA.txt", "a")

    step = 0

    for i in range(len(n_x)):
        s = " ".join(map(str, n_x[i])) + "|" + " ".join(map(str, n_z[i])) + "|" + str(R[i]) + "\n"
        f.write(s)
        step += 1
        print(step)

    R = R[200_000:]

    for i in range(len(other_x)):
        s = " ".join(map(str, other_x[i])) + "|" + " ".join(map(str, other_z[i])) + "|" + str(R[i]) + "\n"
        f.write(s)
        step += 1
        print(step)
    f.close()
    print()
    print("X completed")


def format():
    f = open("dataForRegPCA.txt", "r")
    lines = f.readlines()
    f.close()

    test = lines[:200_000]
    lines = lines[200_000:]
    seq = []
    while lines:
        line = lines.pop()
        x, z, r = line.strip().split("|")

        x, z = x.split(), z.split()
        s_user = len(x)
        s_group = len(z)

        r = float(r.strip(" "))
        x = np.array([list(map(float, x))]).transpose()
        z = np.array([list(map(float, z))]).transpose()
        seq.append((x, z, r))
        del line
    print(len(seq))

    clf = LogReg(s_user, s_group, 5, 0.01)
    clf.fit(seq)

    test_seq = []
    while test:
        line = test.pop()
        x, z, r = line.strip().split("|")

        x, z = x.split(), z.split()
        s_user = len(x)
        s_group = len(z)

        r = float(r.strip(" "))
        x = np.array([list(map(float, x))]).transpose()
        z = np.array([list(map(float, z))]).transpose()
        test_seq.append((x, z, r))
        del line

    pred = [clf.predict(i, j) for i, j, _ in test_seq]
    counter = 0
    p = None
    for i in range(len(pred)):
        if pred[i] < 0.5:
            p = -1.0
        else:
            p = 1.0
        if p == test_seq[i][2]:
            counter += 1

    print(pred)
    print(counter / 200_000)
    return clf
