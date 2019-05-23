"""Example for preprocessing data for contextual MAB"""
from dataProcessing.log_reg import LogReg
import numpy as np
import sklearn.decomposition as sd
import sklearn.mixture as sm
import pickle


def pca(sample):
    # sample -- выборка, состоящая из исходных контекстов пользователей/объектов в виде np.ndarray
    pc = sd.PCA(n_components=0.9)
    pc.fit(sample)
    return pc


def bilinear_transformation(pc, sample_of_users, sample_of_objects, path):
    # path -- путь к файлу, в котором лежит обученный метод класс LogReg с расширением .pickle
    with open(path, "rb") as f:
        b_trans = pickle.load(f)

    new_users = pc.transform(sample_of_users)
    new_objects = pc.transform(sample_of_objects)

    transformed_users = [np.dot(item.T, b_trans.W).flatten() for item in new_users]
    transformed_groups = [np.dot(b_trans.W, item).flatten() for gr_id, item in new_objects]

    return transformed_users, transformed_users


def clustering(users, objects):
    # users и objects -- контексты пользователей и объектов уже полсе PCA и билинейного преобразования
    clust_users = sm.GaussianMixture(n_components=8)
    clust_objects = sm.GaussianMixture(n_components=8)

    clust_users.fit(users)
    clust_objects.fit(objects)

    return clust_users.predict_proba(users), clust_objects.predict_proba(objects)
