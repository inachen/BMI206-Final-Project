import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
import csv
import numpy as np
import math
import matplotlib.cm as cm
from mpltools import style
from mpltools import color
import copy
import os.path
import random
import pandas as pd

from glob import glob
from Bio import SeqIO
import cPickle as pickle

from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scipy import interpolate

# =================================================
# gave up on numpy
# dat_mat = np.genfromtxt(open("data/all_mat.csv","rb"), delimiter=",", dtype=None, names=True)
# headers = dat_mat.dtype.names

# otu_heads = headers[1:82]
# =================================================

# =============================================
# prediction scoring

style.use('ggplot')

def pred_score(pred, actual):
    if len(pred) != len(actual):
        raise Exception("Need equal lengths")

    score = 0
    for p, a in zip(pred, actual):
        score += (p - a) ** 2

    score = score/float(len(actual))

    return score

def spline_fit(x, y):

    sorted_x, sorted_y = zip(*sorted(zip(x, y)))
    new_length = 100
    new_x = np.linspace(x.min(), x.max(), new_length)
    # print x.max()

    # tck = interpolate.splrep(sorted_x, sorted_y, s=0)
    # new_y = interpolate.splev(new_x, tck, der=0)

    # f = interpolate.interp1d(sorted_x, sorted_y, kind='cubic', bounds_error=False)
    f = interpolate.UnivariateSpline(sorted_x, sorted_y, k=3, s=5e7)
    new_y = f(new_x)

    # print new_y

    return(new_x, new_y)


# read in data
dat_df = pd.read_csv("data/all_mat.csv")

headers = list(dat_df)
otu_heads = headers[1:82]

# get integer ages
dat_df["age_10x"] = dat_df["age_in_months"] * 10

# get healthy singletons
hsingle_df = dat_df[dat_df['Health_Analysis_Groups'] == 'Healthy Singletons']

hsingle_train = hsingle_df[otu_heads].values.astype(int) #(hsingle_df[otu_heads]).values.astype(int).tolist()
hsingle_age = hsingle_df["age_in_months"].values.astype(float)

# print hsingle_age.flags

# get healthy twins
htwin_df = dat_df[dat_df['Health_Analysis_Groups'] == 'Healthy Twins Triplets']

htwin_train = htwin_df[otu_heads].values.astype(int)
htwin_age = htwin_df["age_in_months"].values.astype(float)


# =============================================
# SVM/SVR

# # train svm
# svm_model = svm.SVR(verbose=False)
# svm_model.fit(hsingle_train, hsingle_age) # (data, labels)

# # 178 unique ages, 15753 classifiers = 178*177/2

# hsingle_pred = svm_model.predict(hsingle_train)

# print hsingle_pred
# print hsingle_age

# fig = plt.figure(0)
# plt.clf()
# plt.plot(hsingle_age, hsingle_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("SVC_healthy_single")

# htwin_pred = svm_model.predict(htwin_train)

# print htwin_pred
# print htwin_age

# fig = plt.figure(0)
# plt.clf()
# plt.plot(htwin_age, htwin_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("SVC_healthy_twins")

# =============================================

# Lasso

# las = linear_model.Lasso(alpha = 1, max_iter=10000)
# las.fit(hsingle_train, hsingle_age)

# hsingle_pred = las.predict(hsingle_train)

# print las.score(hsingle_train, hsingle_age)

# fig = plt.figure(0)
# plt.clf()
# plt.plot(hsingle_age, hsingle_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("Lasso_iter_10000_healthy_single")

# htwin_pred = las.predict(htwin_train)

# print las.score(htwin_train, htwin_age)

# fig = plt.figure(0)
# plt.clf()
# plt.plot(htwin_age, htwin_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("Lasso_iter_10000_healthy_twins")

# =============================================
# Elastic Net

# enet = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
# enet.fit(hsingle_train, hsingle_age)

# hsingle_pred = enet.predict(hsingle_train)

# print clf.score(hsingle_train, hsingle_age)

# fig = plt.figure(0)
# plt.clf()
# plt.plot(hsingle_age, hsingle_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("Enet_healthy_single")

# htwin_pred = enet.predict(htwin_train)

# print clf.score(htwin_train, htwin_age)

# fig = plt.figure(0)
# plt.clf()
# plt.plot(htwin_age, htwin_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("Enet_healthy_twins")

# =============================================
# Adaboost

# ada = ensemble.AdaBoostClassifier(n_estimators=1000)

# ada.fit(hsingle_train, hsingle_age)

# hsingle_pred = ada.predict(hsingle_train)

# fig = plt.figure(0)
# plt.clf()
# plt.plot(hsingle_age, hsingle_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("Ada_healthy_single")

# htwin_pred = ada.predict(htwin_train)


# fig = plt.figure(0)
# plt.clf()
# plt.plot(htwin_age, htwin_pred, '.')
# plt.xlabel("Biological Age")
# plt.ylabel("Microbiota Age")
# fig.savefig("Ada_healthy_twins")

# =============================================
# Random Forest

for n in [5000, 10000]:

    rf = ensemble.RandomForestRegressor(n_estimators = n)

    rf.fit(hsingle_train, hsingle_age)

    hsingle_pred = rf.predict(hsingle_train)

    sp_x, sp_y = spline_fit(hsingle_age,hsingle_pred)

    train_score = pred_score(hsingle_pred, hsingle_age)

    # print sp_x
    # print sp_y

    fig = plt.figure(0)
    plt.clf()
    plt.plot(hsingle_age, hsingle_pred, '.', color='black')
    plt.plot(sp_x, sp_y)
    plt.xlabel("Biological Age")
    plt.ylabel("Microbiota Age")
    plt.title("Score: " + "%.2f" % train_score)
    fig.savefig("Rfr_n%d_spline_all_healthy_single" % n)

    htwin_pred = rf.predict(htwin_train)

    twin_sp_x, twin_sp_y = spline_fit(htwin_age, htwin_pred)

    twin_score = pred_score(htwin_pred, htwin_age)

    fig = plt.figure(0)
    plt.clf()
    plt.plot(htwin_age, htwin_pred, '.', color='black')
    plt.plot(twin_sp_x, twin_sp_y)
    plt.xlabel("Biological Age")
    plt.ylabel("Microbiota Age")
    plt.title("Score: " + "%.2f" % twin_score)
    fig.savefig("Rfr_n%d_spline_healthy_twins"%n)

# =============================================
# setting alpha value for Lasso

# alpha_lst = [0.1*i for i in range(0, 11)]

# single_scores = []
# twin_scores = []

# for alpha in alpha_lst:
#     las = linear_model.Lasso(alpha = alpha, max_iter=10000)
#     las.fit(hsingle_train, hsingle_age)

#     single_scores.append(las.score(hsingle_train, hsingle_age))
#     twin_scores.append(las.score(htwin_train, htwin_age))

# fig = plt.figure(0)
# plt.clf()
# plt.plot(alpha_lst, single_scores, '.')
# plt.xlabel("Alpha Value")
# plt.ylabel("Score")
# # plt.axis([0, 1, -0.106, -0.102])
# fig.savefig("Lasso_alphas_single")

# fig = plt.figure(0)
# plt.clf()
# plt.plot(alpha_lst, twin_scores, '.')
# plt.xlabel("Alpha Value")
# plt.ylabel("Score")
# # plt.axis([0, 1, -0.106, -0.102])
# fig.savefig("Lasso_alphas_twin")



# =============================================

# k-fold validation

kf = KFold(537, n_folds=10)
# print kf


# c_lst = [0.1*i for i in range(1,11)]

# score_lsts = []

# for c in c_lst:
    
#     score_lst = []

#     for train, test in kf:

#         clf = linear_model.Lasso(alpha = 1, max_iter=10000)
#         clf.fit(hsingle_train[train], hsingle_age[train])
#         s = clf.score(hsingle_train[test], hsingle_age[test])

#         # svm_model = svm.SVR(verbose=False)
#         # svm_model.fit(hsingle_train[train], hsingle_age[train])
#         # s = svm_model.score(hsingle_train[test], hsingle_age[test])
#         score_lst.append(s)

#     score_lsts.append(np.mean(score_lst))

# print score_lsts

# fig = plt.figure(0)
# plt.clf()
# plt.plot(c_lst, score_lsts, '.')
# plt.xlabel("C Value")
# plt.ylabel("Score")
# # plt.axis([0, 1, -0.106, -0.102])
# fig.savefig("10-fold_CV_Lasso")

# =============================================

# rand_train = []

# for i in range(500):
#     rand_train.append([random.randint(0,91)*i for x in range(81)])

# svm_test = svm.SVR()
# svm_test.fit(rand_train, range(500))

# p_lst = [random.randint(0,91)*499 for x in range(81)]
# p_lst = [10 for x in range(81)]
# print p_lst

# test_pred = svm_test.predict(p_lst)
# print test_pred

# rand_nums = [random.randint(0,999) for x in range(81)]

# s_pred = svm_model.predict(rand_nums)

# print rand_nums

# print s_pred

# =============================================
# using multiclass svm

# hsingle_labels = LabelBinarizer().fit_transform(hsingle_age)

# clf = OneVsOneClassifier(LinearSVC())

# # classifier = Pipeline([
# #     ('vectorizer', CountVectorizer()),
# #     ('tfidf', TfidfTransformer()),
# #     ('clf', OneVsRestClassifier(LinearSVC()))])

# clf.fit(hsingle_train, hsingle_labels)
# single_pred = clf.predict(hsingle_train)
# # single_pred_lb = hsingle_labels.inverse_transform(single_pred)

# print hsingle_labels


# =============================================
# MultiLabelBinarizer().fit_transform(Y)

# # svc = svm.SVC(kernel='linear') # linear kernel
# # svc = svm.SVC(kernel='rbf') # radial kernel

# OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y)
# svc.fit(x, y)
# predict(x)

# score(X, y, sample_weight=None)