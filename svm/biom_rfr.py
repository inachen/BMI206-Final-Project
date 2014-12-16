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

style.use('ggplot')

# =============================================
# read in data
dat_df = pd.read_csv("data/all_mat.csv")

headers = list(dat_df)
otu_heads = headers[1:82]

# get integer ages
dat_df["age_10x"] = dat_df["age_in_months"] * 10

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

    return(f, new_x, new_y)

def pred_score(pred, actual):
    if len(pred) != len(actual):
        raise Exception("Need equal lengths")

    score = 0
    for p, a in zip(pred, actual):
        score += (p - a) ** 2

    score = score/float(len(actual))

    return score

def print_lst(lst):
    for l in lst:
        print l

# =============================================
# random forest

def rfr(train_otu, train_age, t_train, test1_otu, test1_age, t_test1, 
    test2_otu=None, test2_age=None, t_test2=None,
    test3_otu=None, test3_age=None, t_test3=None,
    test4_otu=None, test4_age=None, t_test4=None):

    # print "Running Random Forest"

    rf = ensemble.RandomForestRegressor(n_estimators = 1000)

    rf.fit(train_otu, train_age)

    rf_importance = rf.feature_importances_

    rev_ranked_imp, rev_ranked_otu = zip(*sorted(zip(rf_importance, otu_heads)))

    # rev_ranked_imp = list(rev_ranked_imp)
    # rev_ranked_imp.reverse()

    rev_ranked_otu = list(rev_ranked_otu)
    rev_ranked_otu.reverse()

    # print_lst(rev_ranked_otu)

    train_pred = rf.predict(train_otu)

    train_f, sp_x, sp_y = spline_fit(train_age,train_pred)

    train_score = pred_score(train_pred, train_age)

    # print sp_x
    # print sp_y

    fig = plt.figure(0)
    plt.clf()
    plt.plot(train_age, train_pred, '.', color='black')
    plt.plot(sp_x, sp_y)
    plt.xlabel("Biological Age")
    plt.ylabel("Microbiota Age")
    plt.title("Score: " + "%.2f" % train_score)
    fig.savefig(t_train)

    test1_pred = rf.predict(test1_otu)

    t1_f, t1_sp_x, t1_sp_y = spline_fit(test1_age, test1_pred)

    test1_score = pred_score(test1_pred, test1_age)

    fig = plt.figure(0)
    plt.clf()
    plt.plot(test1_age, test1_pred, '.', color='black')
    plt.plot(t1_sp_x, t1_sp_y)
    plt.xlabel("Biological Age")
    plt.ylabel("Microbiota Age")
    plt.title("Score: " + "%.2f" % test1_score)
    fig.savefig(t_test1)

    if test2_otu != None:
        test2_pred = rf.predict(test2_otu)
        test3_pred = rf.predict(test3_otu)

        test2_score = pred_score(test2_pred, test2_age)
        test3_score = pred_score(test3_pred, test3_age)

        # t2_f, t2_sp_x, t2_sp_y = spline_fit(test2_age, test2_pred)
        # print plt.rcParams['axes.color_cycle']
        fig = plt.figure(0)
        plt.clf()
        plt.plot(test2_age, test2_pred, '.', color='black', label="RUTF")
        plt.plot(test3_age, test3_pred, '.', color=plt.rcParams['axes.color_cycle'][1], label="Khichuri")
        plt.plot(t1_sp_x, t1_sp_y, label="spline")
        plt.xlabel("Biological Age")
        plt.ylabel("Microbiota Age")
        plt.title("RUTF Score: " + "%.2f" % test2_score + ", Khichuri Score: " + "%.2f" % test3_score)
        plt.legend(loc="upper left")
        fig.savefig(t_test2)

    if test4_otu != None:
        test4_pred = rf.predict(test4_otu)

        test4_score = pred_score(test4_pred, test4_age)

        # t2_f, t2_sp_x, t2_sp_y = spline_fit(test2_age, test2_pred)
        # print plt.rcParams['axes.color_cycle']
        fig = plt.figure(0)
        plt.clf()
        plt.plot(test4_age, test4_pred, '.', color='black', label="RUTF")
        plt.plot(t1_sp_x, t1_sp_y, label="spline")
        plt.xlabel("Biological Age")
        plt.ylabel("Microbiota Age")
        plt.title("Score: " + "%.2f" % test4_score)
        plt.legend(loc="upper left")
        fig.savefig(t_test4)

    return rev_ranked_otu

# get healthy singletons
hsingle_df = dat_df[dat_df['Health_Analysis_Groups'] == 'Healthy Singletons']

hsingle_train = hsingle_df[otu_heads].values.astype(int) #(hsingle_df[otu_heads]).values.astype(int).tolist()
hsingle_age = hsingle_df["age_in_months"].values.astype(float)

# print hsingle_age.flags

# get healthy twins
htwin_df = dat_df[dat_df['Health_Analysis_Groups'] == 'Healthy Twins Triplets']

htwin_train = htwin_df[otu_heads].values.astype(int)
htwin_age = htwin_df["age_in_months"].values.astype(float) #["age_10x"].values.astype(int)

# get treatment group
treatment_df = dat_df[dat_df['Health_Analysis_Groups'] == "Severe Acute Malnutrition Study"]

treatment_otu = treatment_df[otu_heads].values.astype(int)
treatment_age = treatment_df["age_in_months"].values.astype(float)

# RUTF treatment
rutf_df = treatment_df[treatment_df["treatment_type"] == "RUTF"]

rutf_otu = rutf_df[otu_heads].values.astype(int)
rutf_age = rutf_df["age_in_months"].values.astype(float)

# khich treatment
khich_df = treatment_df[treatment_df["treatment_type"] == "khich"]

khich_otu = khich_df[otu_heads].values.astype(int)
khich_age = khich_df["age_in_months"].values.astype(float)

# need to get the treatment curves for each sample
# set(rutf_df["PersonID"])
rutf_ids = ['Bgmal29', 'Bgmal61', 'Bgmal48', 'Bgmal49', 'Bgmal20', 'Bgmal22', 'Bgmal42', 
    'Bgmal43', 'Bgmal26', 'Bgmal27', 'Bgmal62', 'Bgmal15', 'Bgmal14', 'Bgmal13', 'Bgmal18', 'Bgmal59', 
    'Bgmal52', 'Bgmal54', 'Bgmal57', 'Bgmal33', 'Bgmal58', 'Bgmal35', 'Bgmal25', 'Bgmal1', 'Bgmal5', 'Bgmal7', 
    'Bgmal6', 'Bgmal9', 'Bgmal41', 'Bgmal44', 'Bgmal34']

khich_ids = ['Bgmal28', 'Bgmal64', 'Bgmal60', 'Bgmal53', 'Bgmal46', 'Bgmal47', 'Bgmal45', 
    'Bgmal24', 'Bgmal12', 'Bgmal40', 'Bgmal63', 'Bgmal16', 'Bgmal11', 'Bgmal10', 'Bgmal21', 
    'Bgmal19', 'Bgmal51', 'Bgmal50', 'Bgmal39', 'Bgmal23', 'Bgmal55', 'Bgmal56', 'Bgmal32', 
    'Bgmal31', 'Bgmal30', 'Bgmal37', 'Bgmal36', 'Bgmal3', 'Bgmal2', 'Bgmal4', 'Bgmal8']

rutf_all_df = rutf_df[rutf_df['PersonID'].isin(rutf_ids)]
rutf_all_df.set_index(['id'])
rutf_all_df['treat_time'] = [-2 for i in range(len(rutf_all_df['id']))]

for r_id in rutf_ids:
    rutf_indiv = rutf_df[rutf_df['PersonID'] == r_id]
    # age = rutf_indiv['age_in_months']
    treat_age = rutf_indiv[rutf_indiv['treatment_bool'] == 1]['age_in_months']

    for index, row in rutf_all_df.iterrows():

        if row['PersonID'] == r_id:
            age = row['age_in_months']
            if age < treat_age.min():
                rutf_all_df.ix[index, 'treat_time']= -1
            elif age > treat_age.max():
                age_diff = age - treat_age.max()
                rutf_all_df.ix[index, 'treat_time'] = age_diff
            else:
                rutf_all_df.ix[index, 'treat_time'] = 0

khich_all_df = khich_df[khich_df['PersonID'].isin(khich_ids)]
khich_all_df.set_index(['id'])
khich_all_df['treat_time'] = [-2 for i in range(len(khich_all_df['id']))]

for r_id in khich_ids:
    khich_indiv = khich_df[khich_df['PersonID'] == r_id]
    # age = khich_indiv['age_in_months']
    treat_age = khich_indiv[khich_indiv['treatment_bool'] == 1]['age_in_months']

    for index, row in khich_all_df.iterrows():

        if row['PersonID'] == r_id:
            age = row['age_in_months']
            if age < treat_age.min():
                khich_all_df.ix[index, 'treat_time']= -1
            elif age > treat_age.max():
                age_diff = age - treat_age.max()
                khich_all_df.ix[index, 'treat_time'] = age_diff
            else:
                khich_all_df.ix[index, 'treat_time'] = 0

# RUTF sub treatment
# -1 for pre treatment, 0 for during, .isin([1,2,3]), .isin([3,4,5,6,7,8,9,10]) (highest month is 9)
rutf_sub_df = rutf_all_df[rutf_all_df["treat_time"].isin([3,4,5,6,7,8,9,10])]

rutf_otu_sub = rutf_sub_df[otu_heads].values.astype(int)
rutf_age_sub = rutf_sub_df["age_in_months"].values.astype(float)

# khich sub treatment
khich_sub_df = khich_all_df[khich_all_df["treat_time"].isin([3,4,5,6,7,8,9,10])]

khich_otu_sub = khich_sub_df[otu_heads].values.astype(int)
khich_age_sub = khich_sub_df["age_in_months"].values.astype(float)

# get random 12 singles
single_id_lst = ['Bgsng7018', 'Bgsng7035', 'Bgsng7052', 'Bgsng7063', 'Bgsng7071', 'Bgsng7082',
    'Bgsng7090', 'Bgsng7096', 'Bgsng7106', 'Bgsng7114', 'Bgsng7115', 'Bgsng7128', 'Bgsng7131',
    'Bgsng7142', 'Bgsng7149', 'Bgsng7150', 'Bgsng7155', 'Bgsng7173', 'Bgsng7177', 'Bgsng7178',
    'Bgsng7192', 'Bgsng7202', 'Bgsng7204', 'Bgsng8064', 'Bgsng8169']

single_sub_train_lst = random.sample(single_id_lst, 12)
single_sub_test_lst = list(set(single_id_lst)-set(single_sub_train_lst))

hsingle_sub_train = hsingle_df[hsingle_df['PersonID'].isin(single_sub_train_lst)]
hsingle_sub_train_otu = hsingle_sub_train[otu_heads].values.astype(int)
hsingle_sub_train_age = hsingle_sub_train["age_in_months"].values.astype(float)

hsingle_sub_test = hsingle_df[hsingle_df['PersonID'].isin(single_sub_test_lst)]
hsingle_sub_test_otu = hsingle_sub_test[otu_heads].values.astype(int)
hsingle_sub_test_age = hsingle_sub_test["age_in_months"].values.astype(float)


# =============================================
# All singletons used for training

# train_otu = hsingle_train
# train_age = hsingle_age
# t_train = "Rfr_n1000_maxNone_spline_healthy_single"

# test1_otu = htwin_train
# test1_age = htwin_age
# t_test1 = "Rfr_n1000_maxNone_spline_healthy_twins"

# rfr(train_otu, train_age, t_train, test1_otu, test1_age, t_test1)

# =============================================
# 12 singletons used for training

single_sub_train_lst = random.sample(single_id_lst, 12)
# subset being used:
single_sub_train_lst = ['Bgsng7131', 'Bgsng8064', 'Bgsng7018', 'Bgsng7202', 'Bgsng7155', 
    'Bgsng7115', 'Bgsng7192', 'Bgsng7071', 'Bgsng7082', 'Bgsng7150', 'Bgsng7178', 'Bgsng7128']
single_sub_test_lst = list(set(single_id_lst)-set(single_sub_train_lst))

hsingle_sub_train = hsingle_df[hsingle_df['PersonID'].isin(single_sub_train_lst)]
hsingle_sub_train_otu = hsingle_sub_train[otu_heads].values.astype(int)
hsingle_sub_train_age = hsingle_sub_train["age_in_months"].values.astype(float)

hsingle_sub_test = hsingle_df[hsingle_df['PersonID'].isin(single_sub_test_lst)]
hsingle_sub_test_otu = hsingle_sub_test[otu_heads].values.astype(int)
hsingle_sub_test_age = hsingle_sub_test["age_in_months"].values.astype(float)

# print "12 Singletons:"
print single_sub_train_lst

train_otu = hsingle_sub_train_otu
train_age = hsingle_sub_train_age
t_train = "Rfr_n1000_maxNone_spline_healthy_12_single"

test1_otu = htwin_train
test1_age = htwin_age
t_test1 = "Rfr_n1000_maxNone_spline_healthy_twins"

# test with rest of singletons
# test2_otu = hsingle_sub_test_otu
# test2_age = hsingle_sub_test_age
# t_test2= "Rfr_n1000_maxNone_spline_healthy_test_single"

# plot the treatment group
test2_otu = rutf_otu_sub
test2_age = rutf_age_sub
t_test2 = "Rfr_n1000_maxNone_spline_3post_treatment"

test3_otu = khich_otu_sub
test3_age = khich_age_sub
t_test3 = "Rfr_n1000_maxNone_spline_treatment"

# test4_otu = treatment_otu
# test4_age = treatment_age
# t_test4 = "Rfr_n1000_maxNone_spline_all_treatment"


rfr(train_otu, train_age, t_train, test1_otu, test1_age, t_test1, 
    # test4_otu=test4_otu,test4_age=test4_age, t_test4=t_test4)
    test2_otu, test2_age, t_test2, test3_otu, test3_age, t_test3)


