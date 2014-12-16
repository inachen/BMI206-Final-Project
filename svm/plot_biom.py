import sys
import matplotlib
# matplotlib.use('Agg')
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

from collections import Counter
style.use('ggplot')

colors = ['#a6cee3', '#1f78b4','#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', 
        '#fdbf6f', '#ff7f00', '#cab2d6']

def sumzip(*items):
    return [sum(values) for values in zip(*items)]

def subset_stacked_bar():

    subset_df = pd.read_csv("Rfr_12_singletons/health_12_run100_topotu.csv", header=None)

    breakdown = []

    for col in subset_df.columns:
        breakdown.append(Counter(subset_df[col]))

    # names = []

    # for i in range(5):
    #     names += breakdown[i].keys()

    names = [189827, 470663, 326792, 212619, 170124, 
        148099, 287510, 261912, 533785, 295024, 15141, 9514, 194745, 
        268604, 181834, 175682, 191687, 217996, 364234, 561483, 48207, 
        361809, 517331, 469852, 185951, 198251, 259261, 178122, 469873, 
        72820, 554755, 13823]

    dic = dict((el,[]) for el in names)

    for i in range(5):
        for k, v in dic.iteritems():
            if k in breakdown[i].keys():
                new_num = breakdown[i][k]
                dic[k].append(new_num)
            else:
                dic[k].append(0)

    # get top otu
    filtered_dic = {}
    for k, v in dic.iteritems():
        if sum(v) > 10:
            filtered_dic[k] = v

    print filtered_dic

    ind = [0.75, 1.75, 2.75, 3.75, 4.75]    
    width = 0.5

    # ncolors = len(plt.rcParams['axes.color_cycle'])
    colors2 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', 
        '#b3de69', '#fccde5', '#d9d9d9']

    bottoms = [0, 0, 0, 0, 0]

    fig = plt.figure(0)
    plt.clf()
    fil_keys = filtered_dic.keys()
    for i, n in enumerate(fil_keys):
        if i == 0: 
            plt.bar(ind, filtered_dic[n], width, color=colors[i], label=fil_keys[i])
            bottoms = filtered_dic[n]
        else:
            plt.bar(ind, filtered_dic[n], width, bottom=bottoms, color=colors[i], label=fil_keys[i])
            bottoms = [sum(x) for x in zip(bottoms, filtered_dic[n])]

    plt.xlabel("OTU Rank")
    plt.ylabel("Percentage (%)")
    plt.axis((0.5, 5.5, 0, 120))
    # plt.title("Score: " + "%.2f" % train_score)
    plt.legend()
    fig.savefig('test')

def imp_bar():
    imp_lst = [0.47146149094965006, 0.087902771797110701, 0.053653410325969667, 0.038736782303906536, 
        0.018261851332768468, 0.016036429067049326, 0.015351936605521652, 0.01511926816979231, 
        0.015046225636609281, 0.013970990324742959, 0.013625029198591371, 0.012883542853420258, 
        0.011632086059836061, 0.01081443171367762, 0.010245011592008557, 0.0096927841275030874, 
        0.0093615878823342536, 0.008609765724395306, 0.0073647503392515077, 0.0069674176625549996, 
        0.006706300404562369, 0.0061298120959311945, 0.0054979810951755236, 0.0052403676539383385, 
        0.004784580536665417, 0.0045741065137204703, 0.0044486669581449215, 0.004435589308247394, 
        0.0043890452895819507, 0.0043510092599888485, 0.0043188055255307866, 0.0041291561680709731, 
        0.0038457027633086532, 0.0037704715427749856, 0.0037449551213587745, 0.0035867889598340942, 
        0.0034228170171827169, 0.0033322867280184023, 0.0032764576776128074, 0.0032231647986478177, 
        0.0030823881846464432, 0.0030161676642879516, 0.0029621532801089112, 0.0029248392657187741, 
        0.0028859887918053248, 0.002806543710012677, 0.0027864033075781728, 0.0027391443136591787, 
        0.0026331913612681699, 0.0026216032806205339, 0.0025870280169503612, 0.0025546542598314913, 
        0.0024905773224071499, 0.0020992811410992696, 0.0019554796590165583, 0.0019144350185704073, 
        0.0018728040413941067, 0.0015896619371727012, 0.0015660437384334635, 0.0015120161344982679, 
        0.0014861035354251635, 0.0013868123422364315, 0.0013600095442208987, 0.0012669782590689265, 
        0.0011491281396147077, 0.0011105547369268183, 0.00093869210800632673, 0.00090471239530591702, 
        0.00084213503512184363, 0.0008119495629745658, 0.00078906630666126837, 0.0007871575105385507, 
        0.00077215726084705607, 0.00072713060649522274, 0.00070863217377465954, 0.00064512472744754151, 
        0.0005719806784634583, 0.00054215164701436498, 0.00041940782742475318, 0.00014601619923959306, 
        8.8065889121102377e-05]

    width = 1
    x = [i + 0.5 for i in range(len(imp_lst))]
    fig = plt.figure(0)
    plt.clf()
    plt.bar(x, imp_lst, width)
    plt.xlabel("OTU Rank")
    plt.ylabel("Importance")
    fig.savefig("importancebar")

score_lst = [29.45, 13.11, 11.89, 11.45, 11.46, 11.47]
tree_lst = [1, 10, 100, 1000, 5000, 10000]
fig = plt.figure(0)
plt.clf()
plt.plot(tree_lst, score_lst, 'x-')
plt.xlabel("Number of Trees")
plt.ylabel("Score on Twins")
plt.axis((-1000, 10500, 0, 30))
fig.savefig("treenum")



