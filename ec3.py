#==============================================================================
# #                           load the  basic libraries                          #
#==============================================================================
import numpy as np
#import scipy
import pandas as pd
from copy import copy
import time
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
#==============================================================================
# #                       Load ML libraries                                       #
#==============================================================================

#load the libraries
# from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from numpy import linalg as LA
from math import sqrt
from copy import copy, deepcopy
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#==============================================================================
#                            code starts                             #
#==============================================================================
# user defined function to change categories to number
def cattoval(coln):
    for i in range(len(coln)):
        for j in range(L):
            if(coln[i] == cat[j]):
                coln[i] = j
    return coln
def cattoval2(coln):
    coln = coln.reset_index(drop=True)
    for i in range(len(coln)):
        for j in range(L):
            if(coln[i] == cat[j]):
                coln[i] = j
    return coln

def normalise(A,n=1):
    if n == 1 :
        A = A/A.sum(axis=1)[:,None]
    elif n == 0:
        A = A/A.sum(axis=0)[None,:]
    return A

#-------------------------------------------------------------------------------------#
#-------------K from A -------------#


def StochasticK(Ac,ep = 0.025):
    K = np.zeros((len(X_test), len(X_test)))
    KM =copy(Ac)
    N = len(X_test)
#    KM = np.zero((int(Ac.shape[0]), int(Ac.shape[1])))

    while LA.norm(np.array(K)-np.array(KM)) / (N*N) > ep:
        K=deepcopy(KM)
#        d=KM.sum(axis=1)
#         for i in range(N):
#             for j in  range(N):
#                 K[i][j] = KM[i][j] / d[i]
        K = normalise(KM,1)
        for i in range(N):
            for j in  range(N): #52
                K[i][j] = K[j][i]= sqrt(K[i][j]*K[j][i])
    return K
#-----------------------------------------------------------------------------------#

#==============================================================================
# #-----------------------------------     MEMBERSHIP MATRIX ------------------------------------#
#==============================================================================
def getMemMat(N,G,C):
    MemMat = np.zeros((N, G))
    index = 0
    for k in range(C):
        for i in range(N):
            j = index +  Algo[i,k]
            MemMat[i][j] = 1
        index = index + grp[k]
    return MemMat

#==============================================================================
# #-----------              Co-occurence Matrix                  -----------#
#==============================================================================
def Count(m,n):
    score = 0
    for k in range(C):
        if ( Algo[m,k] == Algo[n,k]):
            score = score + 1
    return score

#--------------------  average object class matrix -------------------------------#
#--------------------only supervised algo is considered-------------------------#
def fun(m,n):
    score = 0
    for k in range(C1):
        if Algo[m,k] == n:
            score = score + 1
    return score

def getObjclass(N,L):
    Objclass = np.zeros((N, L))
    for i in range(N):
        for j in  range(L):
    #        temp = int (Salgo[i,j])
            value = fun(i,j)
#             print('value ',value)
            value2 = value/float(C1)
    #        Objclass[i][temp]= Objclass[i][temp] + 1/4
            Objclass[i,j] = value2
    return Objclass

#----------------- average group class matrix --------------------------------#
def getGrpclass(G,L):
#    global MemDF
    Grpclass = np.zeros(shape=(G,L))
    for gno in range(G):
        idx = MemDF[MemDF[gno] == 1 ].index.tolist()
        tot = float(len(idx) * C1)
#         print('tot',tot)
        for i in idx:
            for j in range(C1):
                score = Salgo[i,j]
#                 print('score',score)
                Grpclass[gno,score]= (Grpclass[gno,score] + 1/tot )   # for average
    return Grpclass

def getdiagonal (A, x):
    A = A.sum(axis = x)
    return np.diag(A)
# the data should be in format pandas dataframe

def EC3(Fo, Fg, Km, Kc, Yo, Yg, alpha=0.25, beta=0.35, gamma=0.35, delta=0.05, eps=0.00001):
    global t

    Fot = copy(Objclass)

    while LA.norm(np.array(Fot) - np.array(Fo)) / (N * L) > eps:  # Fo = Fo(t-1)
        t = t + 1
        # print("loop run")
        lhs = np.linalg.inv(2 * delta * one + alpha * Dm)  # GxG
        rhs = alpha * np.matmul(Km.transpose(), Fo) + 2 * delta * Yg  # GxL
        Fg = np.matmul(lhs, rhs)  # GxG x GxL = G x L

        a = alpha * Dmdash
        b = 2 * beta * Dc
        c = beta * np.matmul(ideN, Kc)
        d = beta * np.matmul(oneN, Kc)
        e = 2 * gamma * oneN

        s = a + b

        s = s - c
        s = s - d
        s = s + e
        lhs = np.linalg.inv(s)
        f = alpha * np.matmul(Km, Fg)
        g = 2 * gamma * Yo
        rhs = f + g

        Fo = copy(Fot)
        Fot = np.matmul(lhs, rhs)

    #         print (Fot)
    return Fot


from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler


param_lgb = {'objective': 'binary',
                     'metric' :'binary_logloss,auc',
                     'max_depth' : 5,
                     'num_leaves' : 20,
                     'learning_rate' : 0.05,
                     'feature_fraction' : 0.5,
                     'min_child_samples':20,
                     'min_child_weight':0.001,
                     'bagging_fraction' : 1,
                     'bagging_freq' : 2,
                     'reg_alpha' : 0.001,
                     'reg_lambda' : 8,
                     'cat_smooth' : 0,
                     'num_iterations' : 200,
          }



aus=RandomUnderSampler(random_state=42)

train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
columns_accept=list(train.columns[0:-1])

Reject_train_ori=pd.read_csv('./data/Reject_data.csv')
Reject_data=Reject_train_ori[columns_accept]

X_train_valid=train[columns_accept]
y_train_valid=train['Y']

X_test=test[columns_accept]
y_test=test['Y']

scaler = StandardScaler()
X_train_valid_scaler=scaler.fit_transform(X_train_valid)
X_test_scaler=scaler.transform(X_test)
Reject_data_scaler=scaler.transform(Reject_data)

X=np.concatenate((X_train_valid_scaler,Reject_data_scaler),axis=0)



X_train_balance_logi,y_train_balance_logi=aus.fit_resample(X_train_valid_scaler,y_train_valid)
# X_train_balance_logi,y_train_balance_logi=X_train_valid_scaler,y_train_valid

X_train_balance,y_train_balance=aus.fit_resample(X_train_valid,y_train_valid)


##############################EC3开始
O = len(Reject_data_scaler)
L = len(np.unique(y_test))
print('L',L)

cat = list(np.unique(y_test))

Salgo = np.zeros(shape=(O, 0), dtype = np.int64)
grp = []

# model1 = LogisticRegression(C=1)
model2 = RandomForestClassifier(n_estimators=300,min_samples_split=30,
                                  min_samples_leaf=20,max_depth=8,criterion='gini'
                                                     )

model3 =xgb.XGBClassifier(
learning_rate =0.01, #
n_estimators=250, #
max_depth=8, #
min_child_weight=3, #
gamma=0.4, #
subsample=0.8, #
colsample_bytree=0.8,#
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1, #
seed=27,
importance_type='gini',
reg_alpha=0)

model4 = lgb.LGBMClassifier(**param_lgb)
model5 = SVC(C=0.4,probability=True)



# model1.fit(X_train_balance_logi, y_train_balance_logi)
# #     print('===========')
# y_train_pred = model1.predict(Reject_data_scaler)
# pred_train_prob = model1.predict_proba(Reject_data_scaler)[:,1]
# grp.append(L)
# y_train_pred = y_train_pred.astype(int)
# Salgo = np.c_[Salgo, y_train_pred]


model2.fit(X_train_balance_logi, y_train_balance_logi)
y_train_pred = model2.predict(Reject_data_scaler)
pred_train_prob = model2.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]


model3.fit(X_train_balance_logi, y_train_balance_logi)
y_train_pred = model3.predict(Reject_data_scaler)
pred_train_prob = model3.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]

model4.fit(X_train_balance_logi, y_train_balance_logi)
y_train_pred = model4.predict(Reject_data_scaler)
pred_train_prob = model4.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]
#
model5.fit(X_train_balance_logi, y_train_balance_logi)
y_train_pred = model5.predict(Reject_data_scaler)
pred_train_prob = model5.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]


Ualgo = np.zeros(shape=(O, 0), dtype=np.int64)
umodel= KMeans(n_clusters=L, random_state=0)
umodel.fit(X)
labels = umodel.labels_
predicted = umodel.predict(Reject_data_scaler)
n_clusters_ = len(np.unique(predicted))
predicted = [((n_clusters_) - 1) if x == -1 else x for x in predicted]
Ualgo = np.c_[Ualgo, predicted]
grp.append(n_clusters_)



# # --------------------    Parameter initialisation  ---------------------------------#
#==============================================================================

N = len(Reject_data_scaler)                        #  defined
# L                            number of classes
C1= 4                    #no. of classifier
C2 = 1                  # no. of clusters
C = C1 + C2
G1 = sum(grp[:C1])
G2 = sum(grp[C1:len(grp)])
G = G1+G2

Algo = np.column_stack((Salgo, Ualgo))
print('algo',Algo)
s = time.time()
print ("starting building matrices")
MemMat = getMemMat(N,G,C)                      #  NxG
MemDF = pd.DataFrame(MemMat)
MemMat =normalise(MemMat,1)

CoMat = np.zeros((N, N))
for i in range(N):
    for j in range(i,N):
        value = Count(i,j)
        CoMat[i][j]=CoMat[j][i] = value
# print(CoMat)
CoMat = normalise(CoMat)
CoMat = normalise(CoMat,0)
# CoMat = StochasticK(CoMat)

Objclass = getObjclass(N,L)
Grpclass = getGrpclass(G,L)

# ------------------- object - class matrix ------------------------------------#
"""
# condition satisfied   Fo >= 0 , |Fo i. | = 1  for every i in 1:n
Fo = np.zeros(shape=(len(X_test),13))

for i in range (len(X_test)):
    Fo[i][0] = 1

"""
Fo = np.random.rand(N, L)
Fo = Fo / Fo.sum(axis=1)[:, None]

# ------------------- Group - class matrix --------------------------------------#
"""# condition satisfied   Fg >= 0 , |Fg .j | = 1  for every j in 1:l

"""
Fg = np.random.rand(G, L)

Fg = Fg / Fg.sum(axis=0)[None, :]

e = time.time()
print ("all matrices have been made ")
print (e-s)
Dm = getdiagonal(MemMat , 0)
one = np.ones((G,G))
Dmdash = getdiagonal(MemMat,1)
Dc     = getdiagonal(CoMat,0)
oneN = np.ones((N,N))
ideN = np.identity(N)
t = 1

startAlgo = time.time()

alpha=0.05
beta=0.35
gamma=0.35
delta=1-alpha-beta-gamma

MainMat = EC3(Fo, Fg, MemMat, CoMat, Objclass, Grpclass, alpha,beta,gamma,delta,0.0001)

Endalgo = time.time()
print('Endalgo - startAlgo', Endalgo - startAlgo)
reject_pred1 = np.argmax(MainMat, axis=1)


#############rejected data used in EC3
X_new=np.concatenate((X_train_balance_logi,Reject_data_scaler),axis=0)
label_new=np.concatenate((y_train_balance_logi,reject_pred1),axis=0)



############################## Starting EC3 algorithm
O = len(Reject_data_scaler)
L = len(np.unique(y_test))
print('L',L)

cat = list(np.unique(y_test))

Salgo = np.zeros(shape=(O, 0), dtype = np.int64)
grp = []

########## list the base classifers and kmeans

# model1.fit(X_new, label_new)
# #     print('===========')
# y_train_pred = model1.predict(Reject_data_scaler)
# pred_train_prob = model1.predict_proba(Reject_data_scaler)[:,1]
# grp.append(L)
# y_train_pred = y_train_pred.astype(int)
# Salgo = np.c_[Salgo, y_train_pred]


model2.fit(X_new, label_new)
y_train_pred = model2.predict(Reject_data_scaler)
pred_train_prob = model2.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]


model3.fit(X_new, label_new)
y_train_pred = model3.predict(Reject_data_scaler)
pred_train_prob = model3.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]

model4.fit(X_new, label_new)
y_train_pred = model4.predict(Reject_data_scaler)
pred_train_prob = model4.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]

model5.fit(X_new, label_new)
y_train_pred = model5.predict(Reject_data_scaler)
pred_train_prob = model5.predict_proba(Reject_data_scaler)[:,1]
grp.append(L)
y_train_pred = y_train_pred.astype(int)
Salgo = np.c_[Salgo, y_train_pred]



Ualgo = np.zeros(shape=(O, 0), dtype=np.int64)
umodel= KMeans(n_clusters=L, random_state=0)
umodel.fit(X_new)
labels = umodel.labels_
predicted = umodel.predict(Reject_data_scaler)
n_clusters_ = len(np.unique(predicted))
predicted = [((n_clusters_) - 1) if x == -1 else x for x in predicted]
Ualgo = np.c_[Ualgo, predicted]
grp.append(n_clusters_)



# # --------------------    Parameter initialisation  ---------------------------------#
#==============================================================================

N = len(Reject_data_scaler)                        #  defined
# L                            number of classes
C1= 4                    #no. of classifier
C2 = 1                  # no. of clusters
C = C1 + C2
G1 = sum(grp[:C1])
G2 = sum(grp[C1:len(grp)])
G = G1+G2

Algo = np.column_stack((Salgo, Ualgo))
print('algo',Algo)
s = time.time()
print ("starting building matrices")
MemMat = getMemMat(N,G,C)                      #  NxG
MemDF = pd.DataFrame(MemMat)
MemMat =normalise(MemMat,1)

CoMat = np.zeros((N, N))
for i in range(N):
    for j in range(i,N):
        value = Count(i,j)
        CoMat[i][j]=CoMat[j][i] = value
# print(CoMat)
CoMat = normalise(CoMat)
CoMat = normalise(CoMat,0)
# CoMat = StochasticK(CoMat)

Objclass = getObjclass(N,L)
Grpclass = getGrpclass(G,L)

# ------------------- object - class matrix ------------------------------------#
"""
# condition satisfied   Fo >= 0 , |Fo i. | = 1  for every i in 1:n
Fo = np.zeros(shape=(len(X_test),13))

for i in range (len(X_test)):
    Fo[i][0] = 1

"""
Fo = np.random.rand(N, L)
Fo = Fo / Fo.sum(axis=1)[:, None]

# ------------------- Group - class matrix --------------------------------------#
"""# condition satisfied   Fg >= 0 , |Fg .j | = 1  for every j in 1:l

"""
Fg = np.random.rand(G, L)

Fg = Fg / Fg.sum(axis=0)[None, :]

e = time.time()
print ("all matrices have been made ")
print (e-s)
Dm = getdiagonal(MemMat , 0)
one = np.ones((G,G))
Dmdash = getdiagonal(MemMat,1)
Dc     = getdiagonal(CoMat,0)
oneN = np.ones((N,N))
ideN = np.identity(N)
t = 1
startAlgo = time.time()

alpha=0.05
beta=0.35
gamma=0.35
delta=1-alpha-beta-gamma

MainMat = EC3(Fo, Fg, MemMat, CoMat, Objclass, Grpclass, alpha,beta,gamma,delta,0.0001)

Endalgo = time.time()
print('Endalgo - startAlgo', Endalgo - startAlgo)
reject_pred2 = np.argmax(MainMat, axis=1)

Reject_label=reject_pred2.reshape(len(reject_pred2),-1)
Reject=np.concatenate((Reject_data,Reject_label),axis=1)
columns_Reject=columns_accept.copy()
columns_Reject.append('Y')
df_reject=pd.DataFrame(Reject,columns=columns_Reject)
# df_reject.to_csv('./data/Reject_pesuolable_EC3.csv',index=False)


