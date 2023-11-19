from __future__ import division
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import pandas as pd
from lagrangian_s3vm import *
# from utils import *
from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,\
                            brier_score_loss, precision_score, recall_score
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()

# train=pd.read_csv('./data/train_final.csv')
roc1_list = []
ks1_list = []
acc1_list = []
bs1_list = []
f11_list = []
precision1_list = []
recall1_list = []
typeOneError1_list = []
typeTwoError1_list = []

roc2_list = []
ks2_list = []
acc2_list = []
bs2_list = []
f12_list = []
precision2_list = []
recall2_list = []
typeOneError2_list = []
typeTwoError2_list = []


train = pd.read_csv('data/train.csv')
columns_accept = list(train.columns[0:-1])
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('data/train.csv')

X_train_valid = train[columns_accept]
y_train_valid = train['Y']

test = pd.read_csv('./data/test.csv')
X_test = test[columns_accept]
y_test = test['Y']

Reject_train_ori = pd.read_csv('./data/Reject_score.csv')
Reject_data = Reject_train_ori[columns_accept]

for i in range(1):

    aus = RandomUnderSampler()
    sm = BorderlineSMOTE(kind='borderline-1')
    scaler = StandardScaler()
    X_train_valid_scaler = scaler.fit_transform(X_train_valid)
    X_test_scaler = scaler.transform(X_test)
    Reject_data_scaler = scaler.transform(Reject_data)
    X_train_balance_logi, y_train_balance_logi = aus.fit_resample(X_train_valid_scaler, y_train_valid)
    ##############Basic
    #     model = LogisticRegression(C=0.1)
    model = RandomForestClassifier(n_estimators=280,min_samples_split=30,
                                  min_samples_leaf=20,max_depth=7,criterion='gini'
                                                     )

    ##################  Augmentation

    model.fit(X_train_balance_logi, y_train_balance_logi)
    Reject_data_pro = model.predict_proba(Reject_data_scaler)[:, 1].astype('float')

    Reject = pd.DataFrame(data=Reject_data_scaler, columns=Reject_data.columns)
    Reject['pro'] = Reject_data_pro
    Reject['Y'] = Reject['pro'].apply(lambda x: 1 if x > 0.6 else 0)
    Reject.drop(columns=['pro'], inplace=True)

    Reject_data = Reject[columns_accept]
    Reject_label = Reject['Y']
    Reject_balance_data, Reject_balance_label = aus.fit_resample(Reject_data, Reject_label)
    #     Reject_balance_data, Reject_balance_label = Reject_data, Reject_label
    Reject = pd.DataFrame(data=Reject_balance_data, columns=columns_accept)
    Reject['Y'] = list(Reject_balance_label)

    Train_accept = pd.DataFrame(data=X_train_balance_logi, columns=Reject_data.columns)
    Train_accept['Y'] = list(y_train_balance_logi)

    Train = pd.concat([Train_accept, Reject], axis=0)
    Train_x = Train[Reject_data.columns]
    Train_y = Train['Y']

    model.fit(Train_x, Train_y)

    predicted = model.predict(X_test_scaler).astype('int')
    pro = model.predict_proba(X_test_scaler)[:, 1].astype('float')
    fpr, tpr, _ = metrics.roc_curve(y_test, pro)
    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)

    acc = accuracy_score(y_test, predicted)
    f1_R = f1_score(y_test, predicted)
    BS = brier_score_loss(y_test, pro)

    pr_score = precision_score(y_test, predicted)
    recal_score = recall_score(y_test, predicted)
    TN, FP, FN, TP = confusion_matrix(y_test, predicted).ravel()
    Type_1_error = FP / (FP + TN)
    Type_2_error = FN / (TP + FN)

    precision1_list.append(pr_score)
    recall1_list.append(recal_score)
    typeOneError1_list.append(Type_1_error)
    typeTwoError1_list.append(Type_2_error)

    acc1_list.append(acc)
    f11_list.append(f1_R)
    roc1_list.append(roc_auc)
    ks1_list.append(ks)
    bs1_list.append(BS)
    #############################################Reweighting

    model.fit(X_train_balance_logi, y_train_balance_logi)
    Reject_data_pro = model.predict_proba(Reject_data_scaler)[:, 1].astype('float')
    Reject_data_pre = model.predict(Reject_data_scaler).astype('float')
    Reject = pd.DataFrame(data=Reject_data_scaler, columns=columns_accept)
    Reject['rejected'] = 1
    Reject['score'] = Reject_data_pro
    Reject['Y'] = Reject_data_pre

    Accept_data_pro = model.predict_proba(X_train_valid_scaler)[:, 1].astype('float')
    Accept = pd.DataFrame(data=X_train_valid_scaler, columns=columns_accept)

    Accept['rejected'] = 0
    Accept['score'] = Accept_data_pro
    Accept['Y'] = y_train_valid

    Train = pd.concat([Accept, Reject], axis=0)

    aug_df = Train.sort_values(by="score", ascending=False)
    aug_df['accept_score'] = 1 - aug_df['score']
    intervals = np.array_split(aug_df, 10)
    print('len intervals', len(intervals))

    accept_list = []
    band_list = []
    for band in intervals:
        accepts = band.loc[band["rejected"] == 0, "rejected"].count()
        rejects = band.loc[band["rejected"] == 1, "rejected"].count()
        weight = (rejects + accepts) / accepts
        band.drop(band[band['rejected'] == 1].index, inplace=True)
        band['weight'] = weight

        accept_list.append(accepts)
        band_list.append(len(band))

    aug_df = pd.concat(intervals)
    print(len(train), len(Reject), len(Train), len(aug_df))
    print(np.sum(accept_list), np.sum(band_list))
    columns_accept1 = columns_accept.copy()
    columns_accept1.append('Y')
    aug_df.drop_duplicates(inplace=True)
    aug_df = Accept.merge(aug_df, on=columns_accept1, how='inner')

    columns_accept_weight1 = columns_accept.copy()
    columns_accept_weight1.append('weight')

    aug_df = aug_df.dropna(subset=['Y'])

    y_train_aug_all = aug_df['Y']
    X_train_aug_all = aug_df[columns_accept_weight1]
    train_weights_aug_all = aug_df['weight']
    #     model = LogisticRegression(C=10)

    X_train_aug_all1, y_train_aug_all1 = aus.fit_resample(X_train_aug_all, y_train_aug_all)
    X_train_aug_all = X_train_aug_all1[columns_accept]
    y_train_aug_all = y_train_aug_all1
    weight = X_train_aug_all1['weight']
    model.fit(X_train_aug_all, y_train_aug_all, sample_weight=weight)

    # model.fit(X_train_aug_all, y_train_aug_all)

    predicted = model.predict(X_test_scaler).astype('int')
    pro = model.predict_proba(X_test_scaler)[:, 1].astype('float')
    fpr, tpr, _ = metrics.roc_curve(y_test, pro)
    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)

    acc = accuracy_score(y_test, predicted)
    f1_R = f1_score(y_test, predicted)
    BS = brier_score_loss(y_test, pro)

    pr_score = precision_score(y_test, predicted)
    recal_score = recall_score(y_test, predicted)
    TN, FP, FN, TP = confusion_matrix(y_test, predicted).ravel()
    Type_1_error = FP / (FP + TN)
    Type_2_error = FN / (TP + FN)

    precision2_list.append(pr_score)
    recall2_list.append(recal_score)
    typeOneError2_list.append(Type_1_error)
    typeTwoError2_list.append(Type_2_error)

    acc2_list.append(acc)
    f12_list.append(f1_R)
    roc2_list.append(roc_auc)
    ks2_list.append(ks)
    bs2_list.append(BS)

df=pd.DataFrame({'auc1':roc1_list,'auc2':roc2_list,
                 'ks1':ks1_list,'ks2':ks2_list,
                 'precision1':precision1_list,
                 'recall1':recall1_list,'recall2':recall2_list,
                 'F11':f11_list,'F12':f12_list,
                 'acc1':acc1_list,'acc2':acc2_list,
                })
df.describe()

# df.to_csv('./data/rf_augu_reweight.csv',index=False)
