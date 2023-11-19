from __future__ import division
#
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from lagrangian_s3vm import *
from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,\
                            brier_score_loss, precision_score, recall_score
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler

##################
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
r = 0.5
rdm = np.random.RandomState()

for i in range(30):
    print(i)
    train = pd.read_csv('data/train.csv')
    train = train.sample(frac=0.8)
    columns_accept = list(train.columns[0:-1])
    train['Y'].replace(0, -1, inplace=True)
    X_train_valid = train[columns_accept]
    y_train_valid = train['Y']

    test = pd.read_csv('./data/test.csv')
    test['Y'].replace(0, -1, inplace=True)
    X_test = test[columns_accept]
    y_test = test['Y']

    Reject_train_ori = pd.read_csv('./data/Reject_score.csv')
    Reject_data = Reject_train_ori[columns_accept]

    aus = RandomUnderSampler(random_state=42)
    scaler = StandardScaler()
    X_train_valid_scaler = scaler.fit_transform(X_train_valid)
    X_test_scaler = scaler.transform(X_test)
    Reject_data_scaler = scaler.transform(Reject_data)
    X_train_balance_logi, y_train_balance_logi = aus.fit_resample(X_train_valid_scaler, y_train_valid)

    svc = SVC(C=0.4, probability=True)

    svc.fit(X_train_balance_logi, y_train_balance_logi)
    svc_pred = svc.predict(X_test_scaler)
    svc_prob = svc.predict_proba(X_test_scaler)[:, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, svc_prob)
    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)
    acc = accuracy_score(y_test, svc_pred)
    f1_R = f1_score(y_test, svc_pred)
    pr_score = precision_score(y_test, svc_pred)
    recal_score = recall_score(y_test, svc_pred)

    print('svm single ', roc_auc, ks)
    acc1_list.append(acc)
    f11_list.append(f1_R)
    roc1_list.append(roc_auc)
    ks1_list.append(ks)
    precision1_list.append(pr_score)
    recall1_list.append(recal_score)

    ######batch_size 要大于Reject data 数值
    lagr_s3vc, y_u = lagrangian_s3vm_train(X_train_balance_logi,
                                           y_train_balance_logi,
                                           Reject_data_scaler,
                                           svc,
                                           batch_size=20000,
                                           r=r,
                                           rdm=rdm)

    s3vm_pred = lagr_s3vc.predict(X_test_scaler)
    s3vm_prob = lagr_s3vc.predict_proba(X_test_scaler)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, s3vm_prob)
    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)
    pr_score = precision_score(y_test, s3vm_pred)
    recal_score = recall_score(y_test, s3vm_pred)
    acc = accuracy_score(y_test, s3vm_pred)
    f1_R = f1_score(y_test, s3vm_pred)

    print('s3vm single ', roc_auc, ks)
    acc2_list.append(acc)
    f12_list.append(f1_R)
    roc2_list.append(roc_auc)
    ks2_list.append(ks)
    precision2_list.append(pr_score)
    recall2_list.append(recal_score)

    print('svm ', roc1_list)
    print('s3vm ', roc2_list, ks2_list)

print('svm ', np.mean(roc1_list), np.mean(ks1_list))
print('s3vm ', np.mean(roc2_list), np.mean(ks2_list))

df=pd.DataFrame({'auc1':roc1_list,'auc2':roc2_list,
                 'ks1':ks1_list,'ks2':ks2_list,
                 'precision1':precision1_list,'precision2':precision2_list,
                 'recall1':recall1_list,'recall2':recall2_list,
                 'F11':f11_list,'F12':f12_list,
                 'acc1':acc1_list,'acc2':acc2_list
                })
# df.to_csv('./data/s3vm.csv',index=False)