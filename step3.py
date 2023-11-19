import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option("display.max_columns", 75)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy  as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler


train=pd.read_csv('data/train.csv')
columns_accept=list(train.columns[0:-1])
test = pd.read_csv('./data/test.csv')
X_test = test[columns_accept]
y_test = test['Y']
#

#########
Reject_train_EC3_select=pd.read_csv('data/Reject_pesuolable_EC3_select.csv')
Reject_train_EC3_whole=pd.read_csv('data/Reject_pesuolable_EC3.csv')

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

roc3_list = []
ks3_list = []
acc3_list = []
bs3_list = []
f13_list = []
precision3_list = []
recall3_list = []
typeOneError3_list = []
typeTwoError3_list = []


for i in range(3):
    print(i)
    X_train_valid, X_valid, y_train_valid, y_valid = train_test_split(train[columns_accept],
                                                              train['Y'], test_size=0.2)
    #####新增
    aus = RandomUnderSampler(random_state=42)
    scaler = StandardScaler()
    X_train_valid_scaler = scaler.fit_transform(X_train_valid)
    X_test_scaler = scaler.transform(X_test)

    X_train_balance_logi, y_train_balance_logi = aus.fit_resample(X_train_valid_scaler, y_train_valid)

    #############增加的索引

    Reject_data_EC3 = Reject_train_EC3_select[columns_accept]
    Reject_label_EC3 = Reject_train_EC3_select['Y2']

    Reject_data_EC3_whole = Reject_train_EC3_whole[columns_accept]
    Reject_label_EC3_whole = Reject_train_EC3_whole['Y2']

    Reject_data_scaler_EC3 = scaler.transform(Reject_data_EC3)
    Reject_data_scaler_EC3_whole = scaler.transform(Reject_data_EC3_whole)

    #####################只是改变了标签
    #     Reject_data_scaler_balance_EC3,Reject_label_balance_EC3=aus.fit_resample(Reject_data_scaler,Reject_label_EC3)

    ###########extra
    ################################为basic+whole
    Reject_data_scaler_balance_EC3, Reject_label_balance_EC3 = aus.fit_resample(Reject_data_scaler_EC3_whole,
                                                                                Reject_label_EC3_whole)
    #     Reject_data_scaler_balance_EC3,Reject_label_balance_EC3=Reject_data_scaler_EC3_extra,Reject_label_EC3_extra
    whole_data = np.concatenate((X_train_balance_logi, Reject_data_scaler_balance_EC3), axis=0)
    whole_label = pd.concat([y_train_balance_logi, Reject_label_balance_EC3], axis=0)

    Reject_data_scaler_balance_EC3, Reject_label_balance_EC3 = aus.fit_resample(Reject_data_scaler_EC3,
                                                                                Reject_label_EC3)
    #     Reject_data_scaler_balance_EC3,Reject_label_balance_EC3=Reject_data_scaler_EC3,Reject_label_EC3
    whole_data_select = np.concatenate((X_train_balance_logi, Reject_data_scaler_balance_EC3), axis=0)
    whole_label_select = pd.concat([y_train_balance_logi, Reject_label_balance_EC3], axis=0)

    ##############1. Basic
    model = LogisticRegression(C=10, max_iter=500)
    model = RandomForestClassifier(n_estimators=280, min_samples_split=10,
                                   min_samples_leaf=10, max_depth=13, max_features=11,
                                   criterion='gini')

    ########xgboost
    model =xgb.XGBClassifier(learning_rate =0.01,n_estimators=280,max_depth=8, min_child_weight=3,
                    gamma=0.4,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',
                    nthread=4,scale_pos_weight=1,seed=27, importance_type='gini',reg_alpha=0)
    #
    # model=SVC(C=0.4,probability=True)

    model.fit(X_train_balance_logi, y_train_balance_logi)
    predicted = model.predict(X_test_scaler).astype('int')
    pro = model.predict_proba(X_test_scaler)[:, 1].astype('float')
    fpr, tpr, _ = metrics.roc_curve(y_test, pro)

    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)
    acc = accuracy_score(y_test, predicted)
    f1_R = f1_score(y_test, predicted)
    pr_score = precision_score(y_test, predicted)
    recal_score = recall_score(y_test, predicted)

    acc1_list.append(acc)
    f11_list.append(f1_R)
    roc1_list.append(roc_auc)
    ks1_list.append(ks)
    precision1_list.append(pr_score)
    recall1_list.append(recal_score)


    ###########2：basic+all
    model.fit(whole_data, whole_label)
    predicted = model.predict(X_test_scaler).astype('int')
    pro = model.predict_proba(X_test_scaler)[:, 1].astype('float')
    fpr, tpr, _ = metrics.roc_curve(y_test, pro)
    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)


    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)
    acc = accuracy_score(y_test, predicted)
    f1_R = f1_score(y_test, predicted)
    pr_score = precision_score(y_test, predicted)
    recal_score = recall_score(y_test, predicted)

    acc2_list.append(acc)
    f12_list.append(f1_R)
    roc2_list.append(roc_auc)
    ks2_list.append(ks)
    precision2_list.append(pr_score)
    recall2_list.append(recal_score)



    #####################3. basic+select
    model.fit(whole_data_select, whole_label_select)
    predicted = model.predict(X_test_scaler).astype('int')
    pro = model.predict_proba(X_test_scaler)[:, 1].astype('float')
    fpr, tpr, _ = metrics.roc_curve(y_test, pro)

    roc_auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)
    acc = accuracy_score(y_test, predicted)
    f1_R = f1_score(y_test, predicted)
    pr_score = precision_score(y_test, predicted)
    recal_score = recall_score(y_test, predicted)

    acc3_list.append(acc)
    f13_list.append(f1_R)
    roc3_list.append(roc_auc)
    ks3_list.append(ks)
    precision3_list.append(pr_score)
    recall3_list.append(recal_score)
#
df=pd.DataFrame({'auc1':roc1_list,'auc2':roc2_list,'auc3':roc3_list,
                 'ks1':ks1_list,'ks2':ks2_list,'ks3':ks3_list,
                 'precision1':precision1_list,'precision2':precision2_list,'precision3':precision3_list,
                 'recall1':recall1_list,'recall2':recall2_list,'recall3':recall3_list,
                 'F11':f11_list,'F12':f12_list,'F13':f13_list,
                 'acc1':acc1_list,'acc2':acc2_list,'acc3':acc3_list,
                })
print(df.describe())
