import pandas as pd
from Fried_Nenmytest import *
import scipy.stats as stats

###########CD的qAlpha 临界表
qAlpha5pct = [1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313, 3.354, 3.391,
              3.426, 3.458, 3.489, 3.517, 3.544, 3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714, 3.732]

qAlpha10pct = [1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780, 2.855, 2.920, 2.978, 3.030, 3.077, 3.120, 3.159,
               3.196, 3.230, 3.261, 3.291, 3.319, 3.346, 3.371, 3.394, 3.417, 3.439, 3.459, 3.479, 3.498, 3.516]

columns_name=['auc','ks','precision','recall','F1','acc']
model1_name=['auc1','ks1','precision1','recall1','F11','acc1']
model2_name=['auc2','ks2','precision2','recall2','F12','acc2']
model3_name=['auc3','ks3','precision3','recall3','F13','acc3']

df=pd.read_csv('./data/lr.csv')

columns_1=model1_name
columns_2=model2_name
columns_3=model3_name
logi1=df[columns_1]
logi2=df[columns_3]
logi3=df[columns_2]
logi1.columns=columns_name
logi2.columns=columns_name
logi3.columns=columns_name


df=pd.read_csv('./data/svc.csv')


svm1=df[columns_1]
svm2=df[columns_3]
svm3=df[columns_2]
svm1.columns=columns_name
svm2.columns=columns_name
svm3.columns=columns_name


df=pd.read_csv('data/rf.csv')
# print(df.describe())

RF1=df[columns_1]
RF2=df[columns_3]
RF3=df[columns_2]
RF1.columns=columns_name
RF2.columns=columns_name
RF3.columns=columns_name

df=pd.read_csv('data/xgb.csv')
# print(df.describe())

xgb1=df[columns_1]
xgb2=df[columns_3]
xgb3=df[columns_2]
xgb1.columns=columns_name
xgb2.columns=columns_name
xgb3.columns=columns_name

df=pd.read_csv('./data/rf_augu_reweight.csv',encoding='gb2312')
print('augu_reweight',df.describe())
reweight=df[columns_1]
reweight.columns=columns_name
augu=df[columns_2]
augu.columns=columns_name
reweight.columns=columns_name

df=pd.read_csv('./data/s3vm.csv')
s3vm=df[columns_name]
s3vm.columns=columns_name

####对于AUC来说，构建数据矩阵
columns_auc=['auc']

columns_logi1=['logi_model1']
columns_logi2=['logi_model2']
columns_logi3=['logi_model3']


columns_svm1=['svm_model1']
columns_svm2=['svm_model2']
columns_svm3=['svm_model3']

columns_RF1=['rf_model1']
columns_RF2=['rf_model2']
columns_RF3=['rf_model3']

columns_xgb1=['xgb_model1']
columns_xgb2=['xgb_model2']
columns_xgb3=['xgb_model3']

# columns_withoutSelect=['D-SSAE(without selection)']
# columns_select=['D-SSAE']
# columns_final=['DD-SSAE(Proposed)']

auc_logi1=logi1[columns_auc]
auc_logi1.columns=columns_logi1
auc_logi2=logi2[columns_auc]
auc_logi2.columns=columns_logi2
auc_logi3=logi3[columns_auc]
auc_logi3.columns=columns_logi3

auc_svm1=svm1[columns_auc]
auc_svm1.columns=columns_svm1
auc_svm2=svm2[columns_auc]
auc_svm2.columns=columns_svm2
auc_svm3=svm3[columns_auc]
auc_svm3.columns=columns_svm3

auc_RF1=RF1[columns_auc]
auc_RF1.columns=columns_RF1
auc_RF2=RF2[columns_auc]
auc_RF2.columns=columns_RF2
auc_RF3=RF3[columns_auc]
auc_RF3.columns=columns_RF3

auc_xgb1=xgb1[columns_auc]
auc_xgb1.columns=columns_xgb1
auc_xgb2=xgb2[columns_auc]
auc_xgb2.columns=columns_xgb2
auc_xgb3=xgb3[columns_auc]
auc_xgb3.columns=columns_xgb3




#############
auc_s3vm=s3vm[['auc']]
auc_s3vm.columns=['s3vm']


auc_augu=augu[['auc']]
auc_augu.columns=['augu']

auc_reweight=reweight[['auc']]
auc_reweight.columns=['reweight']


############

auc_whole=pd.concat([auc_logi1,auc_logi2],axis=1)
auc_whole=pd.concat([auc_whole,auc_logi3],axis=1)
auc_whole=pd.concat([auc_whole,auc_svm1],axis=1)
auc_whole=pd.concat([auc_whole,auc_svm2],axis=1)
auc_whole=pd.concat([auc_whole,auc_svm3],axis=1)
auc_whole=pd.concat([auc_whole,auc_RF1],axis=1)
auc_whole=pd.concat([auc_whole,auc_RF2],axis=1)
auc_whole=pd.concat([auc_whole,auc_RF3],axis=1)

auc_whole=pd.concat([auc_whole,auc_xgb1],axis=1)
auc_whole=pd.concat([auc_whole,auc_xgb2],axis=1)
auc_whole=pd.concat([auc_whole,auc_xgb3],axis=1)

auc_whole=pd.concat([auc_whole,auc_s3vm],axis=1)

auc_whole=pd.concat([auc_whole,auc_augu],axis=1)
auc_whole=pd.concat([auc_whole,auc_reweight],axis=1)

####对于ks来说，构建数据矩阵
columns_ks=['ks']

columns_logi1=['logi_model1']
columns_logi2=['logi_model2']
columns_logi3=['logi_model3']


columns_svm1=['svm_model1']
columns_svm2=['svm_model2']
columns_svm3=['svm_model3']

columns_RF1=['rf_model1']
columns_RF2=['rf_model2']
columns_RF3=['rf_model3']

columns_xgb1=['xgb_model1']
columns_xgb2=['xgb_model2']
columns_xgb3=['xgb_model3']

# columns_withoutSelect=['D-SSAE(without selection)']
# columns_select=['D-SSAE']
# columns_final=['DD-SSAE(Proposed)']

ks_logi1=logi1[columns_ks]
ks_logi1.columns=columns_logi1
ks_logi2=logi2[columns_ks]
ks_logi2.columns=columns_logi2
ks_logi3=logi3[columns_ks]
ks_logi3.columns=columns_logi3

ks_svm1=svm1[columns_ks]
ks_svm1.columns=columns_svm1
ks_svm2=svm2[columns_ks]
ks_svm2.columns=columns_svm2
ks_svm3=svm3[columns_ks]
ks_svm3.columns=columns_svm3

ks_RF1=RF1[columns_ks]
ks_RF1.columns=columns_RF1
ks_RF2=RF2[columns_ks]
ks_RF2.columns=columns_RF2
ks_RF3=RF3[columns_ks]
ks_RF3.columns=columns_RF3

ks_xgb1=xgb1[columns_ks]
ks_xgb1.columns=columns_xgb1
ks_xgb2=xgb2[columns_ks]
ks_xgb2.columns=columns_xgb2
ks_xgb3=xgb3[columns_ks]
ks_xgb3.columns=columns_xgb3




#############
ks_s3vm=s3vm[['ks']]
ks_s3vm.columns=['s3vm']


ks_augu=augu[['ks']]
ks_augu.columns=['augu']

ks_reweight=reweight[['ks']]
ks_reweight.columns=['reweight']


############

ks_whole=pd.concat([ks_logi1,ks_logi2],axis=1)
ks_whole=pd.concat([ks_whole,ks_logi3],axis=1)

ks_whole=pd.concat([ks_whole,ks_svm1],axis=1)
ks_whole=pd.concat([ks_whole,ks_svm2],axis=1)
ks_whole=pd.concat([ks_whole,ks_svm3],axis=1)

ks_whole=pd.concat([ks_whole,ks_RF1],axis=1)
ks_whole=pd.concat([ks_whole,ks_RF2],axis=1)
ks_whole=pd.concat([ks_whole,ks_RF3],axis=1)

ks_whole=pd.concat([ks_whole,ks_xgb1],axis=1)
ks_whole=pd.concat([ks_whole,ks_xgb2],axis=1)
ks_whole=pd.concat([ks_whole,ks_xgb3],axis=1)

ks_whole=pd.concat([ks_whole,ks_s3vm],axis=1)


ks_whole=pd.concat([ks_whole,ks_augu],axis=1)
ks_whole=pd.concat([ks_whole,ks_reweight],axis=1)


####对于precision来说，构建数据矩阵
columns_precision=['precision']

columns_logi1=['logi_model1']
columns_logi2=['logi_model2']
columns_logi3=['logi_model3']


columns_svm1=['svm_model1']
columns_svm2=['svm_model2']
columns_svm3=['svm_model3']

columns_RF1=['rf_model1']
columns_RF2=['rf_model2']
columns_RF3=['rf_model3']

columns_xgb1=['xgb_model1']
columns_xgb2=['xgb_model2']
columns_xgb3=['xgb_model3']

# columns_withoutSelect=['D-SSAE(without selection)']
# columns_select=['D-SSAE']
# columns_final=['DD-SSAE(Proposed)']

precision_logi1=logi1[columns_precision]
precision_logi1.columns=columns_logi1
precision_logi2=logi2[columns_precision]
precision_logi2.columns=columns_logi2
precision_logi3=logi3[columns_precision]
precision_logi3.columns=columns_logi3

precision_svm1=svm1[columns_precision]
precision_svm1.columns=columns_svm1
precision_svm2=svm2[columns_precision]
precision_svm2.columns=columns_svm2
precision_svm3=svm3[columns_precision]
precision_svm3.columns=columns_svm3

precision_RF1=RF1[columns_precision]
precision_RF1.columns=columns_RF1
precision_RF2=RF2[columns_precision]
precision_RF2.columns=columns_RF2
precision_RF3=RF3[columns_precision]
precision_RF3.columns=columns_RF3

precision_xgb1=xgb1[columns_precision]
precision_xgb1.columns=columns_xgb1
precision_xgb2=xgb2[columns_precision]
precision_xgb2.columns=columns_xgb2
precision_xgb3=xgb3[columns_precision]
precision_xgb3.columns=columns_xgb3




#############
precision_s3vm=s3vm[['precision']]
precision_s3vm.columns=['s3vm']


precision_augu=augu[['precision']]
precision_augu.columns=['augu']

precision_reweight=reweight[['precision']]
precision_reweight.columns=['reweight']


############

precision_whole=pd.concat([precision_logi1,precision_logi2],axis=1)
precision_whole=pd.concat([precision_whole,precision_logi3],axis=1)

precision_whole=pd.concat([precision_whole,precision_svm1],axis=1)
precision_whole=pd.concat([precision_whole,precision_svm2],axis=1)
precision_whole=pd.concat([precision_whole,precision_svm3],axis=1)

precision_whole=pd.concat([precision_whole,precision_RF1],axis=1)
precision_whole=pd.concat([precision_whole,precision_RF2],axis=1)
precision_whole=pd.concat([precision_whole,precision_RF3],axis=1)

precision_whole=pd.concat([precision_whole,precision_xgb1],axis=1)
precision_whole=pd.concat([precision_whole,precision_xgb2],axis=1)
precision_whole=pd.concat([precision_whole,precision_xgb3],axis=1)

precision_whole=pd.concat([precision_whole,precision_s3vm],axis=1)


precision_whole=pd.concat([precision_whole,precision_augu],axis=1)
precision_whole=pd.concat([precision_whole,precision_reweight],axis=1)


####对于recall来说，构建数据矩阵
columns_recall=['recall']

columns_logi1=['logi_model1']
columns_logi2=['logi_model2']
columns_logi3=['logi_model3']


columns_svm1=['svm_model1']
columns_svm2=['svm_model2']
columns_svm3=['svm_model3']

columns_RF1=['rf_model1']
columns_RF2=['rf_model2']
columns_RF3=['rf_model3']

columns_xgb1=['xgb_model1']
columns_xgb2=['xgb_model2']
columns_xgb3=['xgb_model3']

# columns_withoutSelect=['D-SSAE(without selection)']
# columns_select=['D-SSAE']
# columns_final=['DD-SSAE(Proposed)']

recall_logi1=logi1[columns_recall]
recall_logi1.columns=columns_logi1
recall_logi2=logi2[columns_recall]
recall_logi2.columns=columns_logi2
recall_logi3=logi3[columns_recall]
recall_logi3.columns=columns_logi3

recall_svm1=svm1[columns_recall]
recall_svm1.columns=columns_svm1
recall_svm2=svm2[columns_recall]
recall_svm2.columns=columns_svm2
recall_svm3=svm3[columns_recall]
recall_svm3.columns=columns_svm3

recall_RF1=RF1[columns_recall]
recall_RF1.columns=columns_RF1
recall_RF2=RF2[columns_recall]
recall_RF2.columns=columns_RF2
recall_RF3=RF3[columns_recall]
recall_RF3.columns=columns_RF3

recall_xgb1=xgb1[columns_recall]
recall_xgb1.columns=columns_xgb1
recall_xgb2=xgb2[columns_recall]
recall_xgb2.columns=columns_xgb2
recall_xgb3=xgb3[columns_recall]
recall_xgb3.columns=columns_xgb3




#############
recall_s3vm=s3vm[['recall']]
recall_s3vm.columns=['s3vm']


recall_augu=augu[['recall']]
recall_augu.columns=['augu']

recall_reweight=reweight[['recall']]
recall_reweight.columns=['reweight']


############

recall_whole=pd.concat([recall_logi1,recall_logi2],axis=1)
recall_whole=pd.concat([recall_whole,recall_logi3],axis=1)

recall_whole=pd.concat([recall_whole,recall_svm1],axis=1)
recall_whole=pd.concat([recall_whole,recall_svm2],axis=1)
recall_whole=pd.concat([recall_whole,recall_svm3],axis=1)

recall_whole=pd.concat([recall_whole,recall_RF1],axis=1)
recall_whole=pd.concat([recall_whole,recall_RF2],axis=1)
recall_whole=pd.concat([recall_whole,recall_RF3],axis=1)

recall_whole=pd.concat([recall_whole,recall_xgb1],axis=1)
recall_whole=pd.concat([recall_whole,recall_xgb2],axis=1)
recall_whole=pd.concat([recall_whole,recall_xgb3],axis=1)

recall_whole=pd.concat([recall_whole,recall_s3vm],axis=1)


recall_whole=pd.concat([recall_whole,recall_augu],axis=1)
recall_whole=pd.concat([recall_whole,recall_reweight],axis=1)

####对于F1来说，构建数据矩阵
columns_F1=['F1']

columns_logi1=['logi_model1']
columns_logi2=['logi_model2']
columns_logi3=['logi_model3']


columns_svm1=['svm_model1']
columns_svm2=['svm_model2']
columns_svm3=['svm_model3']

columns_RF1=['rf_model1']
columns_RF2=['rf_model2']
columns_RF3=['rf_model3']

columns_xgb1=['xgb_model1']
columns_xgb2=['xgb_model2']
columns_xgb3=['xgb_model3']

# columns_withoutSelect=['D-SSAE(without selection)']
# columns_select=['D-SSAE']
# columns_final=['DD-SSAE(Proposed)']

F1_logi1=logi1[columns_F1]
F1_logi1.columns=columns_logi1
F1_logi2=logi2[columns_F1]
F1_logi2.columns=columns_logi2
F1_logi3=logi3[columns_F1]
F1_logi3.columns=columns_logi3

F1_svm1=svm1[columns_F1]
F1_svm1.columns=columns_svm1
F1_svm2=svm2[columns_F1]
F1_svm2.columns=columns_svm2
F1_svm3=svm3[columns_F1]
F1_svm3.columns=columns_svm3

F1_RF1=RF1[columns_F1]
F1_RF1.columns=columns_RF1
F1_RF2=RF2[columns_F1]
F1_RF2.columns=columns_RF2
F1_RF3=RF3[columns_F1]
F1_RF3.columns=columns_RF3

F1_xgb1=xgb1[columns_F1]
F1_xgb1.columns=columns_xgb1
F1_xgb2=xgb2[columns_F1]
F1_xgb2.columns=columns_xgb2
F1_xgb3=xgb3[columns_F1]
F1_xgb3.columns=columns_xgb3




#############
F1_s3vm=s3vm[['F1']]
F1_s3vm.columns=['s3vm']


F1_augu=augu[['F1']]
F1_augu.columns=['augu']

F1_reweight=reweight[['F1']]
F1_reweight.columns=['reweight']


############

F1_whole=pd.concat([F1_logi1,F1_logi2],axis=1)
F1_whole=pd.concat([F1_whole,F1_logi3],axis=1)

F1_whole=pd.concat([F1_whole,F1_svm1],axis=1)
F1_whole=pd.concat([F1_whole,F1_svm2],axis=1)
F1_whole=pd.concat([F1_whole,F1_svm3],axis=1)

F1_whole=pd.concat([F1_whole,F1_RF1],axis=1)
F1_whole=pd.concat([F1_whole,F1_RF2],axis=1)
F1_whole=pd.concat([F1_whole,F1_RF3],axis=1)

F1_whole=pd.concat([F1_whole,F1_xgb1],axis=1)
F1_whole=pd.concat([F1_whole,F1_xgb2],axis=1)
F1_whole=pd.concat([F1_whole,F1_xgb3],axis=1)

F1_whole=pd.concat([F1_whole,F1_s3vm],axis=1)


F1_whole=pd.concat([F1_whole,F1_augu],axis=1)
F1_whole=pd.concat([F1_whole,F1_reweight],axis=1)


####对于acc来说，构建数据矩阵
columns_acc=['acc']

columns_logi1=['logi_model1']
columns_logi2=['logi_model2']
columns_logi3=['logi_model3']


columns_svm1=['svm_model1']
columns_svm2=['svm_model2']
columns_svm3=['svm_model3']

columns_RF1=['rf_model1']
columns_RF2=['rf_model2']
columns_RF3=['rf_model3']

columns_xgb1=['xgb_model1']
columns_xgb2=['xgb_model2']
columns_xgb3=['xgb_model3']

# columns_withoutSelect=['D-SSAE(without selection)']
# columns_select=['D-SSAE']
# columns_final=['DD-SSAE(Proposed)']

acc_logi1=logi1[columns_acc]
acc_logi1.columns=columns_logi1
acc_logi2=logi2[columns_acc]
acc_logi2.columns=columns_logi2
acc_logi3=logi3[columns_acc]
acc_logi3.columns=columns_logi3

acc_svm1=svm1[columns_acc]
acc_svm1.columns=columns_svm1
acc_svm2=svm2[columns_acc]
acc_svm2.columns=columns_svm2
acc_svm3=svm3[columns_acc]
acc_svm3.columns=columns_svm3

acc_RF1=RF1[columns_acc]
acc_RF1.columns=columns_RF1
acc_RF2=RF2[columns_acc]
acc_RF2.columns=columns_RF2
acc_RF3=RF3[columns_acc]
acc_RF3.columns=columns_RF3

acc_xgb1=xgb1[columns_acc]
acc_xgb1.columns=columns_xgb1
acc_xgb2=xgb2[columns_acc]
acc_xgb2.columns=columns_xgb2
acc_xgb3=xgb3[columns_acc]
acc_xgb3.columns=columns_xgb3




#############
acc_s3vm=s3vm[['acc']]
acc_s3vm.columns=['s3vm']


acc_augu=augu[['acc']]
acc_augu.columns=['augu']

acc_reweight=reweight[['acc']]
acc_reweight.columns=['reweight']


############

acc_whole=pd.concat([acc_logi1,acc_logi2],axis=1)
acc_whole=pd.concat([acc_whole,acc_logi3],axis=1)

acc_whole=pd.concat([acc_whole,acc_svm1],axis=1)
acc_whole=pd.concat([acc_whole,acc_svm2],axis=1)
acc_whole=pd.concat([acc_whole,acc_svm3],axis=1)

acc_whole=pd.concat([acc_whole,acc_RF1],axis=1)
acc_whole=pd.concat([acc_whole,acc_RF2],axis=1)
acc_whole=pd.concat([acc_whole,acc_RF3],axis=1)

acc_whole=pd.concat([acc_whole,acc_xgb1],axis=1)
acc_whole=pd.concat([acc_whole,acc_xgb2],axis=1)
acc_whole=pd.concat([acc_whole,acc_xgb3],axis=1)

acc_whole=pd.concat([acc_whole,acc_s3vm],axis=1)


acc_whole=pd.concat([acc_whole,acc_augu],axis=1)
acc_whole=pd.concat([acc_whole,acc_reweight],axis=1)



############开始计算Friedman 与Nemmy test
data=auc_whole.values
# data=ks_whole.values
# data=precision_whole.values
# data=recall_whole.values
# data=acc_whole.values
# data=F1_whole.values


matrix_r = rank_matrix(-data)

alpha=0.05
N=matrix_r.shape[0]
k=matrix_r.shape[1]
print(N,k)
Friedman = friedman(N, k, matrix_r)
dfn=k-1
dfd=(k-1)*(N-1)
F_criticalValue=scipy.stats.f.ppf(q=1-alpha, dfn=dfn, dfd=dfd)
print(F_criticalValue)
print(Friedman,F_criticalValue)
if(Friedman>F_criticalValue):
    print('reject the friedman hypothesis')
else:
    print('accept the friedman hypothesis')

plt.rc('font', family='Times New Roman')

data_auc = auc_whole.values
data_ks = ks_whole.values
data_precision = precision_whole.values
data_recall = recall_whole.values
data_f1 = F1_whole.values
data_accuracy = acc_whole.values

matrix_r_auc = rank_matrix(-data_auc)
matrix_r_ks = rank_matrix(-data_ks)
matrix_r_precision = rank_matrix(-data_precision)
matrix_r_recall = rank_matrix(-data_recall)
matrix_r_f1 = rank_matrix(-data_f1)
matrix_r_acc = rank_matrix(-data_accuracy)

# qAlpha10pct
#######计算Nemmy
# qAlpha5=qAlpha5pct[k-2]
qAlpha5 = qAlpha10pct[k - 2]
print('k', k)
print('qAlpha5', qAlpha5)
CD = 0.85 * nemenyi(N, k, qAlpha5)
print('CD is ', CD)

params = {"text.color": "blue"}
plt.rcParams.update(params)

plt.figure(figsize=(20, 24), dpi=80)
# plt.figure(figsize=(20,8), dpi=80)
plt.figure(1)
ax1 = plt.subplot(321)
# ##画CD图
rank_x = list(map(lambda x: np.mean(x), matrix_r_auc.T))
#########比较rank_x两个算法的序值差大于CD，则这两个算法性能有明显差异
name_y = list(auc_whole.columns)
min_ = [x for x in rank_x - CD / 2]
max_ = [x for x in rank_x + CD / 2]
plt.title("AUC_Friedman test ", fontdict={'fontsize': 18})
plt.scatter(rank_x, name_y, c='crimson')
plt.hlines(name_y, min_, max_, colors='b')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax2 = plt.subplot(322)
rank_x = list(map(lambda x: np.mean(x), matrix_r_ks.T))
#########比较rank_x两个算法的序值差大于CD，则这两个算法性能有明显差异
name_y = list(ks_whole.columns)
min_ = [x for x in rank_x - CD / 2]
max_ = [x for x in rank_x + CD / 2]
plt.title("KS_Friedman test ", fontdict={'fontsize': 18})
plt.scatter(rank_x, name_y, c='crimson')
plt.hlines(name_y, min_, max_, colors='b')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.show()

ax3 = plt.subplot(323)
rank_x = list(map(lambda x: np.mean(x), matrix_r_precision.T))
#########比较rank_x两个算法的序值差大于CD，则这两个算法性能有明显差异
name_y = list(ks_whole.columns)
min_ = [x for x in rank_x - CD / 2]
max_ = [x for x in rank_x + CD / 2]
plt.title("Precision_Friedman test ", fontdict={'fontsize': 18})
plt.scatter(rank_x, name_y, c='crimson')
plt.hlines(name_y, min_, max_, colors='b')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax4 = plt.subplot(324)
rank_x = list(map(lambda x: np.mean(x), matrix_r_recall.T))
#########比较rank_x两个算法的序值差大于CD，则这两个算法性能有明显差异
name_y = list(ks_whole.columns)
min_ = [x for x in rank_x - CD / 2]
max_ = [x for x in rank_x + CD / 2]
plt.title("Recall_Friedman test ", fontdict={'fontsize': 18})
plt.scatter(rank_x, name_y, c='crimson')
plt.hlines(name_y, min_, max_, colors='b')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax5 = plt.subplot(325)
rank_x = list(map(lambda x: np.mean(x), matrix_r_f1.T))
#########比较rank_x两个算法的序值差大于CD，则这两个算法性能有明显差异
name_y = list(ks_whole.columns)
min_ = [x for x in rank_x - CD / 2]
max_ = [x for x in rank_x + CD / 2]
plt.title("F1_Friedman test ", fontdict={'fontsize': 18})
plt.scatter(rank_x, name_y, c='crimson')
plt.hlines(name_y, min_, max_, colors='b')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.show()

ax6 = plt.subplot(326)
rank_x = list(map(lambda x: np.mean(x), matrix_r_acc.T))
#########比较rank_x两个算法的序值差大于CD，则这两个算法性能有明显差异
name_y = list(ks_whole.columns)
min_ = [x for x in rank_x - CD / 2]
max_ = [x for x in rank_x + CD / 2]
plt.title("Accuracy_Friedman test ", fontdict={'fontsize': 18})
plt.scatter(rank_x, name_y, c='crimson')
plt.hlines(name_y, min_, max_, colors='b')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()
