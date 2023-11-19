#################################
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

train_ori=pd.read_csv('data/train.csv')
columns_accept=list(train_ori.columns[0:-1])

Reject=pd.read_csv('./data/Reject_data.csv')
print(Reject.shape)
clf_reject=IsolationForest(random_state=42).fit(Reject)
Reject=Reject[columns_accept]
score=clf_reject.decision_function(Reject)
Reject['score']=score

#########

Reject_data=Reject[columns_accept]

# numbers = np.arange(0, Reject['score'].max(), 0.05)
# numbers_list = numbers.tolist()
# for m in numbers_list:
#     index_list=list(Reject[Reject['score']<0.08].index)
############0.08 is the final threshold from above pool

index_list=list(Reject[Reject['score']<0.08].index)
mask=Reject.index.isin(index_list)
Reject=Reject[mask]
print(Reject.shape)
