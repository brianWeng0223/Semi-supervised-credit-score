from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

################Take lr as example to show how to set hyperparameter of base learner#
train_ori=pd.read_csv('data/train.csv')
columns_accept=list(train_ori.columns[0:-1])

X = train_ori[columns_accept]
y = train_ori['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_valid_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

aus=RandomUnderSampler(random_state=42)
X_train_balance_logi, y_train_balance_logi = aus.fit_resample(X_train_valid_scaler, y_train)
#
model = LogisticRegression(max_iter=200)

# set grid parameter
param_grid = {'C': [0.1, 1, 10]}

#
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_balance_logi, y_train_balance_logi)

#
print("Best parameters from GridSearchCV: ", grid_search.best_params_)
print("Best score from GridSearchCV: ", grid_search.best_score_)

# 使用 cross_val_score 在整个数据集上验证模型性能
cross_val_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print("Cross-validation scores on the entire dataset: ", cross_val_scores)
print("Mean cross-validation score: ", cross_val_scores.mean())



