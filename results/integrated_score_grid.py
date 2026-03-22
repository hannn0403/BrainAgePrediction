# %%
from pycaret.regression import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import *
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost.core import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct

# MAE, MSE, RMSE
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_percentage_error

# %%
ixi_train_sg = pd.read_csv('./new_IXI/ixi_best_model_train_score_full.csv', index_col=0)
ixi_test_sg = pd.read_csv('./new_IXI/ixi_best_model_test_score_full.csv', index_col=0)

hcp_train_sg = pd.read_csv('./new_HCP/hcp_best_model_train_score_full.csv', index_col=0)
hcp_test_sg = pd.read_csv('./new_HCP/hcp_best_model_test_score_full.csv', index_col=0)

cc_train_sg = pd.read_csv('./new_CAMCAN/cc_best_model_train_score_full.csv', index_col=0)
cc_test_sg = pd.read_csv('./new_CAMCAN/cc_best_model_test_score_full.csv', index_col=0)

# %%
test_convert_name_dict = {'Adaboost':'Adaboost',
                     'ARD':'ARD',
                     'Bayesian Ridge':'Bayesianridge',
                     'Decision Tree':'Decisiontree',
                     'ElasticNet':'Elasticnet',
                     'ExtraTreesRegressor':'ExtraTrees',
                     'XGBRegressor':'XGboost',
                     'GradientBoostingRegressor':'Gradientboosting',
                     'HuberRegressor':'Huber',
                     'KNeighborRegressor':'Kneighbors',
                     'KernelRidge':'Kernelridge',
                     'LassoLars':'LassoLars',
                     'Lasso':'Lasso',
                     'Lars':'Lars',
                     'LGBMRegressor':'LightGBM',
                     'LinearRegression':'LinearRegression',
                     'MLPRegressor':'MLP',
                     'OrthogonalMatchingPursuit':'OMP',
                     'PassiveAggressiveRegressor':'PAR',
                     'RandomForestRegressor':'Randomforest',
                     'RANSAC':'RANSAC',
                     'Ridge':'Ridge',
                     'SVR':'SVR',
                     'Catboost':'CatBoost',
                     'GaussianProcess':'GaussianProcess',
                     'TheilsenRegressor':'TheilSen'}


train_convert_name_dict = {'AdaBoost Regressor':'Adaboost',
                     'Automatic Relevance Determination':'ARD',
                     'Bayesian Ridge':'Bayesianridge',
                     'Decision Tree Regressor':'Decisiontree',
                     'Elastic Net':'Elasticnet',
                     'Extra Trees Regressor':'ExtraTrees',
                     'Extreme Gradient Boosting':'XGboost',
                     'Gradient Boosting Regressor':'Gradientboosting',
                     'Huber Regressor':'Huber',
                     'K Neighbors Regressor':'Kneighbors',
                     'Kernel Ridge':'Kernelridge',
                     'Lasso Least Angle Regression':'LassoLars',
                     'Lasso Regression':'Lasso',
                     'Least Angle Regression':'Lars',
                     'Light Gradient Boosting Machine':'LightGBM',
                     'Linear Regression':'LinearRegression',
                     'MLP Regressor':'MLP',
                     'Orthogonal Matching Pursuit':'OMP',
                     'Passive Aggressive Regressor':'PAR',
                     'Random Forest Regressor':'Randomforest',
                     'Random Sample Consensus':'RANSAC',
                     'Ridge Regression':'Ridge',
                     'Support Vector Regression':'SVR',
                     'CatBoost Regressor':'CatBoost',
                     'GaussianProcessRegressor':'GaussianProcess',
                     'TheilSen Regressor':'TheilSen'}


def convert_model_name(pre_model, mode):
    if mode == 'train':
        return train_convert_name_dict[pre_model]
    elif mode == 'test':
        return test_convert_name_dict[pre_model]

# %%
ixi_train_sg['Model'] = ixi_train_sg['Model'].apply(lambda x : convert_model_name(x, mode='train'))
ixi_test_sg['Model'] = ixi_test_sg['Model'].apply(lambda x : convert_model_name(x, mode='test'))

hcp_train_sg['Model'] = hcp_train_sg['Model'].apply(lambda x : convert_model_name(x, mode='train'))
hcp_test_sg['Model'] = hcp_test_sg['Model'].apply(lambda x : convert_model_name(x, mode='test'))

cc_train_sg['Model'] = cc_train_sg['Model'].apply(lambda x : convert_model_name(x, mode='train'))
cc_test_sg['Model'] = cc_test_sg['Model'].apply(lambda x : convert_model_name(x, mode='test'))

# %%
ixi_train_sg = ixi_train_sg.sort_values('Model').reset_index(drop=True)
ixi_test_sg = ixi_test_sg.sort_values('Model').reset_index(drop=True)

hcp_train_sg = hcp_train_sg.sort_values('Model').reset_index(drop=True)
hcp_test_sg = hcp_test_sg.sort_values('Model').reset_index(drop=True)

cc_train_sg = cc_train_sg.sort_values('Model').reset_index(drop=True)
cc_test_sg = cc_test_sg.sort_values('Model').reset_index(drop=True)

# ==== Dataset order: HCP, IXI, CAMCAN. Separate Score Grids for Train and Test. ====

# %%
ixi_train_sg = ixi_train_sg.drop('Model', axis=1)
cc_train_sg = cc_train_sg.drop('Model', axis=1)

# %%
integrated_train_sg = pd.concat([hcp_train_sg, ixi_train_sg, cc_train_sg], axis=1)

# %%
integrated_train_sg.to_csv('./integrated_train_sg.csv')

# %%
ixi_test_sg = ixi_test_sg.drop('Model', axis=1)
cc_test_sg = cc_test_sg.drop('Model', axis=1)

# %%
integrated_test_sg = pd.concat([hcp_test_sg, ixi_test_sg, cc_test_sg], axis=1)
integrated_test_sg.to_csv('./integrated_test_sg.csv')
