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
def load_dataset(dataset):
    if dataset == 'ixi':
        ixi_train = pd.read_csv('./new_IXI/ixi_train.csv', index_col = 0)
        ixi_test = pd.read_csv('./new_IXI/ixi_test.csv', index_col = 0)

        # AGE에서 소수점 아래를 버림 -> for stratified k fold
        ixi_train['age'] = ixi_train['age'].astype('int64')
        ixi_test['age'] = ixi_test['age'].astype('int64')

        # 모든 feature의 dtype : float64
        ixi_train = ixi_train.astype('float64')
        ixi_test = ixi_test.astype('float64')
        return ixi_train, ixi_test

    elif dataset == 'hcp':
        hcp_train = pd.read_csv('./new_HCP/hcp_train.csv', index_col=0)
        hcp_test = pd.read_csv('./new_HCP/hcp_test.csv', index_col=0)

        hcp_train = hcp_train.astype('float64')
        hcp_test = hcp_test.astype('float64')
        return hcp_train, hcp_test

    elif dataset =='cc':
        cc_train = pd.read_csv('./new_CAMCAN/cc_train.csv', index_col = 0)
        cc_test = pd.read_csv('./new_CAMCAN/cc_test.csv', index_col = 0)

        cc_train = cc_train.astype('float64')
        cc_test = cc_test.astype('float64')
        return cc_train, cc_test

# %%
def load_pipelines(dataset):

    # Load Model
    load_ada = load_model(f'./models/{dataset}/best_model/AdaBoostRegressor')# 1
    load_ard = load_model(f'./models/{dataset}/best_model/ARDRegression')# 2
    load_br = load_model(f'./models/{dataset}/best_model/BayesianRidge')# 3
    load_dt = load_model(f'./models/{dataset}/best_model/DecisionTreeRegressor')# 4
    load_en = load_model(f'./models/{dataset}/best_model/ElasticNet')# 5
    load_et = load_model(f'./models/{dataset}/best_model/ExtraTreesRegressor')# 6
    load_gbr = load_model(f'./models/{dataset}/best_model/GradientBoostingRegressor')# 7
    load_hr = load_model(f'./models/{dataset}/best_model/HuberRegressor')# 8
    load_kr = load_model(f'./models/{dataset}/best_model/KernelRidge')# 9
    load_knn = load_model(f'./models/{dataset}/best_model/KNeighborsRegressor')# 10
    load_lars = load_model(f'./models/{dataset}/best_model/Lars')# 11
    load_lasso = load_model(f'./models/{dataset}/best_model/Lasso')# 12
    load_llar = load_model(f'./models/{dataset}/best_model/LassoLars')# 13
    load_lgbm = load_model(f'./models/{dataset}/best_model/LGBMRegressor')# 14
    load_lr = load_model(f'./models/{dataset}/best_model/LinearRegression')# 15
    load_mlp = load_model(f'./models/{dataset}/best_model/MLPRegressor')# 16
    load_omp = load_model(f'./models/{dataset}/best_model/OrthogonalMatchingPursuit')# 17
    load_par = load_model(f'./models/{dataset}/best_model/PassiveAggressiveRegressor')# 18
    load_rf = load_model(f'./models/{dataset}/best_model/RandomForestRegressor')# 19
    load_ransac = load_model(f'./models/{dataset}/best_model/RANSACRegressor')# 10
    load_ridge = load_model(f'./models/{dataset}/best_model/Ridge')# 21
    load_svr = load_model(f'./models/{dataset}/best_model/SVR')# 22
    load_xgbr = load_model(f'./models/{dataset}/best_model/XGBRegressor')# 23
    load_tr = load_model(f'./models/{dataset}/best_model/TheilSenRegressor')# 24
    load_catboost = load_model(f'./models/{dataset}/best_model/Catboost')# 25
    load_gp = load_model(f'./models/{dataset}/best_model/GaussianProcessRegressor') # 26




    model_list = [load_ada, load_ard, load_br, load_dt, load_en, load_et, load_gbr, load_hr, load_kr, load_knn, load_lars, load_lasso, load_llar, load_lgbm,
             load_lr, load_mlp, load_omp, load_par, load_rf, load_ransac, load_ridge, load_svr, load_xgbr, load_tr, load_catboost, load_gp]
    model_name_list = ['Adaboost', 'ARD','Bayesian Ridge', 'Decision Tree', 'ElasticNet', 'ExtraTreesRegressor','GradientBoostingRegressor','HuberRegressor','KernelRidge','KNeighborRegressor','Lars','Lasso',
                  'LassoLars','LGBMRegressor','LinearRegression','MLPRegressor','OrthogonalMatchingPursuit','PassiveAggressiveRegressor','RandomForestRegressor','RANSAC','Ridge','SVR','XGBRegressor',
                  'TheilsenRegressor', 'Catboost','GaussianProcess']

    return model_list, model_name_list

# %%
# 기존에 ixi_test_score_df에서 test_df를 수정함으로써 전체 모델에 대해서 적용될 수 있도록 하였다.
# 이후에, Test Score Grid에도 추가적인 Metric을 넣어주어야 할 때에는 같은 방식으로 코드를 수정하면 된다.
def test_score_full_df(pipe_list, model_name_list, test_df):
    score_dict = {}
    mae_list= []
    mse_list = []
    rmse_list = []
    r2_list = []
    rmsle_list = []
    mape_list= []

    for pipeline in pipe_list:
        pred = pipeline.predict(test_df.drop('age', axis=1))
        mae_list.append(mean_absolute_error(test_df['age'], pred))
        mse_list.append(mean_squared_error(test_df['age'], pred))
        rmse_list.append(mean_squared_error(test_df['age'], pred, squared=False))
        r2_list.append(r2_score(test_df['age'], pred))
        rmsle_list.append(math.sqrt(mean_squared_log_error(test_df['age'], pred)))
        mape_list.append(mean_absolute_percentage_error(test_df['age'], pred))

    score_dict['Model'] = model_name_list
    score_dict['MAE'] = mae_list
    score_dict['MSE'] = mse_list
    score_dict['RMSE'] = rmse_list
    score_dict['R2'] = r2_list
    score_dict['RMSLE'] = rmsle_list
    score_dict['MAPE'] = mape_list

    score_df = pd.DataFrame(score_dict)
    score_df = score_df.sort_values('MAE').reset_index(drop=True)

    return score_df

# %%
ixi_train, ixi_test = load_dataset('ixi')
hcp_train, hcp_test = load_dataset('hcp')
cc_train, cc_test = load_dataset('cc')

# %%
ixi_pipe_list, ixi_pipe_name_list = load_pipelines('ixi')
hcp_pipe_list, hcp_pipe_name_list = load_pipelines('hcp')
cc_pipe_list, cc_pipe_name_list = load_pipelines('cc')

# %%
ixi_test_sg = test_score_full_df(ixi_pipe_list, ixi_pipe_name_list, ixi_test)
hcp_test_sg = test_score_full_df(hcp_pipe_list, hcp_pipe_name_list, hcp_test)
cc_test_sg = test_score_full_df(cc_pipe_list, cc_pipe_name_list, cc_test)

# %%
ixi_test_sg.to_csv('./new_IXI/ixi_best_model_test_score_full.csv')
hcp_test_sg.to_csv('./new_HCP/hcp_best_model_test_score_full.csv')
cc_test_sg.to_csv('./new_CAMCAN/cc_best_model_test_score_full.csv')

# ==== Weighted MAE ====

# %%
ixi_train_sg = pd.read_csv('./new_IXI/ixi_best_model_train_score.csv', index_col=0)
ixi_test_sg = pd.read_csv('./new_IXI/ixi_best_model_test_score_full.csv', index_col=0)

hcp_train_sg = pd.read_csv('./new_HCP/hcp_best_model_train_score.csv', index_col=0)
hcp_test_sg = pd.read_csv('./new_HCP/hcp_best_model_test_score_full.csv', index_col=0)

cc_train_sg = pd.read_csv('./new_CAMCAN/cc_best_model_train_score.csv', index_col=0)
cc_test_sg = pd.read_csv('./new_CAMCAN/cc_best_model_test_score_full.csv', index_col=0)

# %%
print(f'IXI Age Range : {ixi_train["age"].min()} ~ {ixi_train["age"].max()}')
print(f'HCP Age Range : {hcp_train["age"].min()} ~ {hcp_train["age"].max()}')
print(f'CAMCAN Age Range : {cc_train["age"].min()} ~ {cc_train["age"].max()}')

# %%
ixi_age_range = ixi_train['age'].max() - ixi_train['age'].min()

hcp_age_range = hcp_train['age'].max() - hcp_train['age'].min()

cc_age_range = cc_train['age'].max() - cc_train['age'].min()

# %%
ixi_age_range

# %%
hcp_age_range

# %%
cc_age_range

# %%
ixi_train_sg['wMAE'] = ixi_train_sg['MAE'] / ixi_age_range
ixi_test_sg['wMAE'] = ixi_test_sg['MAE'] / ixi_age_range

hcp_train_sg['wMAE'] = hcp_train_sg['MAE'] / hcp_age_range
hcp_test_sg['wMAE'] = hcp_test_sg['MAE'] / hcp_age_range

cc_train_sg['wMAE'] = cc_train_sg['MAE'] / cc_age_range
cc_test_sg['wMAE'] = cc_test_sg['MAE'] / cc_age_range

# %%
ixi_test_sg

# %%
hcp_test_sg

# %%
cc_test_sg

# %%
ixi_test_sg.to_csv('./new_IXI/ixi_best_model_test_score_full.csv')
hcp_test_sg.to_csv('./new_HCP/hcp_best_model_test_score_full.csv')
cc_test_sg.to_csv('./new_CAMCAN/cc_best_model_test_score_full.csv')

# %%
ixi_train_sg = ixi_train_sg[['Model', 'MAE', 'MSE', 'RMSE', 'R2','RMSLE','MAPE', 'wMAE', 'TT (Sec)']]
hcp_train_sg = hcp_train_sg[['Model', 'MAE', 'MSE', 'RMSE', 'R2','RMSLE','MAPE', 'wMAE', 'TT (Sec)']]
cc_train_sg = cc_train_sg[['Model', 'MAE', 'MSE', 'RMSE', 'R2','RMSLE','MAPE', 'wMAE', 'TT (Sec)']]

# %%
ixi_train_sg.to_csv('./new_IXI/ixi_best_model_train_score_full.csv')
hcp_train_sg.to_csv('./new_HCP/hcp_best_model_train_score_full.csv')
cc_train_sg.to_csv('./new_CAMCAN/cc_best_model_train_score_full.csv')

# %%
ixi_test_sg_model_sort = ixi_test_sg.sort_values('Model')
hcp_test_sg_model_sort = hcp_test_sg.sort_values('Model')
cc_test_sg_model_sort = cc_test_sg.sort_values('Model')

# %%
ixi_test_sg_model_sort.to_csv('./ixi_test_sg_model.csv')
hcp_test_sg_model_sort.to_csv('./hcp_test_sg_model.csv')
cc_test_sg_model_sort.to_csv('./cc_test_sg_model.csv')

# ==== Pearson Correlation ====

# %%
ixi_train_sg = pd.read_csv('./new_IXI/ixi_best_model_train_score_full.csv', index_col=0)
ixi_test_sg = pd.read_csv('./new_IXI/ixi_best_model_test_score_full.csv', index_col=0)

hcp_train_sg = pd.read_csv('./new_HCP/hcp_best_model_train_score_full.csv', index_col=0)
hcp_test_sg = pd.read_csv('./new_HCP/hcp_best_model_test_score_full.csv', index_col=0)

cc_train_sg = pd.read_csv('./new_CAMCAN/cc_best_model_train_score_full.csv', index_col=0)
cc_test_sg = pd.read_csv('./new_CAMCAN/cc_best_model_test_score_full.csv', index_col=0)

# %%
# 기존에 ixi_test_score_df에서 test_df를 수정함으로써 전체 모델에 대해서 적용될 수 있도록 하였다.
# 이후에, Test Score Grid에도 추가적인 Metric을 넣어주어야 할 때에는 같은 방식으로 코드를 수정하면 된다.
def test_score_include_corr(pipe_list, model_name_list, test_df):
    score_dict = {}
    mae_list= []
    mse_list = []
    rmse_list = []
    r2_list = []
    rmsle_list = []
    mape_list= []
    corr_list = []

    for pipeline in pipe_list:
        pred = pipeline.predict(test_df.drop('age', axis=1))
        mae_list.append(mean_absolute_error(test_df['age'], pred))
        mse_list.append(mean_squared_error(test_df['age'], pred))
        rmse_list.append(mean_squared_error(test_df['age'], pred, squared=False))
        r2_list.append(r2_score(test_df['age'], pred))
        rmsle_list.append(math.sqrt(mean_squared_log_error(test_df['age'], pred)))
        mape_list.append(mean_absolute_percentage_error(test_df['age'], pred))
        corr_list.append(np.corrcoef(test_df['age'], pred)[0,1])

    score_dict['Model'] = model_name_list
    score_dict['MAE'] = mae_list
    score_dict['MSE'] = mse_list
    score_dict['RMSE'] = rmse_list
    score_dict['R2'] = r2_list
    score_dict['RMSLE'] = rmsle_list
    score_dict['MAPE'] = mape_list
    score_dict['R'] = corr_list

    score_df = pd.DataFrame(score_dict)
    score_df = score_df.sort_values('MAE').reset_index(drop=True)

    return score_df

# %%
ixi_test_corr = test_score_include_corr(ixi_pipe_list, ixi_pipe_name_list, ixi_test)
ixi_train_corr = test_score_include_corr(ixi_pipe_list, ixi_pipe_name_list, ixi_train)

hcp_test_corr = test_score_include_corr(hcp_pipe_list, hcp_pipe_name_list, hcp_test)
hcp_train_corr = test_score_include_corr(hcp_pipe_list, hcp_pipe_name_list, hcp_train)

cc_test_corr = test_score_include_corr(cc_pipe_list, cc_pipe_name_list, cc_test)
cc_train_corr = test_score_include_corr(cc_pipe_list, cc_pipe_name_list, cc_train)

# %%
ixi_test_corr.to_csv('./new_IXI/ixi_best_model_test_score_full.csv')
hcp_test_corr.to_csv('./new_HCP/hcp_best_model_test_score_full.csv')
cc_test_corr.to_csv('./new_CAMCAN/cc_best_model_test_score_full.csv')

# %%
ixi_train_sg['R'] = ixi_train_sg['R2'].apply(lambda x : math.sqrt(x))
hcp_train_sg['R'] = hcp_train_sg['R2'].apply(lambda x : math.sqrt(x))
cc_train_sg['R'] = cc_train_sg['R2'].apply(lambda x : math.sqrt(x))

# %%
ixi_train_sg = ixi_train_sg[['Model', 'MAE','MSE','RMSE','R2','RMSLE','MAPE','wMAE','R','TT (Sec)']]
hcp_train_sg = hcp_train_sg[['Model', 'MAE','MSE','RMSE','R2','RMSLE','MAPE','wMAE','R','TT (Sec)']]
cc_train_sg = cc_train_sg[['Model', 'MAE','MSE','RMSE','R2','RMSLE','MAPE','wMAE','R','TT (Sec)']]

# %%
ixi_train_sg.to_csv('./new_IXI/ixi_best_model_train_score_full.csv')
hcp_train_sg.to_csv('./new_HCP/hcp_best_model_train_score_full.csv')
cc_train_sg.to_csv('./new_CAMCAN/cc_best_model_train_score_full.csv')

# %%
ixi_train_sg = pd.read_csv('./new_IXI/ixi_best_model_train_score_full.csv', index_col=0)
ixi_test_sg = pd.read_csv('./new_IXI/ixi_best_model_test_score_full.csv', index_col=0)

hcp_train_sg = pd.read_csv('./new_HCP/hcp_best_model_train_score_full.csv', index_col=0)
hcp_test_sg = pd.read_csv('./new_HCP/hcp_best_model_test_score_full.csv', index_col=0)

cc_train_sg = pd.read_csv('./new_CAMCAN/cc_best_model_train_score_full.csv', index_col=0)
cc_test_sg = pd.read_csv('./new_CAMCAN/cc_best_model_test_score_full.csv', index_col=0)

# %%
ixi_train_sg = ixi_train_sg.sort_values('Model').reset_index(drop=True)
ixi_test_sg = ixi_test_sg.sort_values('Model').reset_index(drop=True)

hcp_train_sg = hcp_train_sg.sort_values('Model').reset_index(drop=True)
hcp_test_sg = hcp_test_sg.sort_values('Model').reset_index(drop=True)

cc_train_sg = cc_train_sg.sort_values('Model').reset_index(drop=True)
cc_test_sg = cc_test_sg.sort_values('Model').reset_index(drop=True)

# %%
ixi_train_sg

# %%
ixi_test_sg
