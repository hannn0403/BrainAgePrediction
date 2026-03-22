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
from sklearn_rvm.em_rvm import EMRVR

# MAE, MSE, RMSE
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

# %%
def load_dataset(dataset):
    if dataset == 'ixi':
        ixi_train = pd.read_csv('./dataset/ixi_train.csv', index_col = 0)
        ixi_test = pd.read_csv('./dataset/ixi_test.csv', index_col = 0)

        # AGE에서 소수점 아래를 버림 -> for stratified k fold
        ixi_train['age'] = ixi_train['age'].astype('int64')
        ixi_test['age'] = ixi_test['age'].astype('int64')

        # 모든 feature의 dtype : float64
        ixi_train = ixi_train.astype('float64')
        ixi_test = ixi_test.astype('float64')
        return ixi_train, ixi_test
    
    elif dataset == 'hcp':
        hcp_train = pd.read_csv('./dataset/hcp_train.csv', index_col=0)
        hcp_test = pd.read_csv('./dataset/hcp_test.csv', index_col=0)

        hcp_train = hcp_train.astype('float64')
        hcp_test = hcp_test.astype('float64')
        return hcp_train, hcp_test
    
    elif dataset =='cc':
        cc_train = pd.read_csv('./dataset/cc_train.csv', index_col = 0)
        cc_test = pd.read_csv('./dataset/cc_test.csv', index_col = 0)

        cc_train = cc_train.astype('float64')
        cc_test = cc_test.astype('float64')
        return cc_train, cc_test

# ==== IXI ====

# %%
ixi_train, ixi_test = load_dataset('ixi')

# %%
ixi_setting = setup(session_id = 1, data = ixi_train, target = 'age', test_data = ixi_test, normalize = True, normalize_method = 'zscore',transformation=True, fold_strategy='stratifiedkfold', use_gpu = True)

# %%
def cal_pearson(y, y_pred):
    return np.corrcoef(y, y_pred)[0,1]

# %%
add_metric('r', 'R', cal_pearson)

# %%
# 직접 Best Model을 호출 
# 26 Models 
ixi_ada = AdaBoostRegressor(learning_rate=0.5, loss='square', n_estimators=230,random_state=1)
ixi_ard = ARDRegression(lambda_1=1.7, lambda_2=1.7)
ixi_br = BayesianRidge(alpha_1=0.0001, alpha_2=0.001, lambda_1=0.3, lambda_2=0.05)
ixi_dt = DecisionTreeRegressor(ccp_alpha=0.2, criterion='mae', max_depth=6,max_features=1.0, min_impurity_decrease=0.05,min_samples_leaf=2, min_samples_split=10)
ixi_en = ElasticNet(alpha=0.33, l1_ratio=0.668)
ixi_et = ExtraTreesRegressor(max_depth=9, max_features=1.0, min_impurity_decrease=0.02,min_samples_leaf=2, n_estimators=230, n_jobs=-1)
ixi_gbm = GradientBoostingRegressor(loss='lad', max_depth=2, max_features=1.0,min_impurity_decrease=0.01, min_samples_split=5,n_estimators=180, random_state=1, subsample=0.75)
ixi_huber = HuberRegressor(alpha=0.7, epsilon=1.8)
ixi_kr = KernelRidge(kernel='polynomial')
ixi_knn = KNeighborsRegressor(metric='manhattan', n_jobs=-1, n_neighbors=8,weights='distance')
ixi_lar = Lars(eps=1e-05, n_nonzero_coefs=53, normalize=False)
ixi_lasso = Lasso(alpha=0.2)
ixi_llar = LassoLars(alpha=0.2, normalize=False)
ixi_lgbm = LGBMRegressor(bagging_fraction=0.9, bagging_freq=2, feature_fraction=0.6,max_depth=8, min_child_samples=100, min_split_gain=0.6,n_estimators=280, num_leaves=70, reg_alpha=0.1, reg_lambda=0.001)
ixi_lr = LinearRegression()
ixi_mlp = MLPRegressor(activation='tanh', alpha=1e-07, hidden_layer_sizes=[50, 50, 100], solver='lbfgs')
ixi_omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
ixi_par = PassiveAggressiveRegressor(C=0.0001, loss='squared_epsilon_insensitive')
ixi_rf = RandomForestRegressor(criterion='mae', max_depth=13, n_jobs=-1)
ixi_ransac = RANSACRegressor(base_estimator=LinearRegression(), max_skips=20, max_trials=6,min_samples=0.95, stop_n_inliers=10, stop_probability=0.0)
ixi_ridge = Ridge(alpha=0.37, normalize=True)
ixi_svm = SVR(C=0.047, epsilon=1.55, kernel='linear')
ixi_xgboost = XGBRegressor(max_depth=5, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7)
ixi_tr = TheilSenRegressor(n_jobs=-1)
ixi_catboost = CatBoostRegressor()
ixi_gp = GaussianProcessRegressor(alpha=1.0, kernel=DotProduct(sigma_0=1))

ixi_best_models = [ixi_ada, ixi_ard, ixi_br, ixi_dt, ixi_en, ixi_et, ixi_gbm, ixi_huber, ixi_kr,
                   ixi_knn, ixi_lar, ixi_lasso, ixi_llar, ixi_lgbm, ixi_lr, ixi_mlp, ixi_omp, ixi_par,
                   ixi_rf, ixi_ransac, ixi_ridge, ixi_svm, ixi_xgboost, ixi_tr, ixi_catboost, ixi_gp]

ixi_model_names = ['Adaboost', 'ARD','Bayesian Ridge', 'Decision Tree', 'ElasticNet', 'ExtraTreesRegressor','GradientBoostingRegressor','HuberRegressor','KernelRidge','KNeighborRegressor','Lars','Lasso',
                  'LassoLars','LGBMRegressor','LinearRegression','MLPRegressor','OrthogonalMatchingPursuit','PassiveAggressiveRegressor','RandomForestRegressor','RANSAC','Ridge','SVR','XGBRegressor', 
                  'TheilsenRegressor', 'Catboost','GaussianProcess']

# ==== **Training Best Models** ====

# %%
ixi_trained_models = compare_models(n_select=26, sort='MAE', include=ixi_best_models)

# %%
print(len(ixi_trained_models))

# %%
ixi_train_sg = pull().reset_index(drop=True)
ixi_train_sg.loc[1, 'Model'] = 'Lasso Least Angle Regression'
ixi_train_sg.loc[24, 'Model'] = 'Random Sample Consensus'
# ixi_train_sg.loc[6, 'Model'] = 'Relevance Vecotr Machine'

# %%
ixi_train_sg.to_csv('./dataframe/ixi/ixi_train_score_grid.csv')

# ==== **Save Best Models** ====

# %%
for ixi_model in ixi_trained_models:
    if str(ixi_model).startswith('<catboost'):
        save_model(ixi_model, f"./models/ixi/best_model/Catboost")
    else:
        save_model(ixi_model, f"./models/ixi/best_model/{str(ixi_model).split('(')[0]}")

# ==== **Load Models** ====

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
    #load_rvm = load_model(f'./models/{dataset}/best_model/EMRVR')




    model_list = [load_ada, load_ard, load_br, load_dt, load_en, load_et, load_gbr, load_hr, load_kr, load_knn, load_lars, load_lasso, load_llar, load_lgbm,
             load_lr, load_mlp, load_omp, load_par, load_rf, load_ransac, load_ridge, load_svr, load_xgbr, load_tr, load_catboost, load_gp]
    model_name_list = ['Adaboost', 'ARD','Bayesian Ridge', 'Decision Tree', 'ElasticNet', 'ExtraTreesRegressor','GradientBoostingRegressor','HuberRegressor','KernelRidge','KNeighborRegressor','Lars','Lasso',
                  'LassoLars','LGBMRegressor','LinearRegression','MLPRegressor','OrthogonalMatchingPursuit','PassiveAggressiveRegressor','RandomForestRegressor','RANSAC','Ridge','SVR','XGBRegressor', 
                  'TheilsenRegressor', 'Catboost','GaussianProcess']
    
    return model_list, model_name_list

# %%
ixi_trained_model, ixi_trained_model_name = load_pipelines('ixi')

# ==== **Feature Importance** ====

# %%
def save_feature_importance(dataset):
    # MLP와 KNN은 제외 
    print(f"{dataset.upper()} Start!")
    
    train, test = load_dataset(dataset)
    model_list, model_name_list = load_pipelines(dataset)
    
    linear_list = [1,2,4,7,10,11,12,14,16,17,20,21,23] # 13개
    tree_list = [0,3,5,6,13,18,22,24] # 8개
    
    feat_imp_dict = {}
    
    # Save Linear Model Coefficient 
    for i in linear_list:
        model = model_list[i]['trained_model']
        feat_importance = model.coef_
        feat_imp_dict[model_name_list[i]] = feat_importance.flatten()
    
    # Save Tree Model feature importance 
    for i in tree_list:
        model = model_list[i]['trained_model']
        feat_importance = model.feature_importances_
        feat_imp_dict[model_name_list[i]] = feat_importance.flatten()
        
        
        
    # RANSAC ,Gaussian Process, Kernel Ridge는 일반적인 Linear, Tree Model 방식으로는 적용  X 
    # RANSAC  # 22
    load_ransac = model_list[19]['trained_model']
    feat_importance = load_ransac.estimator_.coef_
    feat_imp_dict['RANSAC'] = feat_importance.flatten()
    
    # Gaussian Process  # 23 
    gp_model = model_list[25]['trained_model']
    gp_linear_reg = LinearRegression()
    gp_x = gp_model.X_train_
    gp_y = gp_model.y_train_
    gp_linear_reg.fit(gp_x, gp_y)
    feat_importance = gp_linear_reg.coef_
    feat_imp_dict['GaussianProcess'] = feat_importance.flatten()
    
    # Kernel Ridge  # 24
    load_kr = model_list[8]['trained_model']
    kr_x = train.drop('age', axis=1)
    kr_y = train['age']
    
    scaler = StandardScaler()
    scaler.fit(kr_x)
    zscore_x = scaler.transform(kr_x)
    
    ridge_model = Ridge() 
    ridge_model.fit(zscore_x, kr_y)
    
    feat_importance= ridge_model.coef_
    feat_imp_dict['KernelRidge'] = feat_importance.flatten()

    
    # Convert To DataFrame 
    feat_imp_df = pd.DataFrame(feat_imp_dict)
    
    # Sorting
    #model_name_list = ['LinearRegression', 'Lasso', 'Ridge', 'Elasticnet','Lars','LassoLars','OMP','Bayesianridge','ARD','PAR','RANSAC','TheilSen','Huber','Kernelridge','SVR','GaussianProcess',
      #            'Decisiontree','Randomforest','ExtraTrees', 'Adaboost','Gradientboosting','XGboost','LightGBM','CatBoost', 'Kneighbors','MLP']

    feat_imp_list = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet','Lars','LassoLars','OrthogonalMatchingPursuit','Bayesian Ridge', 'ARD', 'PassiveAggressiveRegressor', 'RANSAC', 'TheilsenRegressor','HuberRegressor','KernelRidge','SVR','GaussianProcess',
                      'Decision Tree','RandomForestRegressor','ExtraTreesRegressor', 'Adaboost','GradientBoostingRegressor','XGBRegressor','LGBMRegressor','Catboost']
    
    feat_imp_df = feat_imp_df[feat_imp_list]
    
    return feat_imp_df


# ixi_feat_imp = save_feature_importance('ixi')
# hcp_feat_imp = save_feature_importance('hcp')
# cc_feat_imp = save_feature_importance('cc')

# %%
ixi_feat_imp = save_feature_importance('ixi')

# %%
ixi_feat_imp

# %%
def scaling(x, col_max):
    return x / col_max

for col in ixi_feat_imp.columns.to_list():
    ixi_feat_imp[col] = ixi_feat_imp[col].apply(lambda x : abs(x))
    col_max = ixi_feat_imp[col].max()
    ixi_feat_imp[col] = ixi_feat_imp[col].apply(lambda x : scaling(x, col_max))

# %%
ixi_feat_imp.describe()

# %%
ixi_feat_imp_corr = ixi_feat_imp.corr()
ixi_feat_imp_corr

# ==== Prediction ====

# ==== Predicted Brain Age ====

# %%
def save_predicted_age(dataset):
    train, test = load_dataset(dataset)
    pipe_list, model_name_list = load_pipelines(dataset)
    
    predicted_dict = {}
    
    for pipeline, model_name in zip(pipe_list, model_name_list):
        predicted_brain_age = pipeline.predict(test.drop('age', axis=1))
        predicted_dict[model_name] = predicted_brain_age
    
    predicted_df = pd.DataFrame(predicted_dict)
    
    predicted_df_list = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet','Lars','LassoLars','OrthogonalMatchingPursuit','Bayesian Ridge', 'ARD', 'PassiveAggressiveRegressor', 'RANSAC', 'TheilsenRegressor','HuberRegressor','KernelRidge','SVR', 'Relevance Vector Regressor',
                      'Decision Tree','RandomForestRegressor','ExtraTreesRegressor', 'Adaboost','GradientBoostingRegressor','XGBRegressor','LGBMRegressor','Catboost', 'GaussianProcess', 'KNeighborRegressor','MLPRegressor']
    
    predicted_df = predicted_df[predicted_df_list]
    
    return predicted_df

# %%
def test_score_df(pipe_list, model_name_list, test_df):
    score_dict = {}
    mae_list= []
    mse_list = []
    rmse_list = []
    r_list = []
    r2_list = []
    rmsle_list = []
    #mape_list=[]
    
    for pipeline in pipe_list:
        pred = pipeline.predict(test_df.drop('age', axis=1))
        mae_list.append(mean_absolute_error(test_df['age'], pred))
        mse_list.append(mean_squared_error(test_df['age'], pred))
        rmse_list.append(mean_squared_error(test_df['age'], pred, squared=False))
        r2_list.append(r2_score(test_df['age'], pred))
        r_list.append(np.corrcoef(test_df['age'], pred)[0,1])
        rmsle_list.append(math.sqrt(mean_squared_log_error(test_df['age'], pred)))
        
    score_dict['Model'] = model_name_list
    score_dict['MAE'] = mae_list
    score_dict['MSE'] = mse_list
    score_dict['RMSE'] = rmse_list
    score_dict['R'] = r_list
    score_dict['R2'] = r2_list
    score_dict['RMSLE'] = rmsle_list
    #score_dict['MAPE'] = mape_list
    
    score_df = pd.DataFrame(score_dict)
    score_df = score_df.sort_values('MAE').reset_index(drop=True)
    
    return score_df

# %%
ixi_predicted_age = save_predicted_age('ixi')

# %%
ixi_predicted_age

# ==== **Predicted Brain Age Correlation** ====

# %%
ixi_predicted_age_corr = ixi_predicted_age.corr()
ixi_predicted_age_corr

# ==== Test Score Grid ====

# %%
ixi_test_sg = test_score_df(ixi_trained_model, ixi_trained_model_name, ixi_test)

# %%
ixi_test_sg

# %%
ixi_train_sg.to_csv('./dataframe/ixi/ixi_train_score_grid.csv')
ixi_feat_imp.to_csv('./dataframe/ixi/ixi_feat_imp_scaled.csv')
ixi_feat_imp_corr.to_csv('./dataframe/ixi/ixi_feat_imp_scaled_corr.csv')
ixi_predicted_age.to_csv('./dataframe/ixi/ixi_predicted_age.csv')
ixi_predicted_age_corr.to_csv('./dataframe/ixi/ixi_predicted_age_corr.csv')
ixi_test_sg.to_csv('./dataframe/ixi/ixi_test_score_grid.csv')

# ==== CAMCAN ====

# %%
cc_train, cc_test = load_dataset('cc')

# %%
cc_setting = setup(session_id = 1, data = cc_train, target = 'age', test_data = cc_test, normalize = True, normalize_method = 'zscore',transformation=True, fold_strategy='stratifiedkfold', use_gpu = True)

# %%
add_metric('r', 'R', cal_pearson)

# %%
# 직접 Best Model을 호출 
# 26 Models 
cc_ada = AdaBoostRegressor(learning_rate=0.1, n_estimators=90, random_state=1)
cc_ard = ARDRegression(lambda_1=4, lambda_2=4)
cc_br = BayesianRidge(alpha_1=0.2, alpha_2=0.2, lambda_1=0.005, lambda_2=0.3)
cc_dt = DecisionTreeRegressor(ccp_alpha=0.15, max_depth=4, max_features=1.0,min_impurity_decrease=0.05, min_samples_leaf=3,min_samples_split=5)
cc_en = ElasticNet(alpha=0.1, l1_ratio=0.8)
cc_et = ExtraTreesRegressor(max_depth=9, max_features=1.0, min_impurity_decrease=0.02,min_samples_leaf=2, n_estimators=230, n_jobs=-1)
cc_gbm = GradientBoostingRegressor(learning_rate=0.05, max_depth=4, max_features='sqrt',min_impurity_decrease=0.05, min_samples_leaf=2,min_samples_split=4, n_estimators=260, random_state=1,subsample=0.8)
cc_huber = HuberRegressor(alpha=6, epsilon=2.5)
cc_kr = KernelRidge(kernel='polynomial')
cc_knn = KNeighborsRegressor(n_jobs=-1, n_neighbors=8, p=1, weights='distance')
cc_lar = Lars(eps=1e-05, n_nonzero_coefs=15)
cc_lasso = Lasso(alpha=0.07)
cc_llar = LassoLars(alpha=0.1, eps=1e-05, normalize=False)
cc_lgbm = LGBMRegressor(bagging_fraction=1.0, bagging_freq=5, feature_fraction=0.5,learning_rate=0.05, max_depth=9, min_child_samples=51,min_split_gain=0.2, n_estimators=230, num_leaves=4,reg_alpha=1e-07, reg_lambda=1e-06)
cc_lr = LinearRegression()
cc_mlp = MLPRegressor(activation='identity', alpha=0.5, hidden_layer_sizes=[100, 100], learning_rate='invscaling', solver='lbfgs')
cc_omp = OrthogonalMatchingPursuit(n_nonzero_coefs=71)
cc_par = PassiveAggressiveRegressor(C=0.0001, loss='squared_epsilon_insensitive')
cc_rf = RandomForestRegressor(max_depth=8, max_features=1.0, min_impurity_decrease=0.4,min_samples_leaf=4, n_estimators=230, n_jobs=-1)
cc_ransac = RANSACRegressor(base_estimator=LinearRegression(), max_skips=20, max_trials=6,min_samples=0.95, stop_n_inliers=10, stop_probability=0.0)
cc_ridge = Ridge(alpha=10.0)
cc_svm = SVR(C=0.163, epsilon=1.9, kernel='linear')
cc_xgboost = XGBRegressor(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7)
cc_tr = TheilSenRegressor(max_subpopulation=10000, n_jobs=-1)
cc_catboost = CatBoostRegressor()
cc_gp = GaussianProcessRegressor(alpha=1.0, kernel=DotProduct(sigma_0=1))

cc_best_models = [cc_ada, cc_ard, cc_br, cc_dt, cc_en, cc_et, cc_gbm, cc_huber, cc_kr,
                   cc_knn, cc_lar, cc_lasso, cc_llar, cc_lgbm, cc_lr, cc_mlp, cc_omp, cc_par,
                   cc_rf, cc_ransac, cc_ridge, cc_svm, cc_xgboost, cc_tr, cc_catboost, cc_gp]

cc_model_names = ['Adaboost', 'ARD','Bayesian Ridge', 'Decision Tree', 'ElasticNet', 'ExtraTreesRegressor','GradientBoostingRegressor','HuberRegressor','KernelRidge','KNeighborRegressor','Lars','Lasso',
                  'LassoLars','LGBMRegressor','LinearRegression','MLPRegressor','OrthogonalMatchingPursuit','PassiveAggressiveRegressor','RandomForestRegressor','RANSAC','Ridge','SVR','XGBRegressor', 
                  'TheilsenRegressor', 'Catboost','GaussianProcess']

# ==== **Training Best Models** ====

# %%
cc_trained_models = compare_models(n_select=26, sort='MAE', include=cc_best_models)

# %%
cc_train_sg = pull().reset_index(drop=True)
cc_train_sg.loc[2, 'Model'] = 'Lasso Least Angle Regression'
cc_train_sg.loc[16, 'Model'] = 'Random Sample Consensus'
# ixi_train_sg.loc[6, 'Model'] = 'Relevance Vecotr Machine'
cc_train_sg

# %%
cc_train_sg.to_csv('./dataframe/cc/cc_train_score_grid.csv')

# %%
for cc_model in cc_trained_models:
    if str(cc_model).startswith('<catboost'):
        save_model(cc_model, f"./models/cc/best_model/Catboost")
    else:
        save_model(cc_model, f"./models/cc/best_model/{str(cc_model).split('(')[0]}")

# %%
cc_trained_model, cc_trained_model_name = load_pipelines('cc')

# ==== Feature Importance ====

# %%
cc_feat_imp = save_feature_importance('cc')

# %%
def scaling(x, col_max):
    return x / col_max

for col in cc_feat_imp.columns.to_list():
    cc_feat_imp[col] = cc_feat_imp[col].apply(lambda x : abs(x))
    col_max = cc_feat_imp[col].max()
    cc_feat_imp[col] = cc_feat_imp[col].apply(lambda x : scaling(x, col_max))

# %%
cc_feat_imp_corr = cc_feat_imp.corr()

# ==== Prediction ====

# ==== Predicted Brain Age ====

# %%
cc_predicted_age = save_predicted_age('cc')

# ==== **Predicted Brain Age Correlation** ====

# %%
cc_predicted_age_corr = cc_predicted_age.corr()
cc_predicted_age_corr

# ==== Test Score Grid ====

# %%
cc_test_sg = test_score_df(cc_trained_model, cc_trained_model_name, cc_test)

# %%
cc_train_sg.to_csv('./dataframe/cc/cc_train_score_grid.csv')
cc_feat_imp.to_csv('./dataframe/cc/cc_feat_imp.csv')
cc_feat_imp_corr.to_csv('./dataframe/cc/cc_feat_imp_corr.csv')
cc_predicted_age.to_csv('./dataframe/cc/cc_predicted_age.csv')
cc_predicted_age_corr.to_csv('./dataframe/cc/cc_predicted_age_corr.csv')
cc_test_sg.to_csv('./dataframe/cc/cc_test_score_grid.csv')

# ==== HCP ====

# %%
hcp_train, hcp_test = load_dataset('hcp')

# %%
hcp_setting = setup(session_id = 1, data = hcp_train, target = 'age', test_data = hcp_test, normalize = True, normalize_method = 'zscore',transformation=True, fold_strategy='stratifiedkfold', use_gpu = True)

# %%
add_metric('r', 'R', cal_pearson)

# %%
# 직접 Best Model을 호출 
# 26 Models 
hcp_ada = AdaBoostRegressor(learning_rate=0.4, loss='exponential', n_estimators=250,random_state=1)
hcp_ard = ARDRegression(lambda_1=0.005, lambda_2=0.005)
hcp_br = BayesianRidge(alpha_1=0.2, alpha_2=0.2, lambda_1=0.005, lambda_2=0.3)
hcp_dt = DecisionTreeRegressor(ccp_alpha=0.1, criterion='mse',max_features='sqrt', min_impurity_decrease=0.0001, splitter='random')
hcp_en = ElasticNet(alpha=0.25, l1_ratio=0.001)
hcp_et = ExtraTreesRegressor(max_depth=9, max_features=1.0, min_impurity_decrease=0.02,min_samples_leaf=2, n_estimators=230, n_jobs=-1)
hcp_gbm = GradientBoostingRegressor(learning_rate=0.05, max_depth=4, max_features='sqrt',min_impurity_decrease=0.05, min_samples_leaf=2,min_samples_split=4, n_estimators=260, subsample=0.8)
hcp_huber = HuberRegressor(alpha=0.2, epsilon=1.9)
hcp_kr = KernelRidge(kernel='polynomial')
hcp_knn = KNeighborsRegressor(n_jobs=-1, n_neighbors=13, weights='distance')
hcp_lar = Lars(eps=1e-05, n_nonzero_coefs=95, normalize=False)
hcp_lasso = Lasso(alpha=0.05)
hcp_llar = LassoLars(alpha=0.05, normalize=False)
hcp_lgbm = LGBMRegressor(bagging_fraction=1.0, bagging_freq=5, feature_fraction=0.5,learning_rate=0.05, max_depth=8, min_child_samples=51,min_split_gain=0.2, n_estimators=230, num_leaves=4,reg_alpha=1e-07, reg_lambda=1e-06)
hcp_lr = LinearRegression()
hcp_mlp =MLPRegressor(activation='identity', alpha=0.3, hidden_layer_sizes=[50, 100], learning_rate='adaptive', solver='lbfgs')
hcp_omp = OrthogonalMatchingPursuit(n_nonzero_coefs=64)
hcp_par = PassiveAggressiveRegressor(C=0.0001, loss='squared_epsilon_insensitive')
hcp_rf = RandomForestRegressor(max_depth=11, max_features=1.0,min_impurity_decrease=0.0001, min_samples_leaf=5,n_estimators=130, n_jobs=-1)
hcp_ransac = RANSACRegressor(base_estimator=LinearRegression(), loss='squared_loss',max_skips=8, max_trials=19, min_samples=0.95, stop_n_inliers=1,stop_probability=0.28)
hcp_ridge = Ridge(alpha=0.37, normalize=True)
hcp_svm = SVR(kernel='linear', C=0.01, epsilon=0.1, shrinking=False)
hcp_xgboost = XGBRegressor(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7)
hcp_tr = TheilSenRegressor(max_subpopulation=10000, n_jobs=-1)
hcp_catboost = CatBoostRegressor()
hcp_gp = GaussianProcessRegressor(alpha=1.0, kernel=DotProduct(sigma_0=1))

hcp_best_models = [hcp_ada, hcp_ard, hcp_br, hcp_dt, hcp_en, hcp_et, hcp_gbm, hcp_huber, hcp_kr,
                   hcp_knn, hcp_lar, hcp_lasso, hcp_llar, hcp_lgbm, hcp_lr, hcp_mlp, hcp_omp, hcp_par,
                   hcp_rf, hcp_ransac, hcp_ridge, hcp_svm, hcp_xgboost, hcp_tr, hcp_catboost, hcp_gp]

hcp_model_names = ['Adaboost', 'ARD','Bayesian Ridge', 'Decision Tree', 'ElasticNet', 'ExtraTreesRegressor','GradientBoostingRegressor','HuberRegressor','KernelRidge','KNeighborRegressor','Lars','Lasso',
                  'LassoLars','LGBMRegressor','LinearRegression','MLPRegressor','OrthogonalMatchingPursuit','PassiveAggressiveRegressor','RandomForestRegressor','RANSAC','Ridge','SVR','XGBRegressor', 
                  'TheilsenRegressor', 'Catboost','GaussianProcess']

# ==== Training ====

# %%
hcp_trained_models = compare_models(n_select=26, sort='MAE', include=hcp_best_models)

# %%
hcp_train_sg = pull().reset_index(drop=True)
hcp_train_sg.loc[2, 'Model'] = 'Lasso Least Angle Regression'
hcp_train_sg.loc[23, 'Model'] = 'Random Sample Consensus'
# ixi_train_sg.loc[6, 'Model'] = 'Relevance Vecotr Machine'
hcp_train_sg

# %%
hcp_train_sg.to_csv('./dataframe/hcp/hcp_train_score_grid.csv')

# %%
for hcp_model in hcp_trained_models:
    if str(hcp_model).startswith('<catboost'):
        save_model(hcp_model, f"./models/hcp/best_model/Catboost")
    else:
        save_model(hcp_model, f"./models/hcp/best_model/{str(hcp_model).split('(')[0]}")

# %%
hcp_trained_model, hcp_trained_model_name = load_pipelines('hcp')

# ==== Feature Importance ====

# %%
hcp_feat_imp = save_feature_importance('hcp')

# %%
def scaling(x, col_max):
    return x / col_max

for col in hcp_feat_imp.columns.to_list():
    hcp_feat_imp[col] = hcp_feat_imp[col].apply(lambda x : abs(x))
    col_max = hcp_feat_imp[col].max()
    hcp_feat_imp[col] = hcp_feat_imp[col].apply(lambda x : scaling(x, col_max))

# %%
hcp_feat_imp_corr = hcp_feat_imp.corr()

# ==== Prediction ====

# %%
hcp_predicted_age = save_predicted_age('hcp')

# %%
hcp_predicted_age_corr = hcp_predicted_age.corr()

# %%
hcp_test_sg = test_score_df(hcp_trained_model, hcp_trained_model_name, hcp_test)

# %%
hcp_test_sg

# ==== HCP Decision Tree Model ====

# ==== Decision Tree 모델은 Normalization과 Transformation을 진행하면 너무 값이 작아져서인지 제대로 측정을 못하고, 데이터 샘플들이 전부 한곳으로 귀결되는 결과를 보여 이 과정을 거치지 않고 학습 및 Prediction을 진행해보자 ====

# %%
hcp_dt_setting = setup(session_id = 1, data = hcp_train, target = 'age', test_data = hcp_test, fold_strategy='stratifiedkfold', use_gpu = True)

# %%
add_metric('r', 'R', cal_pearson)

# ==== 기존에 Normalization & Transformation을 진행한 경우에 Train과 Test 성능은 거의 26가지의 모델들 중에서 가장 안 좋은 성능을 보였다. ====

# %%
hcp_dt = DecisionTreeRegressor(ccp_alpha=0.1, criterion='mse',max_features='sqrt', min_impurity_decrease=0.0001, splitter='random')
hcp_train_dt = create_model(hcp_dt)

# %%
save_model(hcp_train_dt, './models/hcp/best_model/DecisionTreeRegressor')

# %%
hcp_trained_model, hcp_trained_model_name = load_pipelines('hcp')

# %%
hcp_feat_imp = save_feature_importance('hcp')

# %%
def scaling(x, col_max):
    return x / col_max

for col in hcp_feat_imp.columns.to_list():
    hcp_feat_imp[col] = hcp_feat_imp[col].apply(lambda x : abs(x))
    col_max = hcp_feat_imp[col].max()
    hcp_feat_imp[col] = hcp_feat_imp[col].apply(lambda x : scaling(x, col_max))

# %%
hcp_feat_imp_corr = hcp_feat_imp.corr()

# %%
hcp_predicted_age = save_predicted_age('hcp')

# %%
hcp_predicted_age_corr = hcp_predicted_age.corr()

# %%
hcp_test_sg = test_score_df(hcp_trained_model, hcp_trained_model_name, hcp_test)

# %%
hcp_test_sg

# %%
hcp_predicted_age

# %%
hcp_train_sg.to_csv('./dataframe/hcp/hcp_train_score_grid.csv')
hcp_feat_imp.to_csv('./dataframe/hcp/hcp_feat_imp.csv')
hcp_feat_imp_corr.to_csv('./dataframe/hcp/hcp_feat_imp_corr.csv')
hcp_predicted_age.to_csv('./dataframe/hcp/hcp_predicted_age.csv')
hcp_predicted_age_corr.to_csv('./dataframe/hcp/hcp_predicted_age_corr.csv')
hcp_test_sg.to_csv('./dataframe/hcp/hcp_test_score_grid.csv')

# ==== File 정리 ====

# ==== weighted MAE ====

# %%
print('IXI :', ixi_train['age'].min(), '~', ixi_train['age'].max())
print('HCP :', hcp_train['age'].min(), '~', hcp_train['age'].max())
print('CAMCAN :', cc_train['age'].min(), '~', cc_train['age'].max())

# %%
ixi_train_sg['wMAE'] = ixi_train_sg['MAE'].apply(lambda x : x / 66.0)
ixi_train_sg = ixi_train_sg[['Model','MAE','wMAE', 'MSE','RMSE','R','R2','RMSLE','MAPE', 'TT (Sec)']]
ixi_test_sg['wMAE'] = ixi_test_sg['MAE'].apply(lambda x : x / 66.0)
ixi_test_sg = ixi_test_sg[['Model','MAE','wMAE', 'MSE','RMSE','R','R2','RMSLE']]

hcp_train_sg['wMAE'] = hcp_train_sg['MAE'].apply(lambda x : x / 15.0)
hcp_train_sg = hcp_train_sg[['Model','MAE','wMAE', 'MSE','RMSE','R','R2','RMSLE','MAPE', 'TT (Sec)']]
hcp_test_sg['wMAE'] = hcp_test_sg['MAE'].apply(lambda x : x / 15.0)
hcp_test_sg = hcp_test_sg[['Model','MAE','wMAE', 'MSE','RMSE','R','R2','RMSLE']]

cc_train_sg['wMAE'] = cc_train_sg['MAE'].apply(lambda x : x / 70.0)
cc_train_sg = cc_train_sg[['Model','MAE','wMAE', 'MSE','RMSE','R','R2','RMSLE','MAPE', 'TT (Sec)']]
cc_test_sg['wMAE'] = cc_test_sg['MAE'].apply(lambda x : x / 70.0)
cc_test_sg = cc_test_sg[['Model','MAE','wMAE', 'MSE','RMSE','R','R2','RMSLE']]

# %%
ixi_train_sg.to_csv('./dataframe/ixi/ixi_train_score_grid.csv')
ixi_test_sg.to_csv('./dataframe/ixi/ixi_test_score_grid.csv')

hcp_train_sg.to_csv('./dataframe/hcp/hcp_train_score_grid.csv')
hcp_test_sg.to_csv('./dataframe/hcp/hcp_test_score_grid.csv')

cc_train_sg.to_csv('./dataframe/cc/cc_train_score_grid.csv')
cc_test_sg.to_csv('./dataframe/cc/cc_test_score_grid.csv')

# ==== Train&Test Score Grid File Sorting & Rename ====

# ==== FEature Importance나 Predicted Age 의 경우에는 Model의 종류에 따라서 특정한 인사이트를 얻을수도 있으므로, **Linear Model, Non-linear Model, Ensemble MOdel** 순으로 정렬하고, 이름도 논문에 쓸 수 있는 형식으로 간단하게 다시 ====

# %%
ixi_train_sg = pd.read_csv('./dataframe/ixi/ixi_train_score_grid.csv', index_col=0)
hcp_train_sg = pd.read_csv('./dataframe/hcp/hcp_train_score_grid.csv', index_col=0)
cc_train_sg = pd.read_csv('./dataframe/cc/cc_train_score_grid.csv', index_col=0)

# %%
ixi_train_sg_sort= ixi_train_sg.sort_values('Model').reset_index(drop=True).drop('Model', axis=1)
hcp_train_sg_sort = hcp_train_sg.sort_values('Model').reset_index(drop=True)
cc_train_sg_sort = cc_train_sg.sort_values('Model').reset_index(drop=True).drop('Model', axis=1)

total_train_sg_sort = pd.concat([hcp_train_sg_sort, ixi_train_sg_sort, cc_train_sg_sort], axis=1)
total_train_sg_sort

# %%
orig_train_model_name = total_train_sg_sort.Model.to_list()
convert_train_model_name = ['Adaboost','ARD',' Bayesian Ridge','Catboost','Decision Tree','Elastic Net','Extra Trees','XGBoost','Gaussian Process','Gradient Boosting','Huber','KNN','Kernel Ridge',
                           'LassoLars','Lasso','Lars','LightGBM','Linear Regression','MLP','OMP','PAR','Random Forest','RANSAC','Ridge','SVR','TheilSen']

convert_train_name = {}

for orig, convert in zip(orig_train_model_name, convert_train_model_name):
    convert_train_name[orig] = convert

def train_sg_rename(model_name):
    return convert_train_name[model_name]

# %%
total_train_sg_sort['Convert Model Name'] = total_train_sg_sort['Model'].apply(lambda x : train_sg_rename(x))

# %%
total_train_sg_sort['Model'] = total_train_sg_sort['Convert Model Name'].apply(lambda x : x)
total_train_sg_sort = total_train_sg_sort.drop('Convert Model Name', axis=1)
total_train_sg_sort

# %%
ixi_test_sg_sort = ixi_test_sg.sort_values('Model').reset_index(drop=True).drop('Model', axis=1)
hcp_test_sg_sort = hcp_test_sg.sort_values('Model').reset_index(drop=True)
cc_test_sg_sort = cc_test_sg.sort_values('Model').reset_index(drop=True).drop('Model', axis=1)

total_test_sg_sort = pd.concat([hcp_test_sg_sort, ixi_test_sg_sort, cc_test_sg_sort], axis=1)
total_test_sg_sort

# %%
orig_train_model_name = total_test_sg_sort.Model.to_list()
convert_train_model_name = ['ARD','Adaboost',' Bayesian Ridge','Catboost','Decision Tree','Elastic Net','Extra Trees','Gaussian Process', 'Gradient Boosting','Huber','KNN','Kernel Ridge',
                           'LightGBM','Lars', 'Lasso', 'LassoLars','Linear Regression','MLP','OMP','PAR','RANSAC','Random Forest','Ridge','SVR','TheilSen', 'XGBoost']

convert_test_name = {}

for orig, convert in zip(orig_train_model_name, convert_train_model_name):
    convert_test_name[orig] = convert

def test_sg_rename(model_name):
    return convert_test_name[model_name]

total_test_sg_sort['Convert Model Name'] = total_test_sg_sort['Model'].apply(lambda x : test_sg_rename(x))

# %%
total_test_sg_sort['Model'] = total_test_sg_sort['Convert Model Name'].apply(lambda x : x)
total_test_sg_sort = total_test_sg_sort.drop('Convert Model Name', axis=1)
total_test_sg_sort

# %%
total_train_sg_sort.to_csv('./dataframe/total_train_score_grid.csv')
total_test_sg_sort.to_csv('./dataframe/total_test_score_grid.csv')

# ==== Feature importance, Predicted Brain Age Sorting ====

# ==== **Feature Importance** ====

# %%
ixi_feat_imp = pd.read_csv('./dataframe/ixi/ixi_feat_imp_scaled.csv', index_col=0)
hcp_feat_imp = pd.read_csv('./dataframe/hcp/hcp_feat_imp.csv', index_col=0)
cc_feat_imp = pd.read_csv('./dataframe/cc/cc_feat_imp.csv', index_col=0)

# %%
ixi_feat_imp['Decision Tree'].value_counts()

# %%
cc_feat_imp

# %%
model_type_sorting_list = ['ARD','Bayesian Ridge','ElasticNet','HuberRegressor','Lars','Lasso','LassoLars','LinearRegression','OrthogonalMatchingPursuit','PassiveAggressiveRegressor','RANSAC','Ridge','TheilsenRegressor',
                          'KernelRidge','SVR','GaussianProcess','Decision Tree', 'RandomForestRegressor', 'Adaboost','ExtraTreesRegressor','GradientBoostingRegressor','LGBMRegressor','XGBRegressor','Catboost']
print(len(model_type_sorting_list))

# %%
ixi_feat_imp = ixi_feat_imp[model_type_sorting_list]
hcp_feat_imp = hcp_feat_imp[model_type_sorting_list]
cc_feat_imp = cc_feat_imp[model_type_sorting_list]

# %%
ixi_feat_imp_sort = ixi_feat_imp
hcp_feat_imp_sort = hcp_feat_imp
cc_feat_imp_sort = cc_feat_imp

# %%
ixi_feat_imp_corr_sort = ixi_feat_imp_sort.corr()
hcp_feat_imp_corr_sort = hcp_feat_imp_sort.corr()
cc_feat_imp_corr_sort = cc_feat_imp_sort.corr()

# ==== **Predicted Brain Age** ====

# %%
cc_predicted_age.columns

# %%
model_26_type_sorting_list = ['ARD','Bayesian Ridge','ElasticNet','HuberRegressor','Lars','Lasso','LassoLars','LinearRegression','OrthogonalMatchingPursuit','PassiveAggressiveRegressor','RANSAC','Ridge','TheilsenRegressor',
                          'KernelRidge','SVR','GaussianProcess', 'KNeighborRegressor', 'MLPRegressor', 'Decision Tree', 'RandomForestRegressor', 'Adaboost','ExtraTreesRegressor','GradientBoostingRegressor','LGBMRegressor','XGBRegressor','Catboost']

# %%
ixi_predicted_age_sort = ixi_predicted_age[model_26_type_sorting_list]
hcp_predicted_age_sort = hcp_predicted_age[model_26_type_sorting_list]
cc_predicted_age_sort = cc_predicted_age[model_26_type_sorting_list]

# %%
ixi_predicted_age_corr_sort = ixi_predicted_age_sort.corr()
hcp_predicted_age_corr_sort = hcp_predicted_age_sort.corr()
cc_predicted_age_corr_sort = cc_predicted_age_sort.corr()

# ==== Visualization ====

# ==== Feature Importance ====

# %%
def save_feat_heatmap(age_feat_imp, dataset, save_file_name, vmin=None, vmax=None):
    plt.figure(figsize=(15,15))

    dataplot = sns.heatmap(age_feat_imp, vmin=vmin, vmax=vmax, cmap="YlGnBu", annot=False,square=True,cbar_kws={'shrink': 0.4})
    plt.title(f'{dataset} Feature Importance Correlation', fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(f'./visualization/{save_file_name}.png', dpi=300)

# %%
def save_feat_clustermap(age_feat_imp, dataset, save_file_name):
    g = sns.clustermap(age_feat_imp, cmap='YlGnBu', col_cluster=True, figsize=(16,16),square=True, vmin=0, vmax=1)
    #g.fig.suptitle(f'{dataset} Feature Importance Hierachical Clustering', y=0.92, fontsize=20)
    x0, _y0, _wㅁ, _h = g.cbar_pos
    g.ax_cbar.set_position([1.05, 0.3, 0.02, 0.3])
    g.ax_cbar.set_title('Feature Importance Correlation', x=3.5, y=0.15, loc='right', rotation=90)
    g.savefig(f'./visualization/{save_file_name}.png', bbox_inches='tight',pad_inches = 0, dpi=300)

# %%
save_feat_heatmap(ixi_feat_imp_corr_sort, 'IXI', 'ixi_feat_imp_heatmap',vmin=1, vmax=0)
save_feat_heatmap(hcp_feat_imp_corr_sort, 'HCP', 'hcp_feat_imp_heatmap',vmin=1, vmax=0)
save_feat_heatmap(cc_feat_imp_corr_sort, 'CAMCAN', 'cc_feat_imp_heatmap',vmin=1, vmax=0)

# %%
save_feat_clustermap(ixi_feat_imp_corr_sort, 'IXI', 'ixi_feat_imp_hierar_cluster')
save_feat_clustermap(hcp_feat_imp_corr_sort, 'HCP', 'hcp_feat_imp_hierar_cluster')
save_feat_clustermap(cc_feat_imp_corr_sort, 'CAMCAN', 'cc_feat_imp_hierar_cluster')

# ==== Predicted Brain Age ====

# %%
def save_age_heatmap(age_corr_df, dataset, save_file_name, vmin=None, vmax=None):
    plt.figure(figsize=(15,15))

    dataplot = sns.heatmap(age_corr_df, vmin=vmin, vmax=vmax, cmap="YlGnBu", annot=False, square=True,cbar_kws={'shrink': 0.4})
    plt.title(f'{dataset} Brain Age Correlation', fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(f'./visualization/{save_file_name}.png', dpi=300)

# %%
def save_age_clustermap(age_corr_df, dataset, save_file_name):
    g = sns.clustermap(age_corr_df, cmap='YlGnBu', col_cluster=True, figsize=(16,16), square=True)
                    
    #g.fig.suptitle(f'{dataset} Brain Age Hierachical Clustering', y=0.92, fontsize=20)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([1.05, 0.3, 0.02, 0.3])
    g.ax_cbar.set_title('Brain Age Correlation', y=1.05)

    g.savefig(f'./visualization/{save_file_name}.png', bbox_inches='tight',pad_inches = 0, dpi=300)

# %%
save_age_heatmap(ixi_predicted_age_corr_sort, 'IXI', 'ixi_pred_age_corr_heatmap_function')
save_age_heatmap(hcp_predicted_age_corr_sort, 'HCP', 'hcp_pred_age_corr_heatmap_function')
save_age_heatmap(cc_predicted_age_corr_sort, 'CAMCAN', 'cc_pred_age_corr_heatmap_function')

# %%
save_age_clustermap(ixi_predicted_age_corr_sort, 'IXI', 'ixi_pred_age_hierar_cluster')
save_age_clustermap(hcp_predicted_age_corr_sort, 'HCP', 'hcp_pred_age_hierar_cluster')
save_age_clustermap(cc_predicted_age_corr_sort, 'CAMCAN', 'cc_pred_age_hierar_cluster')

# ==== Violin Plot ====

# %%
ixi_predicted_age = pd.read_csv('dataframe/ixi/ixi_predicted_age.csv', index_col=0)
hcp_predicted_age = pd.read_csv('dataframe/hcp/hcp_predicted_age.csv', index_col=0)
cc_predicted_age = pd.read_csv('dataframe/cc/cc_predicted_age.csv', index_col=0)

# %%
model_convert_name_dict = {}
long_model_name_list = ixi_predicted_age.columns.to_list()
short_model_name_list = ['lr','lasso','ridge','en','lar','llar','omp','br','ard','par','ransac','tr','huber','kr','svr','gp','dt','rf','et','ada','gbm','xgb','lgbm','catboost','knn','mlp']
for i in range(len(long_model_name_list)):
    model_convert_name_dict[long_model_name_list[i]] = short_model_name_list[i]

# %%
model_convert_name_dict = {}
long_model_name_list = ixi_predicted_age.columns.to_list()
short_model_name_list = ['lr','lasso','ridge','en','lar','llar','omp','br','ard','par','ransac','tr','huber','kr','svr','gp','dt','rf','et','ada','gbm','xgb','lgbm','catboost','knn','mlp']
for i in range(len(long_model_name_list)):
    model_convert_name_dict[long_model_name_list[i]] = short_model_name_list[i]

def violin_dataframe(predicted_age):
    model_list = predicted_age.columns
    violin_df = pd.DataFrame()
    
    for col in model_list:
        subset = predicted_age.loc[:, [col]]
        subset['Model'] = col 
        subset.columns = ['Predicted_age', 'Model']
        
        violin_df = pd.concat([violin_df, subset], axis=0)
    
    violin_df = violin_df.reset_index(drop=True)
    
    violin_df['Model'] = violin_df['Model'].apply(lambda x : model_convert_name_dict[x])
    
    return violin_df

# %%
ixi_violin_df = violin_dataframe(ixi_predicted_age)
hcp_violin_df = violin_dataframe(hcp_predicted_age)
cc_violin_df = violin_dataframe(cc_predicted_age)

# %%
plt.figure(figsize=(21,7))
sns.violinplot(data=ixi_violin_df, x='Model', y='Predicted_age')
plt.savefig('./visualization/ixi_violinplot.png')
plt.show()

# %%
plt.figure(figsize=(21,7))
sns.violinplot(data=hcp_violin_df, x='Model', y='Predicted_age')
plt.savefig('./visualization/hcp_violinplot.png')
plt.show()

# %%
plt.figure(figsize=(21,7))
sns.violinplot(data=cc_violin_df, x='Model', y='Predicted_age')
plt.savefig('./visualization/cc_violinplot.png')
plt.show()

# ==== Violin Plot with Gender ====

# %%
ixi_gender = pd.read_csv('./dataset_with_sex/ixi_test.csv', index_col=0)
hcp_gender = pd.read_csv('./dataset_with_sex/hcp_test.csv', index_col=0)
cc_gender = pd.read_csv('./dataset_with_sex/cc_test.csv', index_col=0)

# %%
ixi_gender = ixi_gender['SEX'].reset_index(drop=True)
hcp_gender = hcp_gender['Sex'].reset_index(drop=True)
cc_gender = cc_gender['sex'].reset_index(drop=True)

# %%
model_convert_name_dict = {}
long_model_name_list = ixi_predicted_age.columns.to_list()
short_model_name_list = ['lr','lasso','ridge','en','lar','llar','omp','br','ard','par','ransac','tr','huber','kr','svr','gp','dt','rf','et','ada','gbm','xgb','lgbm','catboost','knn','mlp']
for i in range(len(long_model_name_list)):
    model_convert_name_dict[long_model_name_list[i]] = short_model_name_list[i]

def violin_dataframe_with_gender(predicted_age, dataset):
    model_list = predicted_age.columns
    violin_df = pd.DataFrame()
    
    if dataset == 'ixi':
        gender_col = ixi_gender
    elif dataset == 'hcp':
        gender_col = hcp_gender
    elif dataset == 'cc':
        gender_col = cc_gender
        
    if dataset == 'ixi':
        gender_col = gender_col.apply(lambda x : 'male' if x < 0 else 'female' )
    else: 
        gender_col = gender_col.apply(lambda x : 'male' if x==1 else 'female')
    
    for col in model_list:
        subset = predicted_age.loc[:, [col]]
        subset['Model'] = col 
        subset['Gender'] = gender_col
        subset.columns = ['Predicted_age', 'Model', 'Gender']
        
        violin_df = pd.concat([violin_df, subset], axis=0)
    
    violin_df = violin_df.reset_index(drop=True)
    violin_df['Model'] = violin_df['Model'].apply(lambda x : model_convert_name_dict[x])
    
    return violin_df

# %%
ixi_violin_df_with_gender = violin_dataframe_with_gender(ixi_predicted_age, 'ixi')
hcp_violin_df_with_gender = violin_dataframe_with_gender(hcp_predicted_age, 'hcp')
cc_violin_df_with_gender = violin_dataframe_with_gender(cc_predicted_age, 'cc')

# %%
ixi_violin_df_with_gender

# %%
plt.figure(figsize=(21,7))
sns.violinplot( x='Model', y='Predicted_age', data=ixi_violin_df_with_gender, hue='Gender', split=True)
plt.savefig('./visualization/ixi_with_sex_violinplot.png')
plt.show()

# %%
plt.figure(figsize=(21,7))
sns.violinplot( x='Model', y='Predicted_age', data=hcp_violin_df_with_gender, hue='Gender', split=True)
plt.savefig('./visualization/hcp_with_sex_violinplot.png')
plt.show()

# %%
plt.figure(figsize=(21,7))
sns.violinplot( x='Model', y='Predicted_age', data=cc_violin_df_with_gender, hue='Gender', split=True)
plt.savefig('./visualization/cc_with_sex_violinplot.png')
plt.show()

# ==== Relevance Vector Machine ====

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
from sklearn_rvm.em_rvm import EMRVR

# MAE, MSE, RMSE
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

# %%
rvr = EMRVR(kernel='linear', max_iter=5000)


rvr.fit(ixi_train.drop('age', axis=1), ixi_train['age'])

# %%
ixi_rvr_pred = rvr.predict(ixi_test.drop('age', axis=1), return_std=False)

# %%
mean_absolute_error(ixi_rvr_pred, ixi_test['age'])

# %%
cc_gp = GaussianProcessRegressor(alpha=1.0, kernel=DotProduct(sigma_0=1))
cc_gp.fit(cc_train.drop('age', axis=1), cc_train['age'])
cc_gp_pred = cc_gp.predict(cc_test.drop('age', axis=1))

# %%
cc_gp.L_.shape

# %%
cc_train.shape

# %%
pd.DataFrame(cc_gp.L_)

# %%
cc_gp.alpha_

# ==== Model Parameters ====

# %%
ixi_trained_model, ixi_trained_model_name = load_pipelines('ixi')
hcp_trained_model, hcp_trained_model_name = load_pipelines('hcp')
cc_trained_model, cc_trained_model_name = load_pipelines('cc')

# %%
ixi_model_param_dict = {}
for i in range(len(ixi_trained_model)):
    ixi_model_param_dict[ixi_trained_model_name[i]] = ixi_trained_model[i]['trained_model']

ixi_model_param_dict

# %%
hcp_model_param_dict = {}
for i in range(len(hcp_trained_model)):
    hcp_model_param_dict[hcp_trained_model_name[i]] = hcp_trained_model[i]['trained_model']

hcp_model_param_dict
