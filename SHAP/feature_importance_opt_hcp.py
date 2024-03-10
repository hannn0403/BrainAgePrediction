import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
import yaml
from sklearn.model_selection import StratifiedKFold
from functions.models import get_ensemble_model, get_linear_model, get_nonlinear_model
from functions.models import get_model_explanations, get_age_corrected_model_explanations, correct_age_predictions
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings('ignore')


df_sheet_1 = pd.read_excel('./feat_imp/total.xlsx', sheet_name='Sheet1')
df_sheet_2 = pd.read_excel('./feat_imp/total.xlsx', sheet_name='Sheet2')
df_sheet_3 = pd.read_excel('./feat_imp/total.xlsx', sheet_name='Sheet3')
df_sheet_4 = pd.read_excel('./feat_imp/total.xlsx', sheet_name='Sheet4')
df_sheet_5 = pd.read_excel('./feat_imp/total.xlsx', sheet_name='Sheet5')


feature_name = df_sheet_3.iloc[:, 0]
hcp_exp = df_sheet_3.iloc[:, 1:4]
cc_exp = df_sheet_3.iloc[:, 4:7]
ixi_exp = df_sheet_3.iloc[:, 7:]

hcp_exp = pd.concat([feature_name, hcp_exp], axis=1)
cc_exp = pd.concat([feature_name, cc_exp], axis=1)
ixi_exp = pd.concat([feature_name, ixi_exp], axis=1)

hcp_header = hcp_exp.iloc[0]
hcp_exp = hcp_exp[1:]
hcp_exp.rename(columns=hcp_header, inplace=True)
hcp_exp

with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
print('')
print('---------------------------------------------------------')
print('Configuration:')
print(yaml.dump(cfg, default_flow_style=False, default_style=''))
print('---------------------------------------------------------')
print('')

# set paths
datapath = cfg['paths']['datapath']
metricpath = datapath + 'surfaces/'
outpath = cfg['paths']['results']
genpath = cfg['paths']['genpath']

# other params - whether to regress out global metrics and run PCA
preprocessing_params = cfg['preproc']
regress = 'Corrected' if preprocessing_params['regress'] else 'Raw'
run_pca = 'PCA' if preprocessing_params['pca'] else 'noPCA'
run_combat = 'Combat' if preprocessing_params['combat'] else 'noCombat'

# cortical parcellation
parc = cfg['data']['parcellation']

# k-fold CV params
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # n_split : 5

subject_data = pd.read_csv('./data/hcp_train.csv')
n_subs = 890
n_features = 153
num_of_models = 3
num_folds = 5


hcp_lasso = hcp_exp.loc[:, ['Feature','Lasso']]
hcp_gpr = hcp_exp.loc[:, ['Feature','Gaussian Process']]
hcp_gbm = hcp_exp.loc[:, ['Feature','Gradient Boosting Regressor']]

hcp_lasso_sort = hcp_lasso.sort_values(by='Lasso', ascending=False)
hcp_gpr_sort = hcp_gpr.sort_values(by='Gaussian Process', ascending=False)
hcp_gbm_sort = hcp_gbm.sort_values(by='Gradient Boosting Regressor', ascending=False)

hcp_lasso_feat_list = hcp_lasso_sort.Feature.to_list()
hcp_gpr_feat_list = hcp_gpr_sort.Feature.to_list()
hcp_gbm_feat_list = hcp_gbm_sort.Feature.to_list()


# Feature가 하나씩 늘어갈 때마다 어떤 식으로 Metric이 변화하는 지를 저장하는 list
uncorr_mae_list = []
uncorr_r2_list = []
corr_mae_list = []
corr_r2_list = []

# Feature가 하나씩 늘어갈 때마다 어떤 식으로 Metric이 변화하는 지를 저장하는 list
uncorr_mae_list = []
uncorr_r2_list = []
corr_mae_list = []
corr_r2_list = []

for feature_num in range(1, len(hcp_lasso_feat_list) + 1):
    if feature_num % 10 ==0:
        print(f"Using {feature_num} features") 
    # for문을 돌면서 Mean Absolute SHAP value가 가장 높은 순서대로 하나씩 추가해가며 
    # Model Type 변경시 이 부분 수정 
    subject_data_iter = subject_data.loc[:, hcp_lasso_feat_list[:feature_num]]
    
    # fold마다 생성되는 mae, r2 값 저장 
    unmae_fold_list = []
    unr2_fold_list = []
    mae_fold_list = []
    r2_fold_list = []
    
    for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Age)):
        #print('')
        #print('FOLD {:}:------------------------------------------------'.format(n+1))
        
        # Data 선언
        train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]
        # train_x, test_x = subject_data_iter.drop(['Age', 'Subject'], axis=1), subject_data_iter.drop(['Age', 'Subject'], axis=1)

        train_x = subject_data_iter.loc[train_idx]
        test_x = subject_data_iter.loc[test_idx]
        
        # Model 선언
        # Model Type 변경시 이 부분 수정 
        model = get_linear_model(preprocessing_params)
        
        # Fitting
        model.fit(train_x, train_y)
            
        # PREDICT 
        train_predictions = model.predict(train_x)
        test_predictions = model.predict(test_x)
        
        # 각 fold에서 Test sample들에 대한 Prediction
        uncorr_preds = test_predictions
        corr_preds = correct_age_predictions(train_predictions, train_y, test_predictions, test_y)
            
        # MAE, R2 계산 
        unmae_fold_list.append(mean_absolute_error(test_y, uncorr_preds))
        unr2_fold_list.append(r2_score(test_y, uncorr_preds))
        mae_fold_list.append(mean_absolute_error(test_y, corr_preds))
        r2_fold_list.append(r2_score(test_y, corr_preds))
        
    # 5 fold를 전부 다 돌았으면, 평균 MAE, R2 list에 저장 
    uncorr_mae_list.append(np.mean(unmae_fold_list))
    uncorr_r2_list.append(np.mean(unr2_fold_list))
    corr_mae_list.append(np.mean(mae_fold_list))
    corr_r2_list.append(np.mean(r2_fold_list))
    
    
hcp_lasso_metrics = {'uncorr_mae' : uncorr_mae_list, 'uncorr_r2': uncorr_r2_list, 
                    'corr_mae':corr_mae_list, 'corr_r2': corr_r2_list}
hcp_lasso_metrics = pd.DataFrame(hcp_lasso_metrics)


hcp_lasso_metrics.to_csv('./hcp_lasso_metrics.csv')
print("HCP Dataset Lasso Complete!")
print('===========================================================')





# Feature가 하나씩 늘어갈 때마다 어떤 식으로 Metric이 변화하는 지를 저장하는 list
uncorr_mae_list = []
uncorr_r2_list = []
corr_mae_list = []
corr_r2_list = []

for feature_num in range(1, len(hcp_gpr_feat_list) + 1):
    #if feature_num % 10 ==0:
    print(f"Using {feature_num} features") 
    # for문을 돌면서 Mean Absolute SHAP value가 가장 높은 순서대로 하나씩 추가해가며 
    # Model Type 변경시 이 부분 수정 
    subject_data_iter = subject_data.loc[:, hcp_gpr_feat_list[:feature_num]]
    
    # fold마다 생성되는 mae, r2 값 저장 
    unmae_fold_list = []
    unr2_fold_list = []
    mae_fold_list = []
    r2_fold_list = []
    
    for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Age)):
        
        # Data 선언
        train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]
        train_x = subject_data_iter.loc[train_idx]
        test_x = subject_data_iter.loc[test_idx]
        
        # Model 선언
        # Model Type 변경시 이 부분 수정 
        model = get_nonlinear_model(preprocessing_params)
        
        # Fitting
        model.fit(train_x, train_y)
            
        # PREDICT 
        train_predictions = model.predict(train_x)
        test_predictions = model.predict(test_x)
        
        # 각 fold에서 Test sample들에 대한 Prediction
        uncorr_preds = test_predictions
        corr_preds = correct_age_predictions(train_predictions, train_y, test_predictions, test_y)
            
        # MAE, R2 계산 
        unmae_fold_list.append(mean_absolute_error(test_y, uncorr_preds))
        unr2_fold_list.append(r2_score(test_y, uncorr_preds))
        mae_fold_list.append(mean_absolute_error(test_y, corr_preds))
        r2_fold_list.append(r2_score(test_y, corr_preds))
        
    # 5 fold를 전부 다 돌았으면, 평균 MAE, R2 list에 저장 
    uncorr_mae_list.append(np.mean(unmae_fold_list))
    uncorr_r2_list.append(np.mean(unr2_fold_list))
    corr_mae_list.append(np.mean(mae_fold_list))
    corr_r2_list.append(np.mean(r2_fold_list))
    
# Model Type 변경시 변수 명 수정 (4군데)
hcp_gpr_metrics = {'uncorr_mae' : uncorr_mae_list, 'uncorr_r2': uncorr_r2_list, 
                    'corr_mae':corr_mae_list, 'corr_r2': corr_r2_list}
hcp_gpr_metrics = pd.DataFrame(hcp_gpr_metrics)
hcp_gpr_metrics.to_csv('./hcp_gpr_metrics.csv')
print("HCP GPR Model Complete!")
print('===========================================================')


# Feature가 하나씩 늘어갈 때마다 어떤 식으로 Metric이 변화하는 지를 저장하는 list
uncorr_mae_list = []
uncorr_r2_list = []
corr_mae_list = []
corr_r2_list = []

for feature_num in range(1, len(hcp_gbm_feat_list) + 1):
    if feature_num % 10 ==0:
        print(f"Using {feature_num} features") 
    # for문을 돌면서 Mean Absolute SHAP value가 가장 높은 순서대로 하나씩 추가해가며 
    # Model Type 변경시 이 부분 수정 
    subject_data_iter = subject_data.loc[:, hcp_gbm_feat_list[:feature_num]]
    
    # fold마다 생성되는 mae, r2 값 저장 
    unmae_fold_list = []
    unr2_fold_list = []
    mae_fold_list = []
    r2_fold_list = []
    
    for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Age)):
        #print('')
        #print('FOLD {:}:------------------------------------------------'.format(n+1))
        
        # Data 선언
        train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]
        # train_x, test_x = subject_data_iter.drop(['Age', 'Subject'], axis=1), subject_data_iter.drop(['Age', 'Subject'], axis=1)

        train_x = subject_data_iter.loc[train_idx]
        test_x = subject_data_iter.loc[test_idx]
        
        # Model 선언
        # Model Type 변경시 이 부분 수정 
        model = get_ensemble_model(preprocessing_params)
        
        # Fitting
        model.fit(train_x, train_y)
            
        # PREDICT 
        train_predictions = model.predict(train_x)
        test_predictions = model.predict(test_x)
        
        # 각 fold에서 Test sample들에 대한 Prediction
        uncorr_preds = test_predictions
        corr_preds = correct_age_predictions(train_predictions, train_y, test_predictions, test_y)
            
        # MAE, R2 계산 
        unmae_fold_list.append(mean_absolute_error(test_y, uncorr_preds))
        unr2_fold_list.append(r2_score(test_y, uncorr_preds))
        mae_fold_list.append(mean_absolute_error(test_y, corr_preds))
        r2_fold_list.append(r2_score(test_y, corr_preds))
        
    # 5 fold를 전부 다 돌았으면, 평균 MAE, R2 list에 저장 
    uncorr_mae_list.append(np.mean(unmae_fold_list))
    uncorr_r2_list.append(np.mean(unr2_fold_list))
    corr_mae_list.append(np.mean(mae_fold_list))
    corr_r2_list.append(np.mean(r2_fold_list))
    
    
hcp_gbm_metrics = {'uncorr_mae' : uncorr_mae_list, 'uncorr_r2': uncorr_r2_list, 
                    'corr_mae':corr_mae_list, 'corr_r2': corr_r2_list}
hcp_gbm_metrics = pd.DataFrame(hcp_gbm_metrics)
hcp_gbm_metrics.to_csv('./hcp_gbm_metrics.csv')

print("HCP GPR Model Complete!")
print('===========================================================')
