import numpy as np
import pandas as pd
 
import os, glob
from regex import P
import yaml
from tqdm import tqdm
 
from functions.surfaces import load_surf_data, parcellateSurface
from functions.models import get_ensemble_model, get_linear_model, get_nonlinear_model
from functions.models import get_model_explanations, get_age_corrected_model_explanations, correct_age_predictions
from functions.misc import pre_process_metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
 
def main():
    #####################################################################################################
    # CONFIG
    # load configuration file and set parameters accordingly
    #####################################################################################################
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
    outpath = cfg['paths']['results']
    genpath = cfg['paths']['genpath']
 
    # other params - whether to regress out global metrics and run PCA
    preprocessing_params = cfg['preproc']
   
    # k-fold CV params
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # n_split : 5
 
    #####################################################################################################
    # LOADING
    #####################################################################################################
 
    print('---------------------------------------------------------')
    print('loading data')
    print('---------------------------------------------------------')
 
    # load age, sex, site data
    subject_data = pd.read_csv(datapath + 'ixi_train.csv')
   
    df_sheet_3 = pd.read_excel('./feat_imp/total.xlsx', sheet_name='Sheet3')
    # CamCan SHAP Value
    # 데이터 셋을 변경하는 경우 이 부분 변경 필요
    data_name = 'IXI'
    feature_name = df_sheet_3.iloc[:, 0]
    ixi_exp = df_sheet_3.iloc[:, 7:]
    ixi_exp = pd.concat([feature_name, ixi_exp], axis=1)
    ixi_header = ixi_exp.iloc[0]
    ixi_exp = ixi_exp[1:]
    ixi_exp.rename(columns=ixi_header, inplace=True)
 
    # 모델을 변경하는 경우 이 부분 변경 필요
    opt_gpr_feature_num = 38
    ixi_gpr = ixi_exp.loc[:, ['Feature','Gaussian Process']]
    ixi_gpr_sort = ixi_gpr.sort_values(by='Gaussian Process', ascending=False)
    
    
    ixi_gpr_feat_list = ixi_gpr_sort.Feature.to_list()
    ixi_gpr_features = ixi_gpr_feat_list[:opt_gpr_feature_num]
    ixi_gpr_features.append('Age')
    ixi_gpr_features.append('Subject')
 
    subject_data = subject_data.loc[:, ixi_gpr_features]
    #####################################################################################################
    # K-FOLD
    #####################################################################################################
    # 데이터 셋을 변경하는 경우 이 부분 변경 필요
    n_subs = 453
    n_features = opt_gpr_feature_num
    num_of_models = 1
    num_folds = 5
 
    # space for predictions and explanations
    preds = np.zeros((n_subs, num_of_models))
    uncorr_preds = np.zeros((n_subs, num_of_models))
    fold = np.zeros((n_subs, 1))
    feature_explanations = np.zeros((num_of_models, n_subs, n_features))
 
    fold_predictions = np.zeros((num_of_models, n_subs, num_folds)) #num models x num subs x num folds
    fold_feature_explanations = np.zeros((num_of_models, n_subs, n_features, num_folds)) # num models x num subs x num features x num folds
 
    # cross-validation - stratified by site
    for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Age)):
        print('')
        print('FOLD {:}:------------------------------------------------'.format(n+1))
 
        # age data for train and test sets
        train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]
        train_x, test_x = subject_data.drop(['Age','Subject'], axis = 1), subject_data.drop(['Age','Subject'], axis = 1)
        train_x = train_x.loc[train_idx]
        test_x = test_x.loc[test_idx]
 
        # run models in each fold
        fold[test_idx] = n+1
        print('')     
        
        # 모델을 변경하는 경우에 이 부분 변경 필요
        model = get_nonlinear_model(preprocessing_params)
        model_name = 'GPR'
   
 
        # FIT
        print('fitting {:} model'.format(model_name))
        model.fit(train_x, train_y)
        # PREDICT
        train_predictions = model.predict(train_x)
        test_predictions = model.predict(test_x)
        # CORRECT FOR AGE EFFECT
        uncorr_preds[test_idx, 0] = test_predictions
        preds[test_idx, 0] = correct_age_predictions(train_predictions, train_y, test_predictions, test_y)
        # collate brain age delta for later models
        fold_predictions[0, train_idx, n] =  train_predictions - train_y
        fold_predictions[0, test_idx, n] =  test_predictions - test_y
        # EXPLAIN
        print('calculating {:} model explanations for test data'.format(model_name))
        exp_features = round(np.shape(train_x)[1])
        test_model_explanations = np.zeros((np.shape(test_x)[0], np.shape(test_x)[1]))
        train_model_explanations = np.zeros((np.shape(train_x)[0], np.shape(train_x)[1]))
        for s in tqdm(np.arange(len(test_x))):
            test_model_explanations[s,:] = get_age_corrected_model_explanations(model, train_x, train_y, test_x.iloc[s, :].values.reshape(1,-1),
                                                                                age=test_y.iloc[s], num_features=exp_features)
            # train explanations - for training set examples (exclude self)
        print('calculating {:} model explanations for train data'.format(model_name))
        num_train = len(train_x)
        for s in tqdm(np.arange(num_train)):
            train_model_explanations[s,:] = get_age_corrected_model_explanations(model, train_x.iloc[np.arange(num_train)!=s,:], train_y[np.arange(num_train)!=s], train_x.iloc[s,:].values.reshape(1,-1),
                                                                                age=train_y.iloc[s], num_features=exp_features)
        # collate
        print(test_model_explanations.shape)
 
        feature_explanations[0, test_idx, :] = test_model_explanations
        fold_feature_explanations[0, test_idx, :, n] = test_model_explanations
        fold_feature_explanations[0, train_idx, :, n] = train_model_explanations
 
 
 
    #####################################################################################################
    # RESULTS
    #####################################################################################################
    print('---------------------------------------------------------')
    print('compiling results')
    print('---------------------------------------------------------')
    # collate data
    preds = pd.DataFrame(np.hstack((preds, uncorr_preds)), columns = ['GPR_preds', 'GPR_uncorr_preds'])
    fold = pd.DataFrame(fold.astype(int), columns=['fold'])
    predictions = pd.concat((subject_data, fold, preds), axis=1)
 
    # saving
    print('model predictions: {:}revision_shap/model_predictions-.csv'.format(outpath))
    print('')
    predictions.to_csv('{:}revision_shap/{:}{:}_predictions.csv'.format(outpath, data_name, model_name), index=False)
 
    # accuracies
    n_fold = len(np.unique(predictions.fold))
    # 모델 변경시 이 부분 변경 필요
    models = ['GPR']
 
    fold_mae = np.zeros((n_fold, len(models)*2))
    fold_r2 = np.zeros((n_fold, len(models)*2))
 
    for n, f in enumerate(np.unique(predictions.fold)):
        for m, model in enumerate(models):
            fold_mae[n, m] = mean_absolute_error(predictions.Age[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])
            fold_mae[n, m+num_of_models] = mean_absolute_error(predictions.Age[predictions.fold==f], predictions[model+'_uncorr_preds'][predictions.fold==f])
 
            fold_r2[n, m] = r2_score(predictions.Age[predictions.fold==f], predictions[model+'_preds'][predictions.fold==f])
            fold_r2[n, m+num_of_models] = r2_score(predictions.Age[predictions.fold==f], predictions[model+'_uncorr_preds'][predictions.fold==f])
 
    fold_mae = pd.DataFrame(fold_mae, columns=['GPR', 'GPR_uncorr'])
    fold_mae.insert(0, 'fold', np.unique(predictions.fold))
 
    fold_r2 = pd.DataFrame(fold_r2, columns=['GPR', 'GPR_uncorr'])
    fold_r2.insert(0, 'fold', np.unique(predictions.fold))
 
    # saving
    # print('model accuracy (MAE): {:}MAE-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
    fold_mae.to_csv('{:}revision_shap/{:}{:}-MAE.csv'.format(outpath, data_name, model_name), index=False)
    # print('model accuracy (R2): {:}R2-{:}-{:}-{:}-{:}.csv'.format(outpath, run_combat, regress, run_pca, parc))
    fold_r2.to_csv('{:}revision_shap/{:}{:}-R2.csv'.format(outpath, data_name, model_name), index=False)
    print('')
 
    # explanations
    # 모델 변경시 이 부분 변경필요 
    for m, model_name in enumerate(['GPR']):
        exp = pd.DataFrame(feature_explanations[m])
        fold = pd.DataFrame(fold.astype(int), columns=['fold'])
        feat_exp = pd.concat((subject_data, fold, exp), axis=1)
        print('model explanations: {:}revision_shap/{:}{:}-model-feature-explanations.csv'.format(outpath, data_name, model_name))
        print('')
        feat_exp.to_csv('{:}revision_shap/{:}-{:}-model-feature-explanations.csv'.format(outpath, data_name, model_name), index=False)
 
        # save for later CV models
        print('model explanations for cross-validation: {:}revision_shap/{:}-{:}-model-all-fold-feature-explanations.npy'.format(genpath, data_name, model_name))
        np.save('{:}revision_shap/{:}-{:}-model-all-fold-feature-explanations.npy'.format(genpath,data_name, model_name), fold_feature_explanations[m,:,:,:])
        print('model predictions for cross-validation: {:}revision_shap/{:}-{:}-model-all-fold-delta.npy'.format(genpath,data_name, model_name))
        np.save('{:}revision_shap/{:}-{:}-model-all-fold-delta.npy'.format(genpath, data_name, model_name), fold_predictions[m,:,:])
 
       
 
if __name__ == '__main__':
    main()

