"""
Unified Feature Importance with SHAP Explanations

Consolidates the following 9 scripts into a single parameterized script:
  - feature_importance_opt_cc_shap_gbm.py
  - feature_importance_opt_cc_shap_gpr.py
  - feature_importance_opt_cc_shap_lasso.py
  - feature_importance_opt_hcp_shap_gbm.py
  - feature_importance_opt_hcp_shap_gpr.py
  - feature_importance_opt_hcp_shap_lasso.py
  - feature_importance_opt_ixi_shap_gbm.py
  - feature_importance_opt_ixi_shap_gpr.py
  - feature_importance_opt_ixi_shap_lasso.py

Usage:
  python feature_importance_shap.py --dataset cc --model gbm
  python feature_importance_shap.py --dataset hcp --model gpr
  python feature_importance_shap.py --dataset ixi --model lasso
  python feature_importance_shap.py --dataset hcp --model lasso --n_splits 10 --random_state 1
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from functions.models import get_ensemble_model, get_linear_model, get_nonlinear_model
from functions.models import get_age_corrected_model_explanations, correct_age_predictions
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# Dataset-specific configurations
DATASET_CONFIG = {
    'hcp': {
        'train_file': 'hcp_train.csv',
        'n_subs': 890,
        'data_name': 'HCP',
        'excel_col_slice': (1, 4),  # df_sheet_3.iloc[:, 1:4]
    },
    'cc': {
        'train_file': 'cc_train.csv',
        'n_subs': 500,
        'data_name': 'CC',
        'excel_col_slice': (4, 7),  # df_sheet_3.iloc[:, 4:7]
    },
    'ixi': {
        'train_file': 'ixi_train.csv',
        'n_subs': 453,
        'data_name': 'IXI',
        'excel_col_slice': (7, None),  # df_sheet_3.iloc[:, 7:]
    },
}

# Model-specific configurations
MODEL_CONFIG = {
    'gbm': {
        'model_name': 'GBM',
        'feature_col': 'Gradient Boosting Regressor',
        'get_model': get_ensemble_model,
    },
    'gpr': {
        'model_name': 'GPR',
        'feature_col': 'Gaussian Process',
        'get_model': get_nonlinear_model,
    },
    'lasso': {
        'model_name': 'Lasso',
        'feature_col': 'Lasso',
        'get_model': get_linear_model,
    },
}

# Default optimal feature numbers per (dataset, model)
DEFAULT_OPT_FEATURES = {
    ('cc', 'gbm'): 54,
    ('cc', 'gpr'): 83,
    ('cc', 'lasso'): 64,
    ('hcp', 'gbm'): 49,
    ('hcp', 'gpr'): 47,
    ('hcp', 'lasso'): 32,
    ('ixi', 'gbm'): 52,
    ('ixi', 'gpr'): 38,
    ('ixi', 'lasso'): 46,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Feature importance analysis with SHAP explanations'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['hcp', 'cc', 'ixi'],
                        help='Dataset to use (hcp, cc, ixi)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['gbm', 'gpr', 'lasso'],
                        help='Model type (gbm, gpr, lasso)')
    parser.add_argument('--opt_features', type=int, default=None,
                        help='Number of optimal features to use (default: uses preset value)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV splits (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for CV (default: 42)')
    return parser.parse_args()


def load_shap_features(excel_path, dataset_cfg, model_cfg, opt_feature_num):
    """Load SHAP feature importance from Excel and select top-N features."""
    df_sheet_3 = pd.read_excel(excel_path, sheet_name='Sheet3')

    feature_name = df_sheet_3.iloc[:, 0]
    col_start, col_end = dataset_cfg['excel_col_slice']
    exp_data = df_sheet_3.iloc[:, col_start:col_end]
    exp_data = pd.concat([feature_name, exp_data], axis=1)
    header = exp_data.iloc[0]
    exp_data = exp_data[1:]
    exp_data.rename(columns=header, inplace=True)

    feature_col = model_cfg['feature_col']
    model_features = exp_data.loc[:, ['Feature', feature_col]]
    model_features_sort = model_features.sort_values(by=feature_col, ascending=False)
    feat_list = model_features_sort.Feature.to_list()

    selected_features = feat_list[:opt_feature_num]
    selected_features.append('Age')
    selected_features.append('Subject')

    return selected_features


def main():
    args = parse_args()

    dataset_key = args.dataset
    model_key = args.model
    ds_cfg = DATASET_CONFIG[dataset_key]
    md_cfg = MODEL_CONFIG[model_key]

    opt_feature_num = args.opt_features or DEFAULT_OPT_FEATURES[(dataset_key, model_key)]
    model_name = md_cfg['model_name']
    data_name = ds_cfg['data_name']

    #####################################################################################################
    # CONFIG
    #####################################################################################################
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    print('')
    print('---------------------------------------------------------')
    print('Configuration:')
    print(yaml.dump(cfg, default_flow_style=False, default_style=''))
    print(f'Dataset: {data_name}, Model: {model_name}, Opt Features: {opt_feature_num}')
    print(f'CV: {args.n_splits}-fold, random_state={args.random_state}')
    print('---------------------------------------------------------')
    print('')

    datapath = cfg['paths']['datapath']
    outpath = cfg['paths']['results']
    genpath = cfg['paths']['genpath']
    preprocessing_params = cfg['preproc']

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    #####################################################################################################
    # LOADING
    #####################################################################################################
    print('---------------------------------------------------------')
    print('loading data')
    print('---------------------------------------------------------')

    subject_data = pd.read_csv(datapath + ds_cfg['train_file'])

    selected_features = load_shap_features(
        './feat_imp/total.xlsx', ds_cfg, md_cfg, opt_feature_num
    )
    subject_data = subject_data.loc[:, selected_features]

    #####################################################################################################
    # K-FOLD
    #####################################################################################################
    n_subs = ds_cfg['n_subs']
    n_features = opt_feature_num
    num_of_models = 1
    num_folds = args.n_splits

    preds = np.zeros((n_subs, num_of_models))
    uncorr_preds = np.zeros((n_subs, num_of_models))
    fold = np.zeros((n_subs, 1))
    feature_explanations = np.zeros((num_of_models, n_subs, n_features))
    fold_predictions = np.zeros((num_of_models, n_subs, num_folds))
    fold_feature_explanations = np.zeros((num_of_models, n_subs, n_features, num_folds))

    for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Age)):
        print('')
        print('FOLD {:}:------------------------------------------------'.format(n+1))

        train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]
        train_x = subject_data.drop(['Age', 'Subject'], axis=1).loc[train_idx]
        test_x = subject_data.drop(['Age', 'Subject'], axis=1).loc[test_idx]

        fold[test_idx] = n+1
        print('')

        model = md_cfg['get_model'](preprocessing_params)

        # FIT
        print('fitting {:} model'.format(model_name))
        model.fit(train_x, train_y)

        # PREDICT
        train_predictions = model.predict(train_x)
        test_predictions = model.predict(test_x)

        # CORRECT FOR AGE EFFECT
        uncorr_preds[test_idx, 0] = test_predictions
        preds[test_idx, 0] = correct_age_predictions(train_predictions, train_y, test_predictions, test_y)

        fold_predictions[0, train_idx, n] = train_predictions - train_y
        fold_predictions[0, test_idx, n] = test_predictions - test_y

        # EXPLAIN
        print('calculating {:} model explanations for test data'.format(model_name))
        exp_features = round(np.shape(train_x)[1])
        test_model_explanations = np.zeros((np.shape(test_x)[0], np.shape(test_x)[1]))
        train_model_explanations = np.zeros((np.shape(train_x)[0], np.shape(train_x)[1]))

        for s in tqdm(np.arange(len(test_x))):
            test_model_explanations[s,:] = get_age_corrected_model_explanations(
                model, train_x, train_y, test_x.iloc[s, :].values.reshape(1,-1),
                age=test_y.iloc[s], num_features=exp_features)

        print('calculating {:} model explanations for train data'.format(model_name))
        num_train = len(train_x)
        for s in tqdm(np.arange(num_train)):
            train_model_explanations[s,:] = get_age_corrected_model_explanations(
                model, train_x.iloc[np.arange(num_train)!=s,:],
                train_y[np.arange(num_train)!=s],
                train_x.iloc[s,:].values.reshape(1,-1),
                age=train_y.iloc[s], num_features=exp_features)

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

    preds = pd.DataFrame(
        np.hstack((preds, uncorr_preds)),
        columns=[f'{model_name}_preds', f'{model_name}_uncorr_preds']
    )
    fold = pd.DataFrame(fold.astype(int), columns=['fold'])
    predictions = pd.concat((subject_data, fold, preds), axis=1)

    # Use data_name prefix for HCP/IXI (matching original behavior)
    if dataset_key in ('hcp', 'ixi'):
        file_prefix = f'{data_name}{model_name}'
        file_sep_prefix = f'{data_name}-{model_name}'
    else:
        file_prefix = model_name
        file_sep_prefix = model_name

    print(f'model predictions: {outpath}revision_shap/{file_prefix}_predictions.csv')
    print('')
    predictions.to_csv(f'{outpath}revision_shap/{file_prefix}_predictions.csv', index=False)

    # accuracies
    n_fold = len(np.unique(predictions.fold))
    models_list = [model_name]

    fold_mae = np.zeros((n_fold, len(models_list)*2))
    fold_r2 = np.zeros((n_fold, len(models_list)*2))

    for n, f in enumerate(np.unique(predictions.fold)):
        for m, ml in enumerate(models_list):
            fold_mae[n, m] = mean_absolute_error(
                predictions.Age[predictions.fold==f],
                predictions[ml+'_preds'][predictions.fold==f])
            fold_mae[n, m+num_of_models] = mean_absolute_error(
                predictions.Age[predictions.fold==f],
                predictions[ml+'_uncorr_preds'][predictions.fold==f])
            fold_r2[n, m] = r2_score(
                predictions.Age[predictions.fold==f],
                predictions[ml+'_preds'][predictions.fold==f])
            fold_r2[n, m+num_of_models] = r2_score(
                predictions.Age[predictions.fold==f],
                predictions[ml+'_uncorr_preds'][predictions.fold==f])

    fold_mae = pd.DataFrame(fold_mae, columns=[model_name, f'{model_name}_uncorr'])
    fold_mae.insert(0, 'fold', np.unique(predictions.fold))
    fold_r2 = pd.DataFrame(fold_r2, columns=[model_name, f'{model_name}_uncorr'])
    fold_r2.insert(0, 'fold', np.unique(predictions.fold))

    fold_mae.to_csv(f'{outpath}revision_shap/{file_prefix}-MAE.csv', index=False)
    fold_r2.to_csv(f'{outpath}revision_shap/{file_prefix}-R2.csv', index=False)
    print('')

    # explanations
    for m, mn in enumerate([model_name]):
        exp = pd.DataFrame(feature_explanations[m])
        fold_df = pd.DataFrame(fold.astype(int), columns=['fold'])
        feat_exp = pd.concat((subject_data, fold_df, exp), axis=1)
        print(f'model explanations: {outpath}revision_shap/{file_sep_prefix}-model-feature-explanations.csv')
        print('')
        feat_exp.to_csv(f'{outpath}revision_shap/{file_sep_prefix}-model-feature-explanations.csv', index=False)

        print(f'model explanations for cross-validation: {genpath}revision_shap/{file_sep_prefix}-model-all-fold-feature-explanations.npy')
        np.save(f'{genpath}revision_shap/{file_sep_prefix}-model-all-fold-feature-explanations.npy',
                fold_feature_explanations[m,:,:,:])
        print(f'model predictions for cross-validation: {genpath}revision_shap/{file_sep_prefix}-model-all-fold-delta.npy')
        np.save(f'{genpath}revision_shap/{file_sep_prefix}-model-all-fold-delta.npy',
                fold_predictions[m,:,:])


if __name__ == '__main__':
    main()
