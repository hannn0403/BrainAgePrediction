"""
Unified Feature Importance Optimization

Finds the optimal number of SHAP-ranked features by iterating from 1 to 153 features
and evaluating model performance with cross-validation.

Consolidates the following 3 scripts into a single parameterized script:
  - feature_importance_opt_hcp.py
  - feature_importance_opt_ixi.py
  - feature_importance_opt_cc.py (if exists)

Usage:
  # Run a specific dataset + model combination
  python feature_importance_optimize.py --dataset hcp --model lasso
  python feature_importance_optimize.py --dataset ixi --model gpr
  python feature_importance_optimize.py --dataset hcp --model gbm

  # Run all 3 models for a dataset (original behavior)
  python feature_importance_optimize.py --dataset hcp --model all
  python feature_importance_optimize.py --dataset ixi --model all

  # Custom output directory
  python feature_importance_optimize.py --dataset hcp --model lasso --output_dir ./results/
"""

import argparse
import numpy as np
import pandas as pd
import yaml
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from functions.models import get_ensemble_model, get_linear_model, get_nonlinear_model
from functions.models import correct_age_predictions

warnings.filterwarnings('ignore')


# Dataset configurations
DATASET_CONFIG = {
    'hcp': {
        'train_file': 'hcp_train.csv',
        'train_file_alt': './data/hcp_train.csv',  # original scripts used this path
        'n_subs': 890,
        'data_name': 'HCP',
        'excel_col_slice': (1, 4),
    },
    'cc': {
        'train_file': 'cc_train.csv',
        'train_file_alt': './data/cc_train.csv',
        'n_subs': 500,
        'data_name': 'CC',
        'excel_col_slice': (4, 7),
    },
    'ixi': {
        'train_file': 'ixi_train.csv',
        'train_file_alt': './data/ixi_train.csv',
        'n_subs': 453,
        'data_name': 'IXI',
        'excel_col_slice': (7, None),
    },
}

# Model configurations
MODEL_CONFIG = {
    'lasso': {
        'model_name': 'Lasso',
        'feature_col': 'Lasso',
        'get_model': get_linear_model,
        'display_name': 'Lasso',
    },
    'gpr': {
        'model_name': 'GPR',
        'feature_col': 'Gaussian Process',
        'get_model': get_nonlinear_model,
        'display_name': 'Gaussian Process',
    },
    'gbm': {
        'model_name': 'GBM',
        'feature_col': 'Gradient Boosting Regressor',
        'get_model': get_ensemble_model,
        'display_name': 'Gradient Boosting Machine',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find optimal number of SHAP-ranked features for brain age prediction'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['hcp', 'cc', 'ixi'],
                        help='Dataset to use (hcp, cc, ixi)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lasso', 'gpr', 'gbm', 'all'],
                        help='Model type (lasso, gpr, gbm, or all)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to save output metrics CSV (default: ./)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV splits (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for CV (default: 42)')
    return parser.parse_args()


def load_shap_feature_list(excel_path, dataset_cfg, model_cfg):
    """Load SHAP values and return sorted feature list for a given model."""
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

    return model_features_sort.Feature.to_list()


def run_optimization(subject_data, feat_list, model_cfg, n_subs, skf, preprocessing_params):
    """
    Iterate from 1 to len(feat_list) features, evaluating model performance.
    Returns a DataFrame with uncorr_mae, uncorr_r2, corr_mae, corr_r2 per feature count.
    """
    uncorr_mae_list = []
    uncorr_r2_list = []
    corr_mae_list = []
    corr_r2_list = []

    for feature_num in range(1, len(feat_list) + 1):
        if feature_num % 10 == 0:
            print(f"Using {feature_num} features")

        subject_data_iter = subject_data.loc[:, feat_list[:feature_num]]

        unmae_fold_list = []
        unr2_fold_list = []
        mae_fold_list = []
        r2_fold_list = []

        for n, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_subs), subject_data.Age)):
            train_y, test_y = subject_data.Age[train_idx], subject_data.Age[test_idx]
            train_x = subject_data_iter.loc[train_idx]
            test_x = subject_data_iter.loc[test_idx]

            model = model_cfg['get_model'](preprocessing_params)
            model.fit(train_x, train_y)

            train_predictions = model.predict(train_x)
            test_predictions = model.predict(test_x)

            uncorr_preds = test_predictions
            corr_preds = correct_age_predictions(train_predictions, train_y, test_predictions, test_y)

            unmae_fold_list.append(mean_absolute_error(test_y, uncorr_preds))
            unr2_fold_list.append(r2_score(test_y, uncorr_preds))
            mae_fold_list.append(mean_absolute_error(test_y, corr_preds))
            r2_fold_list.append(r2_score(test_y, corr_preds))

        uncorr_mae_list.append(np.mean(unmae_fold_list))
        uncorr_r2_list.append(np.mean(unr2_fold_list))
        corr_mae_list.append(np.mean(mae_fold_list))
        corr_r2_list.append(np.mean(r2_fold_list))

    metrics = pd.DataFrame({
        'uncorr_mae': uncorr_mae_list,
        'uncorr_r2': uncorr_r2_list,
        'corr_mae': corr_mae_list,
        'corr_r2': corr_r2_list,
    })
    return metrics


def main():
    args = parse_args()

    dataset_key = args.dataset
    ds_cfg = DATASET_CONFIG[dataset_key]
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
    print('---------------------------------------------------------')
    print('')

    datapath = cfg['paths']['datapath']
    preprocessing_params = cfg['preproc']

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    #####################################################################################################
    # LOADING
    #####################################################################################################
    # Try config datapath first, fall back to alternative path
    try:
        subject_data = pd.read_csv(datapath + ds_cfg['train_file'])
    except FileNotFoundError:
        subject_data = pd.read_csv(ds_cfg['train_file_alt'])

    n_subs = ds_cfg['n_subs']

    # Determine which models to run
    if args.model == 'all':
        model_keys = ['lasso', 'gpr', 'gbm']
    else:
        model_keys = [args.model]

    #####################################################################################################
    # OPTIMIZATION
    #####################################################################################################
    excel_path = './feat_imp/total.xlsx'

    for model_key in model_keys:
        md_cfg = MODEL_CONFIG[model_key]
        model_name_lower = model_key
        model_display = md_cfg['display_name']

        print(f'\n{"="*60}')
        print(f'Optimizing {data_name} - {model_display}')
        print(f'{"="*60}')

        feat_list = load_shap_feature_list(excel_path, ds_cfg, md_cfg)

        metrics = run_optimization(
            subject_data, feat_list, md_cfg, n_subs, skf, preprocessing_params
        )

        output_file = f'{args.output_dir}{dataset_key}_{model_name_lower}_metrics.csv'
        metrics.to_csv(output_file, index=False)
        print(f'\n{data_name} Dataset {model_display} Complete!')
        print(f'Saved to: {output_file}')

        # Report optimal feature number (minimum corrected MAE)
        opt_idx = metrics['corr_mae'].idxmin()
        print(f'Optimal features: {opt_idx + 1} (corr_MAE={metrics["corr_mae"][opt_idx]:.4f})')
        print('===========================================================')


if __name__ == '__main__':
    main()
