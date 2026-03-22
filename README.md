# Brain Age Prediction: A Comparison between Machine Learning Models Using Brain Morphometric Data


**Juhyuk Han, Seo Yeong Kim, Junhyeok Lee and Won Hee Lee**


Sensors 2022, 22(20), 8077; https://doi.org/10.3390/s22208077

## ❗️ Project Summary

---

1. **진행기간:** 2021.07 ~ 2022.10
2. **역할:** 주저자, 데이터 전처리, 모델 파이프라인 설계, 시각화, 통계 검정
3. **기술스택:** **`Python`**, **`Pycaret`**, **`Scikit-learn`**, **`SHAP`**
4. **결과 및 성과:**
    - MDPI Sensors 논문 게재 (인용수 38회)
    - 경희대학교 학술상
    - 논문 게재 [**[📄]**](https://www.mdpi.com/1424-8220/22/20/8077)
5. **주요내용:** 뇌 질환과 노화의 진행을 측정하는 biomarker 중 하나인 뇌 연령은 일반적으로 생체학적 나이와 MRI 데이터에 머신러닝 모델을 적용하여 예측된 연령 간 차이로 추정합니다.
정상인으로 구성된 MRI Datasets를 전처리하여  뇌의 형태학적 특징을 추출하였고, 뇌연령 예측 모델의 학습 데이터로 활용하였습니다.
27가지 머신러닝 모델의 뇌연령 예측 성능을 비교하였고, SHAP을 통해  모델이 뇌연령 예측 시 주요하게 보는 해부학적 위치와 실제 임상적 지식을 비교하였습니다.

---


### Abstract
Brain structural morphology varies over the aging trajectory, and the prediction of a person's age using brain morphological features can help the detection of an abnormal aging process. Neuroimaging-based brain age is widely used to quantify an individual's brain health as deviation from a normative brain aging trajectory. Machine learning approaches are expanding the potential for accurate brain age prediction but are challenging due to the great variety of machine learning algorithms. Here, we aimed to compare the performance of the machine learning models used to estimate brain age using brain morphological measures derived from structural magnetic resonance imaging scans. We evaluated 27 machine learning models, applied to three independent datasets from the Human Connectome Project (HCP, n = 1113, age range 22–37), the Cambridge Centre for Ageing and Neuroscience (Cam-CAN, n = 601, age range 18–88), and the Information eXtraction from Images (IXI, n = 567, age range 19–86). Performance was assessed within each sample using cross-validation and an unseen test set. The models achieved mean absolute errors of 2.75–3.12, 7.08–10.50, and 8.04–9.86 years, as well as Pearson's correlation coefficients of 0.11–0.42, 0.64–0.85, and 0.63–0.79 between predicted brain age and chronological age for the HCP, Cam-CAN, and IXI samples, respectively. We found a substantial difference in performance between models trained on the same data type, indicating that the choice of model yields considerable variation in brain-predicted age. Furthermore, in three datasets, regularized linear regression algorithms achieved similar performance to nonlinear and ensemble algorithms. Our results suggest that regularized linear algorithms are as effective as nonlinear and ensemble algorithms for brain age prediction, while significantly reducing computational costs. Our findings can serve as a starting point and quantitative reference for future efforts at improving brain age prediction using machine learning models applied to brain morphometric data.


### Supplementary Materials

The following supporting information can be downloaded at: https://www.mdpi.com/article/10.3390/s22208077/s1

---
# BrainAgePrediction

This repository provides a complete pipeline for predicting brain age using morphometric features extracted from MRI scans, comparing multiple machine learning algorithms across three independent datasets (HCP, Cam-CAN, IXI).

## Prerequisites

- Python 3.7 or higher
- Install core libraries:
  ```bash
  pip install numpy pandas scikit-learn pycaret shap tqdm pyyaml regex matplotlib seaborn notebook
  ```

## Requirements

All Python dependencies are listed in `requirements.txt`. To install them at once:

```bash
pip install -r requirements.txt
```

> **Tip:** It is recommended to use a virtual environment (e.g., `venv` or `conda`) to avoid dependency conflicts.

## Configuration

Model training scripts read their settings from **`SHAP/config.yaml`**. The file has three top-level sections:

| Section    | Key           | Description                                                                 |
|------------|---------------|-----------------------------------------------------------------------------|
| `paths`    | `datapath`    | Root directory containing input data files                                  |
| `paths`    | `results`     | Directory where predictions and metrics will be saved                       |
| `paths`    | `genpath`     | Directory for generated/intermediate data                                   |
| `data`     | `parcellation`| Dataset name (`CAMCAN`, `HCP`, or `IXI`)                                   |
| `preproc`  | `combat`      | Whether to apply ComBat harmonization (`yes` / `no`)                        |
| `preproc`  | `regress`     | Whether to apply confound regression (`yes` / `no`)                         |
| `preproc`  | `pca`         | Whether to apply PCA dimensionality reduction (`yes` / `no`)                |
| `preproc`  | `pca_comps`   | Number of PCA components (integer) or variance ratio (float between 0 and 1)|

> **Important:** The default `paths` values point to a Windows machine. You **must** update `datapath`, `results`, and `genpath` to match your local environment before running any training script.

## Datasets

Three independent neuroimaging datasets are used in this study:

| Dataset | Full Name | Subjects (n) | Age Range | Source |
|---------|-----------|:------------:|:---------:|--------|
| **HCP** | Human Connectome Project | 1 113 | 22--37 | [humanconnectome.org](https://www.humanconnectome.org/) |
| **Cam-CAN** | Cambridge Centre for Ageing and Neuroscience | 601 | 18--88 | [cam-can.org](https://www.cam-can.org/) |
| **IXI** | Information eXtraction from Images | 567 | 19--86 | [brain-development.org](https://brain-development.org/ixi-dataset/) |

Pre-split CSV files are provided in the `dataset/` folder:

| File | Description |
|------|-------------|
| `hcp_train.csv` / `hcp_test.csv` | HCP training and test splits |
| `cc_train.csv` / `cc_test.csv` | Cam-CAN training and test splits |
| `ixi_train.csv` / `ixi_test.csv` | IXI training and test splits |

Each CSV contains 153 cortical morphometric features (parcellated brain regions) along with a target age column.

## Repository Structure

```
BrainAgePrediction/
├── SHAP/                           # Model training, evaluation, and SHAP analysis scripts
│   ├── config.yaml                 # Paths and preprocessing/model hyperparameters
│   ├── run_brain_age_models_hcp.py # Train & evaluate on HCP dataset
│   ├── run_brain_age_models_camcan.py # Train & evaluate on Cam-CAN dataset
│   ├── run_brain_age_models_ixi.py    # Train & evaluate on IXI dataset
│   ├── feature_importance_shap.py  # Unified SHAP explanation script (dataset + model args)
│   ├── feature_importance_optimize.py # Unified optimal feature count search script
│   ├── results/                    # SHAP values and feature explanations
│   └── shapy value 구할때...txt       # Notes on modifying dataset paths for SHAP
├── dataset/                        # Pre-split CSV files for each dataset
│   ├── hcp_train.csv / hcp_test.csv
│   ├── cc_train.csv / cc_test.csv
│   └── ixi_train.csv / ixi_test.csv
├── models/                         # Saved trained model artifacts
│   ├── hcp/best_model/             # Best model for HCP (e.g., Ridge.pkl)
│   ├── cc/best_model/              # Best model for Cam-CAN
│   └── ixi/best_model/             # Best model for IXI
├── preprocessing/                  # Data processing notebooks and folders
│   ├── CAMCAN_153/                 # Parcellated & processed Cam-CAN data
│   ├── HCP_153/                    # Parcellated & processed HCP data
│   ├── IXI_153/                    # Parcellated & processed IXI data
│   ├── Age_range.ipynb             # Define age ranges & overview
│   ├── Data_Column_check.ipynb     # Verify dataset columns
│   └── Data_Feature_select_and_Splitting.ipynb # Feature selection & CV splits
├── results/                        # Aggregated test predictions & metrics
│   ├── hcp_test_sg_model.csv
│   ├── cc_test_sg_model.csv
│   ├── ixi_test_sg_model.csv
│   ├── integrated_test_sg.csv
│   ├── Integrated_Score_grid.ipynb # Score grid aggregation notebook
│   ├── Additional_metrics.ipynb    # Additional evaluation metrics notebook
│   └── score_grid/                 # Grid search performance results
├── Visualization/                  # Plotting scripts & saved figures
│   ├── Predicted Brain Age/        # Predicted vs. chronological plots
│   ├── Feature Importance/         # SHAP summary & importance plots
│   ├── bias_correction_delta_plot_*/ # Bias correction visualizations per dataset
│   ├── brain_gap_violin_plot/      # Brain-age gap violin plots
│   ├── violin_plot/                # Distribution comparisons
│   ├── brainage_viz_enigmatoolbox.ipynb  # Brain visualization with ENIGMA toolbox
│   ├── ml_alg_comparing_and_viz.ipynb   # ML algorithm comparison plots
│   └── predicted_age_viz_and_violin_plot.ipynb # Predicted age & violin plots
├── brain_age_prediction_training.ipynb # End-to-end training & evaluation notebook
├── brain_age_prediction_tuning.ipynb   # Hyperparameter tuning & analysis notebook
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Data Preparation

1. Ensure the `dataset/` folder contains the six CSV files (train/test splits for each dataset).
2. No additional downloads required.

## Preprocessing

Open and execute the notebooks in `preprocessing/` to:
- Inspect and clean dataset columns
- Define age ranges
- Select relevant morphometric features (153 cortical regions)
- Split data for cross-validation

Processed data folders (`*_153/`) will be created automatically.

## Models

The pipeline evaluates **27 machine learning models** organized into three categories:

| Category | Examples | Description |
|----------|----------|-------------|
| **Linear** | Ridge, Lasso, ElasticNet, Bayesian Ridge, etc. | Regularized linear regression models. Fast to train and often competitive with more complex approaches. |
| **Nonlinear** | Gaussian Process Regression (GPR), SVR, KNN, etc. | Capture nonlinear relationships between morphometric features and age. |
| **Ensemble** | Gradient Boosting (GBM), Random Forest, AdaBoost, etc. | Combine multiple base learners for improved prediction accuracy. |

Key aspects of the modeling pipeline:

- **5-fold stratified cross-validation:** Folds are stratified by age group to ensure each fold has a representative age distribution.
- **Age bias correction:** A linear correction is applied to predicted brain ages to mitigate the well-known regression-to-the-mean bias, where younger subjects' ages tend to be over-predicted and older subjects' ages under-predicted.
- **Model persistence:** Best-performing models are saved as `.pkl` files in the `models/` directory (organized by dataset).

## Model Training & Evaluation

Adjust parameters in `SHAP/config.yaml`, then train and evaluate models:

```bash
# Example: HCP dataset
python SHAP/run_brain_age_models_hcp.py
# Cam-CAN dataset
python SHAP/run_brain_age_models_camcan.py
# IXI dataset
python SHAP/run_brain_age_models_ixi.py
```

Each script performs 5-fold stratified CV by age group, fits linear, nonlinear, and ensemble models, applies age-correction, and outputs:
- Predictions & deltas (`model_predictions-*.csv` in `SHAP/` output path)
- MAE & R² scores per fold
- SHAP-based feature importance (if enabled)

## Validation

### Evaluation Metrics

Model performance is assessed using three complementary metrics:

| Metric | Description |
|--------|-------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and chronological age (in years). Lower is better. |
| **R²** (Coefficient of Determination) | Proportion of variance in chronological age explained by the model. Ranges from 0 to 1; higher is better. |
| **Pearson correlation (r)** | Linear correlation between predicted and chronological age. Higher values indicate stronger agreement. |

### Data Integrity Checks

To verify that the data pipeline is set up correctly before training, run:

```bash
python validate_pipeline.py
```

This script checks:
- CSV files in `dataset/` are present and well-formed
- Feature columns match the expected 153 morphometric regions
- Train/test splits have no overlapping subjects
- Age distributions are within expected ranges for each dataset

## Feature Importance & SHAP Analysis

Two unified scripts replace the previous per-dataset/per-model scripts. Both must be run from inside the `SHAP/` directory.

### 1. Optimal Feature Count Search (`feature_importance_optimize.py`)

Iterates from 1 to 153 SHAP-ranked features to find the optimal number that minimizes prediction error.

```bash
cd SHAP/

# Run a single dataset + model
python feature_importance_optimize.py --dataset hcp --model lasso
python feature_importance_optimize.py --dataset ixi --model gpr
python feature_importance_optimize.py --dataset cc --model gbm

# Run all 3 models for a dataset at once
python feature_importance_optimize.py --dataset hcp --model all
python feature_importance_optimize.py --dataset ixi --model all
```

| Argument | Required | Options | Default | Description |
|----------|----------|---------|---------|-------------|
| `--dataset` | Yes | `hcp`, `cc`, `ixi` | — | Dataset to evaluate |
| `--model` | Yes | `lasso`, `gpr`, `gbm`, `all` | — | Model type (`all` runs all three sequentially) |
| `--output_dir` | No | any path | `./` | Directory for output metrics CSV |
| `--n_splits` | No | integer | `5` | Number of CV folds |
| `--random_state` | No | integer | `42` | Random seed for CV |

Output: `{dataset}_{model}_metrics.csv` with per-feature-count MAE and R² values.

### 2. SHAP Explanations with Optimal Features (`feature_importance_shap.py`)

Trains a single model using the pre-determined optimal feature count, then computes SHAP-based feature explanations for every sample.

```bash
cd SHAP/

# Cam-CAN + GBM (54 optimal features, 5-fold CV)
python feature_importance_shap.py --dataset cc --model gbm

# HCP + GPR (47 optimal features)
python feature_importance_shap.py --dataset hcp --model gpr

# IXI + Lasso with custom CV settings
python feature_importance_shap.py --dataset ixi --model lasso --n_splits 10 --random_state 1

# Override the default optimal feature count
python feature_importance_shap.py --dataset hcp --model gbm --opt_features 60
```

| Argument | Required | Options | Default | Description |
|----------|----------|---------|---------|-------------|
| `--dataset` | Yes | `hcp`, `cc`, `ixi` | — | Dataset to use |
| `--model` | Yes | `gbm`, `gpr`, `lasso` | — | Model type |
| `--opt_features` | No | integer | preset* | Number of top SHAP features to use |
| `--n_splits` | No | integer | `5` | Number of CV folds |
| `--random_state` | No | integer | `42` | Random seed for CV |

\*Default optimal feature counts (from optimization step):

| Dataset | GBM | GPR | Lasso |
|---------|-----|-----|-------|
| HCP | 49 | 47 | 32 |
| Cam-CAN | 54 | 83 | 64 |
| IXI | 52 | 38 | 46 |

Output: predictions CSV, MAE/R² per fold, SHAP feature explanations (`.csv` and `.npy`).

## Notebook Workflow

For an interactive demonstration, run:

```bash
jupyter notebook brain_age_prediction_training.ipynb
```

This notebook walks through data loading, preprocessing, model fitting, and evaluation with inline visualizations.

## Visualization

Generate or recreate figures by running notebooks or scripts in `Visualization/`:

- Predicted vs. actual age plots
- SHAP summary plots
- Violin distributions of brain-age delta
- Bias correction delta plots per dataset
- ML algorithm comparison charts
- Brain surface visualizations (via ENIGMA toolbox)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **`config.yaml` path errors** | The default paths in `SHAP/config.yaml` point to a Windows directory (`C:/Users/HYUK/...`). Update `datapath`, `results`, and `genpath` to valid paths on your local machine. |
| **`functions/` module not found** | Some scripts in `SHAP/` import from a `functions` module. Ensure the `functions/` package is on your Python path. You may need to add the parent directory to `sys.path` or set `PYTHONPATH` before running SHAP scripts. |
| **SHAP dataset path mismatch** | When switching between datasets for SHAP analysis, refer to the notes file (`shapy value 구할때...txt`) in `SHAP/` for which paths and variables to update. |
| **Missing `requirements.txt`** | If `requirements.txt` is not yet present, install dependencies manually: `pip install numpy pandas scikit-learn pycaret shap tqdm pyyaml regex matplotlib seaborn notebook` |
| **Pickle compatibility** | Saved models in `models/` are Python pickle files (`.pkl`). Ensure you use the same Python and scikit-learn version that was used to train them, or retrain from scratch. |

## Citation

If you use this work, please cite:

J. Han et al., "Brain Age Prediction: A Comparison between Machine Learning Models Using Brain Morphometric Data," _Sensors_, vol. 22, no. 20, p. 8077, Oct. 2022. DOI: 10.3390/s22208077
