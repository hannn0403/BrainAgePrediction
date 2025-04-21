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
Brain structural morphology varies over the aging trajectory, and the prediction of a person’s age using brain morphological features can help the detection of an abnormal aging process. Neuroimaging-based brain age is widely used to quantify an individual’s brain health as deviation from a normative brain aging trajectory. Machine learning approaches are expanding the potential for accurate brain age prediction but are challenging due to the great variety of machine learning algorithms. Here, we aimed to compare the performance of the machine learning models used to estimate brain age using brain morphological measures derived from structural magnetic resonance imaging scans. We evaluated 27 machine learning models, applied to three independent datasets from the Human Connectome Project (HCP, n = 1113, age range 22–37), the Cambridge Centre for Ageing and Neuroscience (Cam-CAN, n = 601, age range 18–88), and the Information eXtraction from Images (IXI, n = 567, age range 19–86). Performance was assessed within each sample using cross-validation and an unseen test set. The models achieved mean absolute errors of 2.75–3.12, 7.08–10.50, and 8.04–9.86 years, as well as Pearson’s correlation coefficients of 0.11–0.42, 0.64–0.85, and 0.63–0.79 between predicted brain age and chronological age for the HCP, Cam-CAN, and IXI samples, respectively. We found a substantial difference in performance between models trained on the same data type, indicating that the choice of model yields considerable variation in brain-predicted age. Furthermore, in three datasets, regularized linear regression algorithms achieved similar performance to nonlinear and ensemble algorithms. Our results suggest that regularized linear algorithms are as effective as nonlinear and ensemble algorithms for brain age prediction, while significantly reducing computational costs. Our findings can serve as a starting point and quantitative reference for future efforts at improving brain age prediction using machine learning models applied to brain morphometric data.


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

## Repository Structure

```
BrainAgePrediction/
├── SHAP/                           # Model training, evaluation, and SHAP analysis scripts
│   ├── config.yaml                 # Paths and preprocessing/model hyperparameters
│   ├── run_brain_age_models_hcp.py # Train & evaluate on HCP dataset
│   ├── run_brain_age_models_camcan.py # Train & evaluate on Cam-CAN dataset
│   ├── run_brain_age_models_ixi.py    # Train & evaluate on IXI dataset
│   ├── feature_importance_*.py     # SHAP & feature importance utilities
│   └── shapy value 구할때...txt       # Notes on modifying dataset paths for SHAP
├── dataset/                        # Pre-split CSV files for each dataset
│   ├── hcp_train.csv
│   ├── hcp_test.csv
│   ├── cc_train.csv
│   ├── cc_test.csv
│   ├── ixi_train.csv
│   └── ixi_test.csv
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
│   └── score_grid/                 # Grid search performance results
├── visualization/                  # Plotting scripts & saved figures
│   ├── Predicted Brain Age/        # Predicted vs. chronological plots
│   ├── Feature Importance/         # SHAP summary & importance plots
│   └── violin_plot/                # Distribution comparisons
├── brain_age_prediction_training.ipynb # End-to-end training & evaluation notebook
├── brain_age_prediction_tuning.ipynb   # Hyperparameter tuning & analysis notebook
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

## Notebook Workflow

For an interactive demonstration, run:

```bash
jupyter notebook brain_age_prediction_training.ipynb
```

This notebook walks through data loading, preprocessing, model fitting, and evaluation with inline visualizations.

## Visualization

Generate or recreate figures by running notebooks or scripts in `visualization/`:

- Predicted vs. actual age plots
- SHAP summary plots
- Violin distributions of brain-age delta

## Citation

If you use this work, please cite:

J. Han et al., “Brain Age Prediction: A Comparison between Machine Learning Models Using Brain Morphometric Data,” _Sensors_, vol. 22, no. 20, p. 8077, Oct. 2022. DOI: 10.3390/s22208077

