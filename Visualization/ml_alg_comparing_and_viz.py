# %%
from pycaret.regression import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
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

# ==== Check Predicted Age Distribution for Kernel Ridge ====

# %%
cc_pred_age = pd.read_csv('./dataframe/cc/cc_predicted_age.csv', index_col=0)

# %%
cc_pred_age

# %%
cc_pred_age_kr = cc_pred_age[['KernelRidge']]

cc_pred_age_kr.sort_values('KernelRidge', ascending=False)

# ==== Check whether there is a difference in the range of the Feature Importance of the Tree family model and the Linear model ====

# %%
ixi_feat_imp = pd.read_csv("./dataframe/ixi/ixi_feat_imp.csv", index_col=0)

# %%
ixi_feat_imp.head()

# %%
ixi_feat_imp.columns

# %%
linear_list=['LinearRegression','Lasso','Ridge','ElasticNet','Lars','LassoLars','OrthogonalMatchingPursuit','Bayesian Ridge','ARD','PassiveAggressiveRegressor','RANSAC',
            'TheilsenRegressor','HuberRegressor','KernelRidge','SVR','GaussianProcess']
tree_list = ['Decision Tree','RandomForestRegressor','ExtraTreesRegressor','Adaboost','GradientBoostingRegressor','XGBRegressor','LGBMRegressor','Catboost']

# %%
for col in linear_list:
    print(f"{col} Feature Importance Range :\n {round(ixi_feat_imp[col].min(),2)} ~ {round(ixi_feat_imp[col].max(),2)}\n")

# %%
for col in tree_list:
    print(f"{col} Feature Importance Range :\n {round(ixi_feat_imp[col].min(),2)} ~ {round(ixi_feat_imp[col].max(),2)}\n")

# %%
ixi_feat_violin = ixi_feat_imp.copy()

# %%
ixi_feat_violin.columns=['lr','lasso','ridge','en','lar','llar','omp','br','ard','par','ransac','tr','huber','kr','svr','gp','dt','rf','et','ada','gbm','xgb','lgbm','catboost']

# %%
sns.violinplot(data=ixi_feat_violin, figsize=(15,5))

# %%
sns.set(rc={'figure.figsize':(21, 15)})

ax = sns.violinplot(data=ixi_feat_violin)
#ax.set_title('IXI Predicted Brain Age', fontsize=25, weight="bold")
ax.set_xlabel('Models', fontsize=15, weight="bold")
ax.set_ylabel('Feature Importance', fontsize=15, weight="bold")
plt.show()

# %%
ixi_feat_violin.astype(bool).sum(axis=0)

# ==== Non-zero value count analysis ====
# The above result is the count of non-zero values for each model's Feature Importance.
# Tree models, except for Decision Tree Regressor, use almost all features.

# ==== How many types of Predicted Age values are there for each model? ====

# %%
cc_tree_pred_age = cc_pred_age.loc[:,tree_list ]

# %%
len(cc_tree_pred_age.iloc[:,0].value_counts())

# %%
for i in range(len(tree_list)):
    print(f"{cc_tree_pred_age.columns[i]}")
    print(f"Predicted Age Value Types : {len(cc_tree_pred_age.iloc[:, i].value_counts())}\n")

# %%
cc_lin_pred_age = cc_pred_age.loc[:, linear_list]

for i in range(len(linear_list)):
    print(f"{cc_lin_pred_age.columns[i]}")
    print(f"Predicted Age Value Types : {len(cc_lin_pred_age.iloc[:, i].value_counts())}\n")

# %%
hcp_pred_age = pd.read_csv('./dataframe/hcp/hcp_predicted_age.csv', index_col=0)

hcp_tree_pred_age = hcp_pred_age.loc[:,tree_list ]
hcp_lin_pred_age = hcp_pred_age.loc[:, linear_list]

for i in range(len(tree_list)):
    print(f"{hcp_tree_pred_age.columns[i]}")
    print(f"Predicted Age Value Types : {len(hcp_tree_pred_age.iloc[:, i].value_counts())}\n")

for i in range(len(linear_list)):
    print(f"{hcp_lin_pred_age.columns[i]}")
    print(f"Predicted Age Value Types : {len(hcp_lin_pred_age.iloc[:, i].value_counts())}\n")

# %%
ixi_pred_age = pd.read_csv('./dataframe/ixi/ixi_predicted_age.csv', index_col=0)

ixi_tree_pred_age = ixi_pred_age.loc[:,tree_list ]
ixi_lin_pred_age = ixi_pred_age.loc[:, linear_list]

for i in range(len(tree_list)):
    print(f"{ixi_tree_pred_age.columns[i]}")
    print(f"Predicted Age Value Types : {len(ixi_tree_pred_age.iloc[:, i].value_counts())}\n")

for i in range(len(linear_list)):
    print(f"{ixi_lin_pred_age.columns[i]}")
    print(f"Predicted Age Value Types : {len(ixi_lin_pred_age.iloc[:, i].value_counts())}\n")

# ==== Feature Importance MinMax Scaling ====

# %%
ixi_feat_imp = pd.read_csv("./dataframe/ixi/ixi_feat_imp.csv", index_col=0)
hcp_feat_imp = pd.read_csv("./dataframe/hcp/hcp_feat_imp.csv", index_col=0)
cc_feat_imp = pd.read_csv("./dataframe/cc/cc_feat_imp.csv", index_col=0)

# %%
ixi_feat_imp_copy = ixi_feat_imp.copy()
hcp_feat_imp_copy = hcp_feat_imp.copy()
cc_feat_imp_copy = cc_feat_imp.copy()

# %%
def scaling(x, col_max):
    return x / col_max

for col in ixi_feat_imp_copy.columns.to_list():
    ixi_feat_imp_copy[col] = ixi_feat_imp[col].apply(lambda x : abs(x))
    col_max = ixi_feat_imp_copy[col].max()
    ixi_feat_imp_copy[col] = ixi_feat_imp_copy[col].apply(lambda x : scaling(x, col_max))


for col in hcp_feat_imp_copy.columns.to_list():
    hcp_feat_imp_copy[col] = hcp_feat_imp[col].apply(lambda x : abs(x))
    col_max = hcp_feat_imp_copy[col].max()
    hcp_feat_imp_copy[col] = hcp_feat_imp_copy[col].apply(lambda x : scaling(x, col_max))

for col in cc_feat_imp_copy.columns.to_list():
    cc_feat_imp_copy[col] = cc_feat_imp[col].apply(lambda x : abs(x))
    col_max = cc_feat_imp_copy[col].max()
    cc_feat_imp_copy[col] = cc_feat_imp_copy[col].apply(lambda x : scaling(x, col_max))

# %%
ixi_feat_imp.head()

# %%
ixi_feat_imp_copy.head()

# %%
ixi_feat_imp_copy.describe()

# %%
hcp_feat_imp_copy.describe()

# %%
def save_feat_heatmap(age_feat_imp, dataset, save_file_name, vmin=None, vmax=None):
    plt.figure(figsize=(15,10))

    dataplot = sns.heatmap(age_feat_imp, vmin=vmin, vmax=vmax, cmap="YlGnBu", annot=False, square=True)
    plt.title(f'{dataset} Feature Importance Correlation', fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(f'./visualization/Scaling_{save_file_name}.png', dpi=300)

# %%
save_feat_heatmap(ixi_feat_imp_copy.corr(), 'IXI', 'ixi_feat_imp')

# %%
save_feat_heatmap(hcp_feat_imp_copy.corr(), 'HCP', 'hcp_feat_imp')

# %%
save_feat_heatmap(cc_feat_imp_copy.corr(), 'CAMCAN', 'cc_feat_imp')

# %%
def save_feat_clustermap(age_feat_imp, dataset, save_file_name):
    g = sns.clustermap(age_feat_imp, cmap='YlGnBu', col_cluster=True, figsize=(16,16),square=True, vmin=0, vmax=1)
    #g.fig.suptitle(f'{dataset} Feature Importance Hierachical Clustering', y=0.92, fontsize=20)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([1.05, 0.3, 0.02, 0.3])
    g.ax_cbar.set_title('Feature Importance Correlation', x=3.5, y=0.15, loc='right', rotation=90)
    g.savefig(f'./visualization/{save_file_name}.png', bbox_inches='tight',pad_inches = 0, dpi=300)
