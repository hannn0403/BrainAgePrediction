# ==== CAMCAN Dataset ====

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


cc_train = pd.read_csv('./CAMCAN_train.csv')
cc_test = pd.read_csv('./CAMCAN_test.csv')



cc_train = cc_train.iloc[:, :159]
cc_test = cc_test.iloc[:, :159]

cc_train.drop(['ID', 'hand','cattell', 'ICV'], axis = 1, inplace =True)
cc_test.drop(['ID','hand','cattell', 'ICV'], axis = 1, inplace =True)

print('cc_train.csv Shape : ', cc_train.shape)
print('cc_test.csv Shape : ', cc_test.shape)

# In this dataset, they include sex, age columns.
# So, practically we use 153 columns when we use this dataset

# %%
cc_x_train = cc_train.drop('age', axis = 1)
cc_y_train = cc_train[['age']]

cc_x_test = cc_test.drop('age', axis = 1)
cc_y_test = cc_test[['age']]

# %%
cc_y_test

# %%
cc_x_train.to_csv('./CAMCAN_153/cc_x_train.csv')
cc_y_train.to_csv('./CAMCAN_153/cc_y_train.csv')

cc_x_test.to_csv('./CAMCAN_153/cc_x_test.csv')
cc_y_test.to_csv('./CAMCAN_153/cc_y_test.csv')

# %%
cc_train.to_csv('./CAMCAN_153/cc_train.csv')
cc_test.to_csv('./CAMCAN_153/cc_test.csv')

# ==== HCP Dataset ====

# %%
hcp_152 = pd.read_csv('./HCP_152features.csv')
hcp_1200 = pd.read_csv('./HCP_1200.csv')

print('hcp_152 shape : ', hcp_152.shape)
print('hcp_1200 shape : ', hcp_1200.shape)

# %%
for i in hcp_1200.columns:
    print(i)

# %%
hcp_1200 = hcp_1200.drop(['Subject','PMAT24_A_CR'], axis = 1)
print(hcp_1200.shape)

# ==== HCP Dataset Split ====

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from verstack.stratified_continuous_split import scsplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def df_info_hcp(df):
    # Shape
    print('shape : ', df.shape)

    # DF가 바뀌면 이부분만 바뀐다.
    label = df['Age']
    # Minimum Value
    print('Min : {:.2f}'.format(np.min(label)))
    # Maximum Value
    print('Max : {:.2f}'.format( np.max(label)))

    print('Range : {:.2f} ~ {:.2f}'.format( np.min(label), np.max(label)))

    print('Median : {:.2f}'.format(np.median(label)))
    print('Standard Deviation : {:.2f}'.format(np.std(label)) )

    plt.title("Age Distribution")
    sns.histplot(data = df, x = 'Age')

# %%
df_info_hcp(hcp_1200)

# %%
hcp_y = hcp_1200['Age']
hcp_X = hcp_1200.drop('Age', axis = 1)

# 5 bins may be too few for larger datasets.
bins     = np.linspace(start=22, stop=37, num=5)
y_binned = np.digitize(hcp_y, bins, right=True)

hcp_x_train, hcp_x_test, hcp_y_train, hcp_y_test = train_test_split(hcp_X, hcp_y, test_size = 0.2, stratify=y_binned, shuffle = True, random_state = 1)

# %%
print(hcp_x_train.shape)
print(hcp_y_train.shape)

print(hcp_x_test.shape)
print(hcp_y_test.shape)

# %%
hcp_train = pd.concat([hcp_x_train, hcp_y_train], axis =1)
hcp_test = pd.concat([hcp_x_test, hcp_y_test], axis =1)

# %%
df_info_hcp(hcp_train)

# %%
df_info_hcp(hcp_test)

# %%
hcp_x_train = hcp_train.drop('Age', axis = 1)
hcp_y_train = hcp_train[['Age']]

hcp_x_test = hcp_test.drop('Age', axis =1)
hcp_y_test = hcp_test[['Age']]

# %%
hcp_x_train.to_csv('./HCP_153/hcp_x_train.csv')
hcp_y_train.to_csv('./HCP_153/hcp_y_train.csv')

hcp_x_test.to_csv('./HCP_153/hcp_x_test.csv')
hcp_y_test.to_csv('./HCP_153/hcp_y_test.csv')

# %%
hcp_train.to_csv('./HCP_153/hcp_train.csv')
hcp_test.to_csv('./HCP_153/hcp_test.csv')

# ==== IXI Dataset ====

# %%
ixi_train = pd.read_csv('ixi_train.csv', index_col = 0)
ixi_test = pd.read_csv('ixi_test.csv', index_col = 0)
print(ixi_train.shape)

# %%
ixi_train = ixi_train.drop(['HEIGHT', 'WEIGHT','scan_site_HH', 'scan_site_IOP', 'BrainSegVolNotVent', 'lh_MeanThickness_thickness', 'lh_WhiteSurfArea_area', 'rh_MeanThickness_thickness', 'rh_WhiteSurfArea_area'], axis = 1)
ixi_test = ixi_test.drop(['HEIGHT', 'WEIGHT','scan_site_HH', 'scan_site_IOP', 'BrainSegVolNotVent', 'lh_MeanThickness_thickness', 'lh_WhiteSurfArea_area', 'rh_MeanThickness_thickness', 'rh_WhiteSurfArea_area'], axis = 1)

# %%
ixi_train.drop('brain_gap', axis = 1, inplace = True)
ixi_test.drop('brain_gap', axis = 1, inplace =True)

# %%
ixi_train.drop('predicted_age', axis = 1, inplace = True)
ixi_test.drop('predicted_age', axis = 1, inplace =True)

# %%
print(ixi_train.shape)
print(ixi_test.shape)

# %%
ixi_train.to_csv('./IXI_153/ixi_train.csv')
ixi_test.to_csv('./IXI_153/ixi_test.csv')

# ==== Final Check ====

# %%
ixi_train = pd.read_csv('./IXI_153/ixi_train.csv', index_col = 0)
ixi_test = pd.read_csv('./IXI_153/ixi_test.csv', index_col = 0)

cc_train = pd.read_csv('./CAMCAN_153/cc_train.csv', index_col = 0)
cc_test = pd.read_csv('./CAMCAN_153/cc_test.csv', index_col = 0)

hcp_train = pd.read_csv('./HCP_153/hcp_train.csv', index_col = 0)
hcp_test = pd.read_csv('./HCP_153/hcp_test.csv', index_col = 0)

# %%
print(ixi_train.shape)
print(ixi_test.shape)

print(cc_train.shape)
print(cc_test.shape)

print(hcp_train.shape)
print(hcp_test.shape)
