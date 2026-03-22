# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from verstack.stratified_continuous_split import scsplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cc_train = pd.read_csv('./integrated_data/CAMCAN_train.csv')
cc_test = pd.read_csv('./integrated_data/CAMCAN_test.csv')

hcp = pd.read_csv('./integrated_data/HCP.csv')

ixi = pd.read_csv('./integrated_data/thickness_area_sub.csv')

# ==== CAMCAN Data Set ====

# %%
print('train shape : ', cc_train.shape)
print('test shape : ', cc_test.shape)

# %%
cc_train

# %%
train_label = cc_train['Age']
plt.hist(train_label)

# %%
#Maximum Value
print('Max : ', np.max(train_label))

#Minimum Value
print('Min : ', np.min(train_label))

# %%
np.median(train_label)

# %%
np.std(train_label)

# %%
def df_info_cc(df):
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
df_info_cc(cc_train)

# %%
df_info_cc(cc_test)

# ==== HCP Data Set ====

# %%
hcp

# %%
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
df_info_hcp(hcp)

# %%
y = hcp['Age']
X = hcp.drop('Age', axis = 1)

# %%
# 5 bins may be too few for larger datasets.
bins     = np.linspace(start=22, stop=37, num=5)
y_binned = np.digitize(y, bins, right=True)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, stratify=y_binned, shuffle = True, random_state = 1)

# %%
hcp_train = pd.concat([X_train, y_train], axis =1)
hcp_test = pd.concat([X_test, y_test], axis =1)

# %%
df_info_hcp(hcp_train)

# %%
df_info_hcp(hcp_test)

# %%
hcp_train.to_csv('./integrated_data/hcp_train.csv')
hcp_test.to_csv('./integrated_data/hcp_test.csv')

# ==== IXI Data Set ====

# %%
ixi = pd.read_csv('./integrated_data/thickness_area_sub.csv')
ixi = ixi.drop(['Unnamed: 0', 'ID'], axis =1)
ixi = ixi.dropna()
ixi

# %%
def df_info_ixi(df):
    # Shape
    print('shape : ', df.shape)

    # DF가 바뀌면 이부분만 바뀐다.
    label = df['AGE']
    # Minimum Value
    print('Min : {:.2f}'.format(np.min(label)))
    # Maximum Value
    print('Max : {:.2f}'.format( np.max(label)))

    print('Range : {:.2f} ~ {:.2f}'.format( np.min(label), np.max(label)))

    print('Median : {:.2f}'.format(np.median(label)))
    print('Standard Deviation : {:.2f}'.format(np.std(label)) )

    plt.title("Age Distribution")
    sns.histplot(data = df, x = 'AGE')

# %%
df_info_ixi(ixi)

# %%
ixi

# %%
y = ixi['AGE']
X = ixi.drop('AGE', axis = 1)

# 5 bins may be too few for larger datasets.
bins     = np.linspace(start=19.98, stop=86.32, num=5)
y_binned = np.digitize(y, bins, right=True)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, stratify=y_binned, shuffle = True, random_state = 1)

# %%
ixi_train = pd.concat([X_train, y_train], axis =1)
ixi_test = pd.concat([X_test, y_test], axis = 1)

# %%
df_info_ixi(ixi_train)

# %%
df_info_ixi(ixi_test)

# %%
ixi_train.to_csv('./integrated_data/ixi_train.csv')
ixi_test.to_csv('./integrated_data/ixi_test.csv')

# ==== Sex Value Counts ====

# %%
import pandas as pd
ixi_train = pd.read_csv('./integrated_data/ixi_train.csv')
ixi_test = pd.read_csv('./integrated_data/ixi_test.csv')

# %%
ixi_train['SEX'].value_counts()

# %%
ixi_test['SEX'].value_counts()

# %%
import pandas as pd
cc_train = pd.read_csv('./integrated_data/CAMCAN_train.csv')
cc_test = pd.read_csv('./integrated_data/CAMCAN_test.csv')

# %%
cc_train['Sex'].value_counts()

# %%
cc_test['Sex'].value_counts()

# %%
import pandas as pd
hcp_train = pd.read_csv('./integrated_data/hcp_train.csv')
hcp_test = pd.read_csv('./integrated_data/hcp_test.csv')

# %%
hcp_train['Sex'].value_counts()

# %%
hcp_test['Sex'].value_counts()
