# %%
import pandas as pd

ixi_train = pd.read_csv('./IXI_153/ixi_train.csv', index_col = 0)
ixi_test = pd.read_csv('./IXI_153/ixi_test.csv', index_col = 0)

cc_train = pd.read_csv('./CAMCAN_153/cc_train.csv', index_col = 0)
cc_test = pd.read_csv('./CAMCAN_153/cc_test.csv', index_col = 0)

hcp_train = pd.read_csv('./HCP_153/hcp_train.csv', index_col = 0)
hcp_test = pd.read_csv('./HCP_153/hcp_test.csv', index_col = 0)

# %%
print(ixi_train.shape)

# %%
print(cc_train.shape)

# %%
print(hcp_train.shape)

# %%
for i in ixi_train.columns:
    print(i)

# %%
right = []
left = []
etc = []

for i in ixi_train.columns:
    if (i[:4] == 'Left') or (i[:2] == 'lh'):
        left.append(i)
    elif (i[:5] == 'Right') or (i[:2] == 'rh'):
        right.append(i)
    else:
        etc.append(i)

# %%
len(right)

# %%
len(left)

# %%
len(etc)

# %%
etc

# %%
area = []
thickness = []
etc_col = []

for i in ixi_train.columns:
    if i[-4:] == 'area':
        area.append(i)
    elif i[-9:] == 'thickness':
        thickness.append(i)
    else :
        etc_col.append(i)

# %%
len(area)

# %%
len(thickness)

# %%
etc_col

# %%
area

# %%
thickness
