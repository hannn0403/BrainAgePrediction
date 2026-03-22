# %%
import numpy as np
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt

dflist_ar = "./revision_shap_mean_absolute/viz/cc_feature_importance_ar.csv"
dflist_ct = "./revision_shap_mean_absolute/viz/cc_feature_importance_ar.csv"

df_ar = pd.read_csv(dflist_ar)
df_ar.head()

# ==== area / thickness viz ====

# ==== GBR ====

# %%
flot_data_ar = np.array(df_ar["Gradient Boosting Regressor scale"])
#flot_data_ar = abs(flot_data_ar)

df_ct = pd.read_csv(dflist_ct)

flot_data_ct = np.array(df_ct["Gradient Boosting Regressor scale"])
#flot_data_ct = abs(flot_data_ct)
#flot_data

# %%
from enigmatoolbox.datasets import load_summary_stats

# Load summary statistics for ENIGMA-22q
sum_stats = load_summary_stats('22q')

# Get case-control cortical thickness and surface area tables
CT = sum_stats['CortThick_case_vs_controls']
SA = sum_stats['CortSurf_case_vs_controls']

# Extract Cohen's d values
CT_d = CT['d_icv']
SA_d = SA['d_icv']

# %%
from enigmatoolbox.datasets import load_example_data

# Load all example data from an individual site
cov, metr1_SubVol, metr2_CortThick, metr3_CortSurf = load_example_data()

# %%
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical


# Map parcellated data to the surface
area = parcel_to_surface(flot_data_ar, 'aparc_fsa5')
thickness = parcel_to_surface(flot_data_ct, 'aparc_fsa5')

# Project the results on the surface brain
plot_cortical(array_name=area, surface_name="fsa5", size=(800, 250),
              cmap='Reds', color_bar="bottom", color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="cc_area_GBRtest_scale.png", transparent_bg =True)

plot_cortical(array_name=thickness, surface_name="fsa5", size=(800, 400),
              cmap='Blues', color_bar="bottom", color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="cc_thickness_GBRtest_scale.png", transparent_bg =True)

# ==== LASSO ====

# %%
flot_data_ar = np.array(df_ar["Lasso Scale"])
#flot_data_ar = abs(flot_data_ar)

df_ct = pd.read_csv(dflist_ct)

flot_data_ct = np.array(df_ct["Lasso Scale"])
#flot_data_ct = abs(flot_data_ct)
#flot_data

# %%
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical


# Map parcellated data to the surface
area = parcel_to_surface(flot_data_ar, 'aparc_fsa5')
thickness = parcel_to_surface(flot_data_ct, 'aparc_fsa5')

# Project the results on the surface brain
plot_cortical(array_name=area, surface_name="fsa5", size=(800, 250),
              cmap='Reds', color_bar= False, color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="ixi_area_Lasso_scale.png", transparent_bg =True)

plot_cortical(array_name=thickness, surface_name="fsa5", size=(800, 250),
              cmap='Blues', color_bar= False, color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="ixi_thickness_Lasso_scale.png", transparent_bg =True)

# ==== GPR ====

# %%
flot_data_ar = np.array(df_ar["Gaussian Process Scale"])
#flot_data_ar = abs(flot_data_ar)

df_ct = pd.read_csv(dflist_ct)

flot_data_ct = np.array(df_ct["Gaussian Process Scale"])
#flot_data_ct = abs(flot_data_ct)
#flot_data

# %%
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical


# Map parcellated data to the surface
area = parcel_to_surface(flot_data_ar, 'aparc_fsa5')
thickness = parcel_to_surface(flot_data_ct, 'aparc_fsa5')

# Project the results on the surface brain
plot_cortical(array_name=area, surface_name="fsa5", size=(800, 250),
              cmap='Reds', color_bar= False, color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="ixi_area_GP_scale.png", transparent_bg =True)

plot_cortical(array_name=thickness, surface_name="fsa5", size=(800, 250),
              cmap='Blues', color_bar= False, color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="ixi_thickness_GP_scale.png", transparent_bg =True)

# ==== subcortical volume ====

# %%
from enigmatoolbox.datasets import load_example_data

# Load all example data from an individual site
cov, metr1_SubVol, metr2_CortThick, metr3_CortSurf = load_example_data()

# %%
from enigmatoolbox.datasets import load_summary_stats

# Load summary statistics for ENIGMA-Epilepsy
sum_stats = load_summary_stats('epilepsy')

# Get case-control subcortical volume and cortical thickness tables
SV = sum_stats['SubVol_case_vs_controls_ltle']
CT = sum_stats['CortThick_case_vs_controls_ltle']

# Extract Cohen's d values
SV_d = SV['d_icv']
CT_d = CT['d_icv']

# ==== GBR ====

# %%
dflist_sv = "./shap_value/visual/ixi_feature_importance_sv.csv"

df_sv = pd.read_csv(dflist_sv)
print(df_sv.head())
flot_data_sv = np.array(df_sv["Gradient Boosting Regressor Scale"])

# %%
from enigmatoolbox.plotting import plot_subcortical

# Project the results on the surface brain
plot_subcortical(array_name=flot_data_sv, size=(800, 250),
                 cmap='Greens', color_bar="bottom", color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="ixi_SV_GBR_scale.png", transparent_bg =True)

# ==== Lasso ====

# %%
dflist_sv = "./shap_value/visual/ixi_feature_importance_sv.csv"

df_sv = pd.read_csv(dflist_sv)
df_sv.head()
flot_data_sv = np.array(df_sv["Lasso Scale"])

from enigmatoolbox.plotting import plot_subcortical

# Project the results on the surface brain
plot_subcortical(array_name=flot_data_sv, size=(800, 250),
                 cmap='Greens', color_bar= False, color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="ixi_SV_Lasso_scale.png", transparent_bg =True)

# ==== GP ====

# %%
dflist_sv = "./shap_value/visual/ixi_feature_importance_sv.csv"

df_sv = pd.read_csv(dflist_sv)
df_sv.head()
flot_data_sv = np.array(df_sv["Gaussian Process Scale"])

from enigmatoolbox.plotting import plot_subcortical

# Project the results on the surface brain
plot_subcortical(array_name=flot_data_sv, size=(800, 250),
                 cmap='Greens', color_bar= False, color_range=(0, 1), share="both",zoom =1.2,scale=2,
              screenshot=True, filename="ixi_SV_GP_scale.png", transparent_bg =True)

# ==== DK atlas ====

# %%
from enigmatoolbox.datasets import load_example_data

# Load all example data from an individual site
cov, metr1_SubVol, metr2_CortThick, metr3_CortSurf = load_example_data()

# %%
from enigmatoolbox.datasets import load_summary_stats

# Load summary statistics for ENIGMA-22q
sum_stats = load_summary_stats('22q')

# Get case-control cortical thickness and surface area tables
CT = sum_stats['CortThick_case_vs_controls']
SA = sum_stats['CortSurf_case_vs_controls']

# Extract Cohen's d values
CT_d = CT['d_icv']
SA_d = SA['d_icv']

# %%
from enigmatoolbox.utils.useful import reorder_sctx

# Re-order the subcortical data alphabetically and by hemisphere
metr1_SubVol_r = reorder_sctx(metr1_SubVol)

# %%
import numpy as np
from enigmatoolbox.plotting import plot_cortical
from enigmatoolbox.utils.parcellation import parcel_to_surface

# Extract FDR-corrected p-values and find regions with p < 0.01
region_idx = np.where(CT['fdr_p'].to_numpy() <= 0.01)

# Visualize thresholded Cohen's d map
CT_d_thr = np.arange(68)
np.random.shuffle(CT_d_thr)
plot_cortical(array_name=parcel_to_surface(CT_d_thr, 'aparc_fsa5'), surface_name="fsa5", size=(800, 250),
              cmap='gist_rainbow', color_bar=False, color_range=(0,67),
             share="both",zoom =1.2,scale=2, transparent_bg =True) #, screenshot=True, filename="DK_atlastest.png"
# hsv nipy_spectral jet

# %%
CT_d_thr = np.arange(0,68,1)

# %%
CT_d_thr = np.arange(68)
print(CT_d_thr)

np.random.shuffle(CT_d_thr)
print(CT_d_thr)

# %%
from enigmatoolbox.datasets import load_example_data

# Load all example data from an individual site
cov, metr1_SubVol, metr2_CortThick, metr3_CortSurf = load_example_data()

from enigmatoolbox.datasets import load_summary_stats

# Load summary statistics for ENIGMA-Epilepsy
sum_stats = load_summary_stats('epilepsy')

# Get case-control subcortical volume and cortical thickness tables
SV = sum_stats['SubVol_case_vs_controls_ltle']
CT = sum_stats['CortThick_case_vs_controls_ltle']

# Extract Cohen's d values
SV_d = SV['d_icv']
CT_d = CT['d_icv']

sv_vis = np.arange(16)
np.random.shuffle(sv_vis)

# %%
from enigmatoolbox.plotting import plot_subcortical

# Project the results on the surface brain

plot_subcortical(array_name= sv_vis, size=(800, 250),
                 cmap='gist_rainbow', color_bar=False, color_range=(0,16),
                share="both",zoom =1.2,scale=2, screenshot=True, filename="DK_SV.png", transparent_bg =True)

# %%
len(SV_d)

# %%
from enigmatoolbox.datasets import load_summary_stats

# Load summary statistics for ENIGMA-22q
sum_stats = load_summary_stats('22q')

# Get case-control cortical thickness and surface area tables
CT = sum_stats['CortThick_case_vs_controls']
SA = sum_stats['CortSurf_case_vs_controls']

# Extract Cohen's d values
CT_d = CT['d_icv']
SA_d = SA['d_icv']

# %%
from enigmatoolbox.utils.useful import reorder_sctx

# Re-order the subcortical data alphabetically and by hemisphere
metr1_SubVol_r = reorder_sctx(metr1_SubVol)

# %%
import numpy as np
from enigmatoolbox.plotting import plot_cortical
from enigmatoolbox.utils.parcellation import parcel_to_surface

# Extract FDR-corrected p-values and find regions with p < 0.01
region_idx = np.where(CT['fdr_p'].to_numpy() <= 0.01)

# Visualize thresholded Cohen's d map
CT_d_thr = np.arange(68)
np.random.shuffle(CT_d_thr)
plot_cortical(array_name=parcel_to_surface(CT_d_thr, 'aparc_fsa5'), surface_name="fsa5", size=(800, 250),
              cmap='gist_rainbow', color_bar=False, color_range=(0,67),
             share="both",zoom =1.2,scale=2, screenshot=True, filename="DK_atlas.png", transparent_bg =True)
# hsv nipy_spectral jet

# %%
CT_d_thr = np.arange(0,68,1)

# %%
CT_d_thr = np.arange(68)
print(CT_d_thr)

np.random.shuffle(CT_d_thr)
print(CT_d_thr)

# %%
from enigmatoolbox.datasets import load_example_data

# Load all example data from an individual site
cov, metr1_SubVol, metr2_CortThick, metr3_CortSurf = load_example_data()

from enigmatoolbox.datasets import load_summary_stats

# Load summary statistics for ENIGMA-Epilepsy
sum_stats = load_summary_stats('epilepsy')

# Get case-control subcortical volume and cortical thickness tables
SV = sum_stats['SubVol_case_vs_controls_ltle']
CT = sum_stats['CortThick_case_vs_controls_ltle']

# Extract Cohen's d values
SV_d = SV['d_icv']
CT_d = CT['d_icv']

sv_vis = np.arange(16)
np.random.shuffle(sv_vis)

# %%
from enigmatoolbox.plotting import plot_subcortical

# Project the results on the surface brain

plot_subcortical(array_name= sv_vis, size=(800, 250),
                 cmap='gist_rainbow', color_bar=False, color_range=(0,16),
                share="both",zoom =1.2,scale=2, screenshot=True, filename="DK_SV.png", transparent_bg =True)

# %%
len(SV_d)
