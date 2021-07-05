# import libraries and package
from functions import *
import os
import numpy as np


# Load datasets
data_air = 'Data/Airwave1_Data_LPOS.xlsx'
var_info_air = 'Data/Airwave1_VarInfo_LPOS.xlsx'

data_mesa = 'Data/MESA_Data_LPOS.xlsx'
var_info_mesa = 'Data/MESA_VarInfo_LPOS.xlsx'

data_rtd = 'Data/Rotterdam_Data_LPOS.xlsx'
var_info_rtd = 'Data/Rotterdam_VarInfo_LPOS.xlsx'

data_ha = 'hardData/Airwave1xcms_SLPOS_scaled_Data.xlsx'
var_info_ha = 'hardData/Airwave1xcms_SLPOS_scaled_VarInfo.xlsx'

data_hm = 'hardData/MESA_LPOS_unmatched_Data.xlsx'
var_info_hm = 'hardData/MESA_LPOS_unmatched_VarInfo.xlsx'

add_info = 'Data/posAdductsTable_RP.xlsx'


# create subdirectory ./outfig if not exist
if not os.path.isdir('outfig'):
    os.makedirs('outfig')


# define variables
# m/z threshold
d_mz = 0.01
# retention time threshold
d_rt = 1/60
# correlation threshold
d_corr = 0.5
# list of m/z thresholds
d_mz_list = list(np.linspace(0.01, 0.001, 5))
# list of retention time thresholds
d_rt_list = list(np.linspace(1/60, 0.001, 5))
# list of correlation thresholds
d_corr_list = list(np.linspace(0.5, 0.9, 5))


# for the first dataset
# find all isotope matches with large thresholds
# input: variable information, data, m/z threshold, retention time threshold, correlation threshold
# output: dataframe of all isotope matches ['reference', 'target', 'correlation', 'mz_ref', 'mz_tar', 'rt_ref', 'rt_tar', 'fi_ref', 'fi_tar']
isotope_all_ha = iso_match_all(var_info_ha, data_ha, d_mz, d_rt, d_corr)

# calculate the incorrect number and total number of isotopic matches with different thresholds
# input: variable information, dataframe of all isotope matches, list of mz thresholds, list of rt thresholds,
#        list of correlation thresholds, classification method, whether consider reassign errors
# output: number of incorrect matches of cube grid search, number of total matches of cube grid search
incorrect_count_ha, total_count_ha = iso_grid_pre(var_info_ha, isotope_all_ha, d_mz_list, d_rt_list, d_corr_list, model='rlm', reassign_err=False)

# select the best setting of thresholds using linear regression
# find the sublist and reassigned list of isotope matches with the best setting of thresholds
# input: dataframe of all isotope matches, number of incorrect matches, number of total matches,
#        list of mz thresholds, list of rt thresholds, list of correlation thresholds
# output: best setting of thresholds, sublist of isotope matches, reassigned sublist of isotope matches
settings_ha, iso_sub_ha, isotope_match_ha = best_threshold_1(isotope_all_ha, incorrect_count_ha, total_count_ha, d_mz_list, d_rt_list, d_corr_list)

# obtain the dataframe of all features with corresponding isotope type
# input: variable information, reassigned sublist of isotope matches
# output: dataframe of all features with corresponding isotope type
isotope_type_ha = iso_final(var_info_ha, isotope_match_ha)

# find all adduct matches within each layer of isotope
# input: variable information, data, adduct information, isotope type, best setting of thresholds
# output: dataframe of all adduct matches
add_all_ha = add_match_all(var_info_ha, data_ha, add_info, isotope_type_ha, settings_ha)

# select valid adduct matches and obtain the dataframe of all features with corresponding isotope and adduct type
# input: isotope type, all adduct matches
# output: valid adduct matches, dataframe of all features with isotope and adduct type
adduct_match_ha, iso_add_type_ha = add_final(isotope_type_ha, add_all_ha)

# find carbon chain matches
# input: variable information, data, isotope and adduct type, mz threshold, rt threshold, correlation threshold, carbon chain mz difference
# output: all c2h4 matches with the same or unknown adduct type, all c2h4 matches with the same and known adduct type
c2h4_all_ha, c2h4_strict_ha = carbonchain_match(var_info_ha, data_ha, iso_add_type_ha, d_mz=0.002, d_rt=1, d_corr=-1, chain=28.031300128)

# delete the false positive c2h4 matches and plot c2h4 matches
# input: all c2h4 matches, regression model, regression model smooth parameter, accept range around regression line, whether plot
# output: true positive c2h4 matches
c2h4_ha = carbonchain_tp(c2h4_all_ha, model='lowess', frac=0.3, mad=3, plot=False)


# same for the first dataset
isotope_all_hm = iso_match_all(var_info_hm, data_hm, d_mz, d_rt, d_corr)
incorrect_count_hm, total_count_hm = iso_grid_pre(var_info_hm, isotope_all_hm, d_mz_list, d_rt_list, d_corr_list, model='rlm')
settings_hm, iso_sub_hm, isotope_match_hm = best_threshold_1(isotope_all_hm, incorrect_count_hm, total_count_hm, d_mz_list, d_rt_list, d_corr_list)
isotope_type_hm = iso_final(var_info_hm, isotope_match_hm)
add_all_hm = add_match_all(var_info_hm, data_hm, add_info, isotope_type_hm, settings_hm)
adduct_match_hm, iso_add_type_hm = add_final(isotope_type_hm, add_all_hm)
c2h4_all_hm = carbonchain_match(var_info_hm, data_hm, iso_add_type_hm, 0.002, 1, -1)
c2h4_hm = carbonchain_tp(c2h4_all_hm)


# match two datasets
# find all possible feature matches
# input: isotope and adduct type of first dataset, isotope and adduct type of second dataset, mz threshold, rt threshold
# output: all possible feature matches with the same or unknown adduct type, all possible feature matches with the same and known adduct type
feature_all_am, feature_strict_am = allfeature_match(iso_add_type_ha, iso_add_type_hm, d_mz=0.015, d_rt=0.1)

# obtain feature matches for the other direction
feature_all_ma = feature_all_am[['target', 'reference', 'mz_tar', 'mz_ref', 'rt_tar', 'rt_ref', 'fi_tar', 'fi_ref', 'iso_type', 'add_type']]
feature_strict_ma = feature_strict_am[['target', 'reference', 'mz_tar', 'mz_ref', 'rt_tar', 'rt_ref', 'fi_tar', 'fi_ref', 'iso_type', 'add_type']]

# find disconnected subnetworks in a dataset
# input: isotope and adduct type, isotope matches, adduct matches, whether include single node subnetworks
# output: list of disconnected subnetworks
network_list_ha = all_networks(iso_add_type_ha, isotope_match_ha, adduct_match_ha, single=False)
network_list_hm = all_networks(iso_add_type_hm, isotope_match_hm, adduct_match_hm, single=False)

# obtain the information of each match of subnetworks
# input: possible feature matches, disconnected subnetworks of first dataset, disconnected subnetworks of second dataset,
#        rt threshold, whether allow multiple subnetwork matches, least match in subnetworks, whether only allow perfect matches
#        multi_match: whether allow multiple target subnetworks connect to a reference subnetwork
#        least_match: the least number of feature matches in a subnetwork connection
#        perfect_match: no need to select between overlapped target subnetworks
# output: subnetwork connections
network_match_am = network_connection(feature_all_am, network_list_ha, network_list_hm, d_rt, multi_match=False, least_match=2, perfect_match=False)
network_match_ma = network_connection(feature_all_am, network_list_hm, network_list_ha, d_rt, multi_match=False, least_match=2, perfect_match=False)

# select high probability feature matches from all feature matches using subnetwork connections
# input: possible feature matches, disconnected subnetworks of first dataset, disconnected subnetworks of second dataset, subnetwork connections
# output: high probability feature matches
feature_hp_am = hpfeature_match(feature_all_am, network_list_ha, network_list_hm, network_match_am)


# obtain the lowess regression using high probability matches
# input: possible feature matches, high probability feature matches
# output: possible feature matches with lowess regression results
feature_all_am, feature_hp_am = feature_match_regression(feature_all_am, feature_hp_am)


# obtain penalisation score
# input: possible feature matches, high probability feature matches, RT/MZ/FI weight of penalisation score, whether plot
# output: possible feature matches with penalisation score
feature_all_am = all_match_penal(feature_all_am, feature_hp_am, 5, [1, 1, 0], plot=True)


# reduce multiple matches
# input: possible feature matches
# output: single possible feature matches
feature_single_am = reduce_multimatch_ps(feature_all_am)


# delete poor matches
# input: single possible feature matches
# output: good feature matches
feature_good_am = reduce_poormatch(feature_single_am)


# plot good, poor and multiple match
plot_goodmatch(feature_all_am, feature_single_am, feature_good_am)
