# import libraries and package
from functions import *
import os
import numpy as np
import pickle


# Load datasets
data_1 = 'Airwave1xcms_SLPOS_scaled_Data.xlsx'
var_info_1 = 'Airwave1xcms_SLPOS_scaled_VarInfo.xlsx'

data_2 = 'MESA_LPOS_unmatched_Data.xlsx'
var_info_2 = 'MESA_LPOS_unmatched_VarInfo.xlsx'

add_info = 'posAdductsTable_RP.xlsx'

annotation = 'SLPOS_annotations_201121.xlsx'

# define variables
# m/z threshold
d_mz = 0.005
# retention time threshold
d_rt = 0.005
# correlation threshold
d_corr = 0.5
# list of m/z thresholds
d_mz_list = list(np.linspace(0.005, 0.001, 9))
# list of retention time thresholds
d_rt_list = list(np.linspace(0.005, 0.001, 9))
# list of correlation thresholds
d_corr_list = list(np.linspace(0.5, 0.9, 9))


# for the first dataset
# find all isotope matches with large thresholds
# input: variable information, data, m/z threshold, retention time threshold, correlation threshold
# output: dataframe of all isotope matches ['reference', 'target', 'correlation', 'mz_ref', 'mz_tar', 'rt_ref', 'rt_tar', 'fi_ref', 'fi_tar']
isotope_all_1 = iso_match_all(var_info_1, data_1, d_mz, d_rt, d_corr)
isotope_all_1.to_excel('isotope_all_ha.xlsx')

# plot isotope features in RT, MZ and correlation
plot_isohist(isotope_all_1, title='ref')

# plot Fi ratio vs MZ between isotopic matches
isotope_all_type_1 = iso_classify(isotope_all_1)
plot_isomatch(isotope_all_type_1, isotope_all_type_1, model='plain', title='ref_all')

# calculate the incorrect number and total number of isotopic matches with different thresholds
# input: variable information, dataframe of all isotope matches, list of mz thresholds, list of rt thresholds,
#        list of correlation thresholds, classification method, whether consider reassign errors
# output: number of incorrect matches of cube grid search, number of total matches of cube grid search
incorrect_count_1, total_count_1 = iso_grid_pre(var_info_1, isotope_all_1, d_mz_list, d_rt_list, d_corr_list, model='rlm', reassign_err=False)
np.save('incorrect_count_ha', incorrect_count_1)
np.save('total_count_ha', total_count_1)

# incorrect_count_1_, total_count_1_ = iso_grid_pre(var_info_1, isotope_all_1, d_mz_list, d_rt_list, d_corr_list, model='svm', reassign_err=False)
# np.save('incorrect_count_ha_', incorrect_count_1_)
# np.save('total_count_ha_', total_count_1_)

# select the best setting of thresholds using linear regression
# find the sublist and reassigned list of isotope matches with the best setting of thresholds
# input: dataframe of all isotope matches, number of incorrect matches, number of total matches,
#        list of mz thresholds, list of rt thresholds, list of correlation thresholds
# output: best setting of thresholds, sublist of isotope matches, reassigned sublist of isotope matches
settings_1, iso_sub_1, isotope_match_1 = best_threshold_1(isotope_all_1, incorrect_count_1, total_count_1, d_mz_list, d_rt_list, d_corr_list, plot='wolabel')

# settings_1_, iso_sub_1_, isotope_match_1_ = best_threshold_1(isotope_all_1, incorrect_count_1_, total_count_1_, d_mz_list, d_rt_list, d_corr_list, plot='wolabel', model='svm')

# plot Fi ratio vs MZ between isotopic matches
plot_isomatch(iso_sub_1, iso_sub_1, model='rlm', title='ref_sublist')
plot_isomatch(iso_sub_1, isotope_match_1, model='rlm', title='ref_reassign')

# plot_isomatch(iso_sub_1_, iso_sub_1_, model='svm', title='ref_sublist')
# plot_isomatch(iso_sub_1_, isotope_match_1_, model='svm', title='ref_reassign')

with open('settings_ha.pkl', 'wb') as f:
    pickle.dump(settings_1, f)
iso_sub_1.to_excel('iso_sub_ha.xlsx')
isotope_match_1.to_excel('isotope_matah_ha.xlsx')

# obtain the dataframe of all features with corresponding isotope type
# input: variable information, reassigned sublist of isotope matches
# output: dataframe of all features with corresponding isotope type
isotope_type_1 = iso_final(var_info_1, isotope_match_1)
isotope_type_1.to_excel('isotope_type_ha.xlsx')

# find all adduct matches within each layer of isotope
# input: variable information, data, adduct information, isotope type, best setting of thresholds
# output: dataframe of all adduct matches
add_all_1 = add_match_all(var_info_1, data_1, add_info, isotope_type_1, settings_1)
add_all_1.to_excel('add_all_ha.xlsx')

# select valid adduct matches and obtain the dataframe of all features with corresponding isotope and adduct type
# input: isotope type, all adduct matches
# output: valid adduct matches, dataframe of all features with isotope and adduct type
adduct_match_1, iso_add_type_1 = add_final(isotope_type_1, isotope_match_1, add_all_1)
adduct_match_1.to_excel('adduct_match_ha.xlsx')
iso_add_type_1.to_excel('iso_add_type_ha.xlsx')

# plot adduct networks
plot_add(adduct_match_1, title='ref')


# same for the second dataset
isotope_all_2 = iso_match_all(var_info_2, data_2, d_mz, d_rt, d_corr)
isotope_all_2.to_excel('isotope_all_hm.xlsx')
incorrect_count_2, total_count_2 = iso_grid_pre(var_info_2, isotope_all_2, d_mz_list, d_rt_list, d_corr_list, model='rlm')
np.save('incorrect_count_hm', incorrect_count_2)
np.save('total_count_hm', total_count_2)
settings_2, iso_sub_2, isotope_match_2 = best_threshold_1(isotope_all_2, incorrect_count_2, total_count_2, d_mz_list, d_rt_list, d_corr_list)
with open('settings_hm.pkl', 'wb') as f:
    pickle.dump(settings_2, f)
iso_sub_2.to_excel('iso_sub_hm.xlsx')
isotope_match_2.to_excel('isotope_match_hm.xlsx')
isotope_type_2 = iso_final(var_info_2, isotope_match_2)
isotope_type_2.to_excel('isotope_type_hm.xlsx')
add_all_2 = add_match_all(var_info_2, data_2, add_info, isotope_type_2, settings_2)
add_all_2.to_excel('add_all_hm.xlsx')
adduct_match_2, iso_add_type_2 = add_final(isotope_type_2, isotope_match_2, add_all_2)
adduct_match_2.to_excel('adduct_match_hm.xlsx')
iso_add_type_2.to_excel('iso_add_tyme_hm.xlsx')

# plot two datasets in MZ/RT coloured by FI
plot_datasets(var_info_1, var_info_2)

# match two datasets
# find all possible feature matches
# input: isotope and adduct type of first dataset, isotope and adduct type of second dataset, mz threshold, rt threshold
# output: all possible feature matches with the same or unknown adduct type, all possible feature matches with the same and known adduct type
feature_all_12 = allfeature_match(iso_add_type_1, iso_add_type_2, d_mz=0.015, d_rt_l=0.1, d_rt_r=0.1)

# obtain feature matches for the other direction
feature_all_21 = feature_all_12[['target', 'reference', 'mz_tar', 'mz_ref', 'rt_tar', 'rt_ref', 'fi_tar', 'fi_ref', 'iso_type', 'add_type']]
feature_all_21.columns = feature_all_12.columns

# find disconnected subnetworks in a dataset
# input: isotope and adduct type, isotope matches, adduct matches, whether include single node subnetworks
# output: list of disconnected subnetworks
network_list_1 = all_networks(iso_add_type_1, isotope_match_1, adduct_match_1, single=False)
with open('network_list_a.pkl', 'wb') as f:
    pickle.dump(network_list_1, f)
network_list_2 = all_networks(iso_add_type_2, isotope_match_2, adduct_match_2, single=False)
with open('network_list_m.pkl', 'wb') as f:
    pickle.dump(network_list_2,	f)

# obtain the information of each match of subnetworks
# input: possible feature matches, disconnected subnetworks of first dataset, disconnected subnetworks of second dataset,
#        rt threshold, whether allow multiple subnetwork matches, least match in subnetworks, whether only allow perfect matches
#        multi_match: whether allow multiple target subnetworks connect to a reference subnetwork
#        least_match: the least number of feature matches in a subnetwork connection
#        perfect_match: no need to select between overlapped target subnetworks
# output: subnetwork connections
network_match_12 = network_connection(feature_all_12, network_list_1, network_list_2, d_rt, multi_match=False, least_match=2, perfect_match=False)
network_match_12.to_excel('network_match_am.xlsx')
network_match_21 = network_connection(feature_all_21, network_list_2, network_list_1, d_rt, multi_match=False, least_match=2, perfect_match=False)
network_match_21.to_excel('network_match_ma.xlsx')

network_match = network_valid(network_match_12, network_match_21)
network_match.to_excel('network_match.xlsx')

# select high probability feature matches from all feature matches using subnetwork connections
# input: possible feature matches, disconnected subnetworks of first dataset, disconnected subnetworks of second dataset, subnetwork connections
# output: high probability feature matches
feature_hp_12 = hpfeature_match(feature_all_12, network_list_1, network_list_2, network_match)

# obtain the lowess regression using high probability matches
# input: possible feature matches, high probability feature matches
# output: possible feature matches with lowess regression results
feature_all_12, feature_hp_12 = feature_match_regression(feature_all_12, feature_hp_12, interp='linear', frac_rt=0.1, frac_mz=0.1, frac_c=0.1)
feature_hp_12.to_excel('feature_hp_am.xlsx')

# obtain penalisation score
# input: possible feature matches, high probability feature matches, RT/MZ/FI weight of penalisation score, whether plot
# output: possible feature matches with penalisation score
feature_all_12 = all_match_penal(feature_all_12, feature_hp_12, 5, [1, 1, 0], plot=True)
feature_all_12.to_excel('feature_all_am.xlsx')

# reduce multiple matches
# input: possible feature matches
# output: single possible feature matches
feature_single_12 = reduce_multimatch_ps(feature_all_12)
feature_single_12.to_excel('feature_single_am.xlsx')

# delete poor matches
# input: single possible feature matches
# output: good feature matches
feature_good_12 = reduce_poormatch(feature_single_12, mad=5)

# plot good, poor and multiple match
plot_goodmatch(feature_all_12, feature_single_12, feature_good_12)

# validate each match by calculating a score
feature_good_12 = fmatch_eval(var_info_1, var_info_2, data_1, data_2, feature_good_12, d_rt_r=settings_1[1], d_rt_t=settings_2[1], corr_r=settings_1[2], corr_t=settings_2[2])
feature_good_12.to_excel('feature_good_am.xlsx')

# plot evaluation
plot_eval(feature_good_12)


# find carbon chain matches
# input: variable information, data, isotope and adduct type, mz threshold, rt threshold, correlation threshold, carbon chain mz difference
# output: all c2h4 matches with the same or unknown adduct type, all c2h4 matches with the same and known adduct type
c2h4_all_1, c2h4_strict_1 = carbonchain_match(var_info_1, data_1, iso_add_type_1, d_mz=settings_1[0], d_rt=1, d_corr=-1, chain=28.031300128)
plot_c2h4(var_info_1, c2h4_all_1, title='ref')

# delete the false positive c2h4 matches and plot c2h4 matches
# input: all c2h4 matches, regression model, regression model smooth parameter, accept range around regression line, whether plot
# output: true positive c2h4 matches
c2h4_1 = carbonchain_tp(c2h4_all_1, model='lowess', frac=0.1, mad=3, plot=True, title='ref_all')

c2h4_all = carbonchain_match_pm(feature_good_12, d_mz, d_rt, chain=28.031300128)
c2h4 = carbonchain_tp(c2h4_all, model='lowess', frac=0.1, mad=3, plot=True, title='ref_good')
