import igraph
from loess.loess_1d import loess_1d
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# find all isotope matches with large thresholds
def iso_match_all(var_info, data, d_mz, d_rt, d_corr):
    file = pd.read_excel(var_info, engine='openpyxl')
    name_mz = file['mzmed'].astype(str).str.split(".")
    name_rt = file['rtmed'].astype(str).str.split(".")
    file['MZRT_str'] = 'SLPOS_' + name_mz.str[0] + '.' + name_mz.str[1].str[:4] + '_' + name_rt.str[0] + '.' + name_rt.str[1].str[:4]
    var = file[['MZRT_str', 'mzmed', 'rtmed', 'fimed']]
    var = var.sort_values(by=['mzmed'])
    ind = var.index
    var = var.reset_index(drop=True)
    if data[-3:] == 'csv':
        data_pd = pd.read_csv(data, header=None)
    else:
        data_pd = pd.read_excel(data, header=None)
    iso_list = []
    for i in range(len(var) - 1):
        # set threshold
        rt_min = var.iloc[i, 2] - d_rt
        rt_max = var.iloc[i, 2] + d_rt
        mz_min = var.iloc[i, 1] + 1.003055 - d_mz
        mz_max = var.iloc[i, 1] + 1.003055 + d_mz
        for j in range(i + 1, len(var)):
            # mz threshold
            if mz_min <= var.iloc[j, 1] <= mz_max:
                # rt threshold
                if rt_min <= var.iloc[j, 2] <= rt_max:
                    data_1, data_2 = zip(*sorted(zip(list(data_pd.iloc[:, ind[i]]), list(data_pd.iloc[:, ind[j]]))))
                    corr_0, p_value = stats.pearsonr(data_1, data_2)
                    corr_1, p_value = stats.pearsonr(data_1[:round(len(data_1) / 2)], data_2[:round(len(data_1) / 2)])
                    corr_2, p_value = stats.pearsonr(data_1[round(len(data_1) / 2):], data_2[round(len(data_1) / 2):])
                    corr_3, p_value = stats.pearsonr(data_1[round(len(data_1) / 4):round(len(data_1) * 3 / 4)],
                                                     data_2[round(len(data_1) / 4):round(len(data_1) * 3 / 4)])
                    corr = max(corr_0, corr_1, corr_2, corr_3)
                    # corr threshold
                    if corr >= d_corr and var.iloc[i, 3] > var.iloc[j, 3]:
                        iso_list.append([var.iloc[i, 0], var.iloc[j, 0], corr, var.iloc[i, 1], var.iloc[j, 1],
                                         var.iloc[i, 2], var.iloc[j, 2], var.iloc[i, 3], var.iloc[j, 3]])
            elif var.iloc[j, 1] >= mz_max:
                break
    isotope_all = pd.DataFrame(iso_list, columns=['reference', 'target', 'correlation', 'mz_ref', 'mz_tar',
                                                  'rt_ref', 'rt_tar', 'fi_ref', 'fi_tar'])
    return isotope_all


# plot isotope networks
def plot_iso(iso_df):
    g = igraph.Graph.DataFrame(iso_df['reference', 'target', 'correlation'], directed=False)
    igraph.plot(g, './outfile/isotope_graph.png', vertex_size=3)


# get connected components dataframe
def connected_comp(iso_df):
    g = nx.from_pandas_edgelist(iso_df, 'reference', 'target')
    cc_list = list(nx.connected_components(g))
    # cc_df = pd.DataFrame({'isotope': cc_list})
    return cc_list


# classify the isotopic matches
# also accept false double isotopes, such as +0, +0, +1, +2, record the false double isotope matches as -1
def iso_classify(iso_df, w=1.003055):
    cc_list = connected_comp(iso_df)
    iso_class = []
    for i in range(len(cc_list)):
        cc = []
        for j in cc_list[i]:
            mzrt = iso_df[['mz_ref', 'rt_ref']][iso_df['reference'] == j]
            if not mzrt.empty:
                cc.append([j, mzrt.iloc[0, 0], mzrt.iloc[0, 1]])
            else:
                mzrt = iso_df[['mz_ref', 'rt_ref']][iso_df['target'] == j]
                cc.append([j, mzrt.iloc[0, 0], mzrt.iloc[0, 1]])
        cc.sort(key=lambda x: x[1])
        cc[0].append(0)
        for m in range(1, len(cc)):
            cc[m].append(int(np.round(cc[m][1]-cc[0][1])))
        cc_df = pd.DataFrame(cc, columns=['name', 'mz', 'rt', 'iso_type'])
        for k in range(int(cc_df.iloc[-1, 3])):
            cc_k_1 = cc_df[cc_df['iso_type'] == k].reset_index(drop=True)
            cc_k_2 = cc_df[cc_df['iso_type'] == k+1].reset_index(drop=True)
            for a in range(len(cc_k_1)):
                iso_match = []
                for b in range(len(cc_k_2)):
                    if np.sum(np.logical_and(iso_df['reference'] == cc_k_1.iloc[a, 0], iso_df['target'] == cc_k_2.iloc[b, 0])) >= 1:
                        iso_match.append([cc_k_1.iloc[a, 0], cc_k_2.iloc[b, 0], cc_k_1.iloc[a, 1], np.abs(cc_k_1.iloc[a, 1]-cc_k_2.iloc[b, 1]-w), cc_k_1.iloc[a, 2], cc_k_1.iloc[a, 3]])
                if len(iso_match) == 0:
                    continue
                if len(iso_match) > 1:
                    iso_match.sort(key=lambda x: x[3])
                    for n in range(1, len(iso_match)):
                        iso_class.append([iso_match[n][0], iso_match[n][1], iso_match[n][2], iso_match[n][4], 0, -1])
                    iso_match = [iso_match[0]]
                tars_match = iso_df[['reference', 'mz_ref', 'mz_tar']][iso_df['target'] == iso_match[0][1]]
                tars_match['mz_diff'] = np.abs(tars_match['mz_tar'] - tars_match['mz_ref'] - w)
                tars_match = tars_match.sort_values(by=['mz_diff']).reset_index(drop=True)
                if tars_match.iloc[0, 0] != iso_match[0][0]:
                    iso_class.append([iso_match[0][0], iso_match[0][1], iso_match[0][2], iso_match[n][4], 0, -1])
                    continue
                if len(iso_class) >= 1:
                    tars = list(list(zip(*iso_class))[1])
                else:
                    fi_r = iso_df['fi_tar'][iso_df['reference'] == iso_match[0][0]].iloc[0] / iso_df['fi_ref'][iso_df['reference'] == iso_match[0][0]].iloc[0]
                    iso_class.append([iso_match[0][0], iso_match[0][1], iso_match[0][2], iso_match[n][4], fi_r, 0])
                    continue
                if iso_match[0][0] in tars:
                    pre_type = iso_class[tars.index(iso_match[0][0])][5]
                    if pre_type != -1:
                        fi_r = iso_df['fi_tar'][iso_df['reference'] == iso_match[0][0]].iloc[0]/iso_df['fi_ref'][iso_df['reference'] == iso_match[0][0]].iloc[0]
                        iso_class.append([iso_match[0][0], iso_match[0][1], iso_match[0][2], iso_match[n][4], fi_r, pre_type+1])
                    else:
                        iso_class.append([iso_match[0][0], iso_match[0][1], iso_match[0][2], iso_match[n][4], 0, 0])
                else:
                    fi_r = iso_df['fi_tar'][iso_df['reference'] == iso_match[0][0]].iloc[0]/iso_df['fi_ref'][iso_df['reference'] == iso_match[0][0]].iloc[0]
                    iso_class.append([iso_match[0][0], iso_match[0][1], iso_match[0][2], iso_match[n][4], fi_r, 0])
    iso_type = pd.DataFrame(iso_class, columns=['reference', 'target', 'mz', 'rt', 'fi_ratio', 'iso_type'])
    return iso_type


# find the best line to fit each class
def fit_rlm(iso_type):
    gradients = []
    for i in range(min(max(iso_type['iso_type'])+1, 3)):
        mz = list(iso_type['mz'][iso_type['iso_type'] == i])
        fi_r = list(iso_type['fi_ratio'][iso_type['iso_type'] == i])
        if len(mz) > 1:
            model = sm.RLM(fi_r, mz, M=sm.robust.norms.HuberT()).fit()
            gradients.append(model.params[0])
        else:
            gradients.append(fi_r[0]/mz[0])
    return gradients


# calculate the std for each class
def std_class(iso_type):
    gradients = fit_rlm(iso_type)
    std = []
    for i in range(min(max(iso_type['iso_type'])+1, 3)):
        x = iso_type['mz'][iso_type['iso_type'] == i].to_numpy()
        y = iso_type['fi_ratio'][iso_type['iso_type'] == i].to_numpy()
        if len(x) > 1:
            std.append(np.sqrt(np.mean(((y - x * gradients[i]) * (np.sqrt(1 + (gradients[i]) ** 2))) ** 2)))
        else:
            # if there is only one type +2 connection, set std of type +2  = std of type +1 / 2
            std.append(std[-1]/2)
    return std


# calculate the number of incorrect labels with robust linear model or support vector machine
def count_incorrect(iso_type, model='rlm'):
    num_incorrect = 0
    if model == 'rlm':
        gradients = fit_rlm(iso_type)
        std = std_class(iso_type)
        for i in range(len(iso_type)):
            dis = []
            x = iso_type.iloc[i, 2]
            y = iso_type.iloc[i, 4]
            label = iso_type.iloc[i, 5]
            for j in range(min(max(iso_type['iso_type']) + 1, 3)):
                dis.append(np.abs((y - x * gradients[j]) * (np.sqrt(1 + (gradients[j]) ** 2))))
            dis = [a / b for a, b in zip(dis, std)]
            if dis.index(min(dis)) != label:
                num_incorrect += 1
    elif model == 'svm':
        for i in range(min(max(iso_type['iso_type']), 2)):
            x_1 = iso_type[['mz', 'fi_ratio']][iso_type['iso_type'] == i].to_numpy()
            x_2 = iso_type[['mz', 'fi_ratio']][iso_type['iso_type'] == i + 1].to_numpy()
            x = np.concatenate((x_1, x_2))
            y = np.array([i] * len(x_1) + [i + 1] * len(x_2))
            model = svm.LinearSVC(penalty='l2', fit_intercept=False, C=5, dual=False, class_weight='balanced')
            model.fit(x, y)
            pred = model.predict(x)
            num_incorrect += sum(pred != y)
    else:
        raise Exception("model name: 'rlm' or 'svm'")
    num_incorrect += len(iso_type[iso_type['iso_type'] == -1])
    return num_incorrect


# reassign class labels with robust linear model or support vector machine
def reassign_label(isotope_sub, model='rlm'):
    iso_type = iso_classify(isotope_sub)
    if model == 'rlm':
        gradients = fit_rlm(iso_type)
        std = std_class(iso_type)
        for i in range(len(isotope_sub)):
            dis = []
            x = iso_type.iloc[i, 2]
            y = iso_type.iloc[i, 4]
            label = iso_type.iloc[i, 5]
            for j in range(min(max(iso_type['iso_type']) + 1, 3)):
                dis.append(np.abs((y - x * gradients[j]) * (np.sqrt(1 + (gradients[j]) ** 2))))
            dis = [a / b for a, b in zip(dis, std)]
            if dis.index(min(dis)) != label:
                iso_type.iloc[i, 5] = dis.index(min(dis))
    elif model == 'svm':
        for i in range(min(max(iso_type['iso_type']), 2)):
            x_1 = iso_type[['mz', 'fi_ratio']][iso_type['iso_type'] == i].to_numpy()
            x_2 = iso_type[['mz', 'fi_ratio']][iso_type['iso_type'] == i + 1].to_numpy()
            x = np.concatenate((x_1, x_2))
            y = np.array([i for j in range(len(x_1))] + [i + 1 for j in range(len(x_2))])
            model = svm.LinearSVC(penalty='l2', fit_intercept=False, C=5, dual=False, class_weight='balanced')
            model.fit(x, y)
            pred = model.predict(x)
            iso_type['iso_type'][iso_type['iso_type'] == i] = pred[:len(x_1)]
            iso_type['iso_type'][iso_type['iso_type'] == i + 1] = pred[len(x_1):]
    else:
        raise Exception("model name: 'rlm' or 'svm'")
    return iso_type


# calculate the incorrect number and total number of isotopic matches with different thresholds
def iso_grid_pre(var_info, isotope_all, d_mz_list, d_rt_list, d_corr_list, model='rlm', reassign_err=False):
    file = pd.read_excel(var_info, engine='openpyxl')
    name_mz = file['mzmed'].astype(str).str.split(".")
    name_rt = file['rtmed'].astype(str).str.split(".")
    file['MZRT_str'] = 'SLPOS_' + name_mz.str[0] + '.' + name_mz.str[1].str[:4] + '_' + name_rt.str[0] + '.' + name_rt.str[1].str[:4]
    total_count = np.zeros((len(d_mz_list), len(d_rt_list), len(d_corr_list)))
    incorrect_count = np.zeros((len(d_mz_list), len(d_rt_list), len(d_corr_list)))
    for i in range(len(d_mz_list)):
        for j in range(len(d_rt_list)):
            for k in range(len(d_corr_list)):
                index_mz = (isotope_all['mz_tar']-isotope_all['mz_ref']).between(1.003055-d_mz_list[i], 1.003055+d_mz_list[i])
                index_rt = (isotope_all['rt_tar']-isotope_all['rt_ref']).between(-d_rt_list[j], d_rt_list[j])
                index_corr = isotope_all['correlation'].ge(d_corr_list[k])
                index = index_mz & index_rt & index_corr
                isotope_sublist = isotope_all[index].reset_index(drop=True)
                iso_type = iso_classify(isotope_sublist)
                iso_reassign = reassign_label(isotope_sublist, model=model)
                total_count[i, j, k] = len(isotope_sublist)
                incorrect_count[i, j, k] = count_incorrect(iso_type, model)
                if reassign_err:
                    var = file.loc[:, ['MZRT_str', 'mzmed', 'rtmed', 'fimed']].copy(deep=True)
                    var = var.sort_values(by=['mzmed'], ignore_index=True)
                    var['iso_type'] = 'Nan'
                    number_incorrect = 0
                    for m in range(len(iso_reassign)):
                        id_r = np.where(var.iloc[:, 0] == iso_reassign.iloc[m, 0])[0][0]
                        if var.iloc[id_r, 4] == 'Nan':
                            var.iloc[id_r, 4] = iso_reassign.iloc[m, 4]
                            id_t = np.where(var.iloc[:, 0] == iso_reassign.iloc[m, 1])[0][0]
                            var.iloc[id_t, 4] = iso_reassign.iloc[m, 4] + 1
                        elif var.iloc[id_r, 4] == iso_reassign.iloc[m, 4]:
                            id_t = np.where(var.iloc[:, 0] == iso_reassign.iloc[m, 1])[0][0]
                            var.iloc[id_t, 4] = iso_reassign.iloc[m, 4] + 1
                        else:
                            number_incorrect += 1
                    incorrect_count[i, j, k] += number_incorrect
    return incorrect_count, total_count


def plot_isomatch(isotope_all, model='rlm', title='all'):
    iso_all_type = iso_classify(isotope_all)
    for i in range(min(max(iso_all_type['iso_type']) + 1, 3)):
        x = list(iso_all_type['mz'][iso_all_type['iso_type'] == i])
        y = list(iso_all_type['fi_ratio'][iso_all_type['iso_type'] == i])
        sns.scatterplot(x, y)
        if model == 'rlm':
            gradients = fit_rlm(iso_all_type)
            plt.plot([0] + [x[-1]], [0] + [x[-1] * gradients[i]], 'r')
    plt.xlim([0, 1200])
    plt.ylim([0, 1])
    plt.title(title + 'isotope matches')
    plt.xlabel('m/z')
    plt.ylabel('fi_ratio')
    plt.savefig('./outfile/'+title+'_isotope_matches.png')
    plt.show()


# select the best setting of thresholds using cost function
# find the sublist and reassigned list of isotope matches with the best setting of thresholds
def best_threshold(isotope_all, incorrect_count, total_count, d_mz_list, d_rt_list, d_corr_list, i=1):
    log_ratio = np.log((incorrect_count + i) / total_count)
    index = np.unravel_index(np.argmin(log_ratio), log_ratio.shape)
    settings = [d_mz_list[index[0]], d_rt_list[index[1]], d_corr_list[index[2]]]
    index_mz = (isotope_all['mz_tar'] - isotope_all['mz_ref']).between(1.003055 - settings[0], 1.003055 + settings[0])
    index_rt = (isotope_all['rt_tar'] - isotope_all['rt_ref']).between(-settings[1], settings[1])
    index_corr = isotope_all['correlation'].ge(settings[2])
    index = index_mz & index_rt & index_corr
    isotope_sublist = isotope_all[index].reset_index(drop=True)
    iso_type = iso_classify(isotope_sublist)
    iso_reassign = reassign_label(isotope_sublist, model='rlm')
    return settings, iso_type, iso_reassign


# select the best setting of thresholds using linear regression
def best_threshold_1(isotope_all, incorrect_count, total_count, d_mz_list, d_rt_list, d_corr_list):
    reg = LinearRegression().fit(total_count.flatten().reshape((-1, 1)), incorrect_count.flatten())
    y = incorrect_count - reg.coef_[0] * total_count
    index = np.unravel_index(np.argmin(y), y.shape)
    settings = [d_mz_list[index[0]], d_rt_list[index[1]], d_corr_list[index[2]]]
    index_mz = (isotope_all['mz_tar'] - isotope_all['mz_ref']).between(1.003055 - settings[0], 1.003055 + settings[0])
    index_rt = (isotope_all['rt_tar'] - isotope_all['rt_ref']).between(-settings[1], settings[1])
    index_corr = isotope_all['correlation'].ge(settings[2])
    index = index_mz & index_rt & index_corr
    isotope_sublist = isotope_all[index].reset_index(drop=True)
    iso_type = iso_classify(isotope_sublist)
    iso_reassign = reassign_label(isotope_sublist, model='rlm')
    return settings, iso_type, iso_reassign


# obtain the list of all features with corresponding isotope type
def iso_final(var_info, iso_reassign):
    file = pd.read_excel(var_info, engine='openpyxl')
    name_mz = file['mzmed'].astype(str).str.split(".")
    name_rt = file['rtmed'].astype(str).str.split(".")
    file['MZRT_str'] = 'SLPOS_' + name_mz.str[0] + '.' + name_mz.str[1].str[:4] + '_' + name_rt.str[0] + '.' + \
                       name_rt.str[1].str[:4]
    var = file[['MZRT_str', 'mzmed', 'rtmed', 'fimed']].copy(deep=True)
    var = var.sort_values(by=['mzmed'], ignore_index=True)
    var['iso_type'] = 'Nan'
    for i in range(len(iso_reassign)):
        id_r = np.where(var.iloc[:, 0] == iso_reassign.iloc[i, 0])[0][0]
        if var.iloc[id_r, 4] == 'Nan':
            var.iloc[id_r, 4] = iso_reassign.iloc[i, 4]
            id_t = np.where(var.iloc[:, 0] == iso_reassign.iloc[i, 1])[0][0]
            var.iloc[id_t, 4] = iso_reassign.iloc[i, 4] + 1
        elif var.iloc[id_r, 4] == iso_reassign.iloc[i, 4]:
            id_t = np.where(var.iloc[:, 0] == iso_reassign.iloc[i, 1])[0][0]
            var.iloc[id_t, 4] = iso_reassign.iloc[i, 4] + 1
    var['iso_type'][var['iso_type'] == 'Nan'] = 0
    return var


# find all adduct matches within each layer of isotope
def add_match_all(var_info, data, add_info, isotope_type, settings):
    file = pd.read_excel(var_info, engine='openpyxl')
    var = file[['mzmed']].copy(deep=True)
    var = var.sort_values(by=['mzmed'])
    var_ind = var.index
    data_csv = pd.read_excel(data, header=None)
    add_df = pd.read_excel(add_info, engine='openpyxl').head(10)
    d_mz, d_rt, d_corr = settings
    add_list = []
    for j in range(max(isotope_type['iso_type'])+1):
        var_sub = isotope_type[isotope_type['iso_type'] == j]
        ind = var_ind[var_sub.index]
        var_sub = var_sub.reset_index(drop=True)
        for m in range(len(var_sub)-1):
            mz_r = var_sub.iloc[m, 1]
            rt_min = var_sub.iloc[m, 2] - d_rt
            rt_max = var_sub.iloc[m, 2] + d_rt
            for n in range(m+1, len(var_sub)):
                mz_t = var_sub.iloc[n, 1]
                if rt_min <= var_sub.iloc[n, 2] <= rt_max:
                    data_1, data_2 = zip(*sorted(zip(list(data_csv.iloc[:, ind[m]]), list(data_csv.iloc[:, ind[n]]))))
                    corr_0, p_value = stats.pearsonr(data_1, data_2)
                    corr_1, p_value = stats.pearsonr(data_1[:round(len(data_1) / 2)], data_2[:round(len(data_1) / 2)])
                    corr_2, p_value = stats.pearsonr(data_1[round(len(data_1) / 2):], data_2[round(len(data_1) / 2):])
                    corr_3, p_value = stats.pearsonr(data_1[round(len(data_1) / 4):round(len(data_1) * 3 / 4)],
                                                     data_2[round(len(data_1) / 4):round(len(data_1) * 3 / 4)])
                    corr = max(corr_0, corr_1, corr_2, corr_3)
                    if corr >= d_corr:
                        mm_r = (np.ones((len(add_df))) / add_df['oligomerIndex'] * (mz_r * add_df['chargeFactor'] - add_df['massToSubtract'] + add_df['massToAdd'])).to_numpy()
                        mm_t = (np.ones((len(add_df))) / add_df['oligomerIndex'] * (mz_t * add_df['chargeFactor'] - add_df['massToSubtract'] + add_df['massToAdd'])).to_numpy()
                        matrix_diff = np.abs(np.tile(mm_r, (10, 1)) - np.tile(mm_t, (10, 1)).transpose())
                        x, y = np.nonzero(matrix_diff <= d_mz)
                        for k in range(len(x)):
                            add_list.append([var_sub.iloc[m, 0], add_df.iloc[y[k], 0], var_sub.iloc[n, 0], add_df.iloc[x[k], 0],
                                             j, add_df.iloc[y[k], 0][2:]+'__'+add_df.iloc[x[k], 0][2:], mz_r, mz_t,
                                             var_sub.iloc[m, 2], var_sub.iloc[n, 2], corr])
    adduct_all = pd.DataFrame(add_list, columns=['reference', 'ref_add', 'target', 'tar_add',
                                                 'iso_type', 'add_type', 'mz_r', 'mz_t', 'rt_r', 'rt_t', 'correlation'])
    return adduct_all


# select valid adduct matches and obtain the dataframe of all features with corresponding isotope and adduct type
def add_final(isotope_type, add_df):
    frequency = add_df['add_type'].value_counts()
    adduct_match_ = pd.DataFrame(columns=['reference', 'ref_add', 'target', 'tar_add', 'iso_type', 'add_type',
                                         'mz_r', 'mz_t', 'rt_r', 'rt_t', 'correlation'])
    adduct_match = pd.DataFrame(columns=['reference', 'ref_add', 'target', 'tar_add', 'iso_type', 'add_type',
                                          'mz_r', 'mz_t', 'rt_r', 'rt_t', 'correlation'])
    isotope_type['add_type'] = 'Nan'
    ref = ''
    for i in range(len(add_df)):
        if add_df.iloc[i, 0] == ref:
            continue
        else:
            ref = add_df.iloc[i, 0]
            connections = add_df[add_df['reference']==ref].reset_index(drop=True)
            if len(connections) == 1:
                adduct_match_ = adduct_match_.append(connections)
            else:
                ref_type = connections['ref_add'].value_counts()
                if ref_type[0] == 1:
                    l = []
                    for j in range(len(connections)):
                        l.append(frequency[connections.iloc[j, 5]])
                    adduct_match_ = adduct_match_.append(connections.iloc[np.argmax(l)])
                elif len(ref_type)==1:
                    adduct_match_ = adduct_match_.append(connections)
                elif ref_type[0] > ref_type[1]:
                    connections = connections[connections['ref_add'] == ref_type.index[0]]
                    adduct_match_ = adduct_match_.append(connections)
                else:
                    # if two adduct type have the same number of adduct match for a reference feature
                    connections_0 = connections[connections['ref_add'] == ref_type.index[0]]
                    f0 = np.sum(frequency[connections_0['add_type']])
                    connections_1 = connections[connections['ref_add'] == ref_type.index[1]]
                    f1 = np.sum(frequency[connections_1['add_type']])
                    if f0 > f1:
                        adduct_match_ = adduct_match_.append(connections_0)
                    else:
                        adduct_match_ = adduct_match_.append(connections_1)
    adduct_match_ = adduct_match_.reset_index(drop=True)
    for i in range(len(adduct_match_)):
        if adduct_match_.iloc[i, 1] == adduct_match_.iloc[i, 3]:
            continue
        # check if the ref in this match appear in previous tar
        ref = adduct_match[adduct_match['target']==adduct_match_.iloc[i, 0]]
        x = 0
        for j in range(len(ref)):
            if ref.iloc[j, 3] != adduct_match_.iloc[i, 1]:
                x -= 1
                break
        if x < 0:
            continue
        reftar = adduct_match_[adduct_match_['target']==adduct_match_.iloc[i, 2]]
        y = 0
        for j in range(len(reftar)):
            if reftar.iloc[j, 3] != adduct_match_.iloc[i, 3]:
                if frequency[reftar.iloc[j, 5]] > frequency[adduct_match_.iloc[i, 5]]:
                    y -= 1
                    break
        if y < 0:
            continue
        tar = adduct_match_[adduct_match_['reference']==adduct_match_.iloc[i, 2]]
        z = 0
        for j in range(len(tar)):
            if tar.iloc[j, 1] != adduct_match_.iloc[i, 3]:
                if frequency[tar.iloc[j, 5]] > frequency[adduct_match_.iloc[i, 5]]:
                    z -= 1
                    break
        if z == 0:
            adduct_match = adduct_match.append(adduct_match_.iloc[i])
    for i in range(len(adduct_match)):
        id_r = np.where(isotope_type.iloc[:, 0] == adduct_match.iloc[i, 0])[0][0]
        if isotope_type.iloc[id_r, 5] == 'Nan':
            isotope_type.iloc[id_r, 5] = adduct_match.iloc[i, 1]
            id_t = np.where(isotope_type.iloc[:, 0] == adduct_match.iloc[i, 2])[0][0]
            isotope_type.iloc[id_t, 5] = adduct_match.iloc[i, 3]
        elif isotope_type.iloc[id_r, 5] == adduct_match.iloc[i, 1]:
            id_t = np.where(isotope_type.iloc[:, 0] == adduct_match.iloc[i, 2])[0][0]
            isotope_type.iloc[id_t, 5] = adduct_match.iloc[i, 3]
    return adduct_match, isotope_type


# plot adduct networks
def plot_add(add_df):
    for i in range(max(add_df['iso_type'])+1):
        adduct_i = add_df[add_df['iso_type'] == i]
        g = igraph.Graph.DataFrame(adduct_i[['reference', 'target']], directed=True)
        g.es["label"] = list(adduct_i['add_type'])
        igraph.plot(g, './outfile/adduct_graph_'+str(i)+'.png', bbox=((3.2-i)*1000, (3.2-i)*1000), vertex_size=5,
                    edge_arrow_size=1, edge_arrow_width=0.5, vertex_label_size=5)


# find carbon chain matches
def carbonchain_match(var_info, data, iso_add_type, d_mz, d_rt, d_corr=-1, chain=28.031300128):
    file = pd.read_excel(var_info, engine='openpyxl')
    data_csv = pd.read_excel(data, header=None)
    var_ind = file[['mzmed']].copy(deep=True).sort_values(by=['mzmed']).index
    c2h4_list = []
    c2h4_list_strict = []
    for j in range(max(iso_add_type['iso_type'])+1):
        var_sub = iso_add_type[iso_add_type['iso_type'] == j]
        ind = var_ind[var_sub.index]
        var_sub = var_sub.reset_index(drop=True)
        for m in range(len(var_sub)-1):
            mz_r = var_sub.iloc[m, 1]
            mz_min = mz_r + chain - d_mz
            mz_max = mz_r + chain + d_mz
            rt_r = var_sub.iloc[m, 2]
            for n in range(m+1, len(var_sub)):
                mz_t = var_sub.iloc[n, 1]
                rt_t = var_sub.iloc[n, 2]
                if mz_min <= mz_t <= mz_max:
                    if rt_r <= rt_t <= rt_r + d_rt:
                        data_1, data_2 = zip(*sorted(zip(list(data_csv.iloc[:, ind[m]]), list(data_csv.iloc[:, ind[n]]))))
                        corr_0, p_value = stats.pearsonr(data_1, data_2)
                        corr_1, p_value = stats.pearsonr(data_1[:round(len(data_1) / 2)], data_2[:round(len(data_1) / 2)])
                        corr_2, p_value = stats.pearsonr(data_1[round(len(data_1) / 2):], data_2[round(len(data_1) / 2):])
                        corr_3, p_value = stats.pearsonr(data_1[round(len(data_1) / 4):round(len(data_1) * 3 / 4)],
                                                         data_2[round(len(data_1) / 4):round(len(data_1) * 3 / 4)])
                        corr = max(corr_0, corr_1, corr_2, corr_3)
                        if corr >= d_corr:
                            if var_sub.iloc[m, 5] == var_sub.iloc[n, 5] and var_sub.iloc[m, 5]!='Nan':
                                c2h4_list_strict.append([var_sub.iloc[m, 0], var_sub.iloc[n, 0], var_sub.iloc[m, 1], var_sub.iloc[n, 1],
                                                  var_sub.iloc[m, 2], var_sub.iloc[n, 2],
                                                  corr, j, var_sub.iloc[m, 5]])
                            if var_sub.iloc[m, 5] == 'Nan' or var_sub.iloc[n, 5]=='Nan' or var_sub.iloc[m, 5]==var_sub.iloc[n, 5]:
                                c2h4_list.append([var_sub.iloc[m, 0], var_sub.iloc[n, 0], var_sub.iloc[m, 1], var_sub.iloc[n, 1],
                                                  var_sub.iloc[m, 2], var_sub.iloc[n, 2],
                                                  corr, j, var_sub.iloc[m, 5], var_sub.iloc[n, 5]])
    c2h4_all = pd.DataFrame(c2h4_list, columns=['reference', 'target', 'mz_ref', 'mz_tar', 'rt_ref', 'rt_tar',
                                                'correlation', 'iso_type', 'add_ref', 'add_tar'])
    c2h4_all_strict = pd.DataFrame(c2h4_list_strict, columns=['reference', 'target', 'mz_ref', 'mz_tar', 'rt_ref', 'rt_tar',
                                                              'correlation', 'iso_type', 'add_type'])
    return c2h4_all, c2h4_all_strict


# delete the false positive c2h4 matches and plot c2h4 matches
def carbonchain_tp(c2h4_all, model='lowess', frac=0.3, mad=3, plot=False):
    x = c2h4_all['rt_ref'].values
    y = (c2h4_all['rt_tar'] - c2h4_all['rt_ref']).values
    s = np.argsort(x)
    y = y[s]
    x = np.sort(x)
    if model=='lowess':
        lowess = sm.nonparametric.lowess
        z = lowess(y, x, is_sorted=True, frac=frac)
        xout = z[:, 0]
        yout = z[:, 1]
    elif model == 'loess':
        xout, yout, weigts = loess_1d(x, y, frac=frac)
    else:
        raise Exception("model name: 'lowess' or 'loess'")
    c2h4 = c2h4_all.iloc[s[np.abs(y - yout) <= stats.median_abs_deviation(y-yout) * mad]].sort_index()
    if plot:
        plt.scatter(x[np.abs(y - yout) > stats.median_abs_deviation(y-yout) * mad], y[np.abs(y - yout) > stats.median_abs_deviation(y-yout) * mad], s=5, c='orange', label='outliers')
        plt.scatter(x[np.abs(y - yout) <= stats.median_abs_deviation(y-yout) * mad], y[np.abs(y - yout) <= stats.median_abs_deviation(y-yout) * mad], s=5, c='b', label='c2h4')
        plt.plot(xout, yout, 'r')
        plt.xlabel('rt')
        plt.ylabel('rt difference')
        plt.legend()
        plt.title('c2h4 match')
    return c2h4


# match two datasets
# find all possible feature matches
def allfeature_match(iso_add_type_1, iso_add_type_2, d_mz=0.015, d_rt=0.1):
    feature_list = []
    feature_list_strict = []
    for i in range(len(iso_add_type_1)):
        # set threshold
        rt_min = iso_add_type_1.iloc[i, 2] - d_rt
        rt_max = iso_add_type_1.iloc[i, 2] + d_rt
        mz_min = iso_add_type_1.iloc[i, 1] - d_mz
        mz_max = iso_add_type_1.iloc[i, 1] + d_mz
        for j in range(len(iso_add_type_2)):
            # mz threshold
            if mz_min <= iso_add_type_2.iloc[j, 1] <= mz_max:
                # rt threshold
                if rt_min <= iso_add_type_2.iloc[j, 2] <= rt_max:
                    if iso_add_type_2.iloc[i, 4] == iso_add_type_2.iloc[j, 4]:
                        if iso_add_type_2.iloc[i, 5] == iso_add_type_2.iloc[j, 5] and iso_add_type_2.iloc[i, 5] != 'Nan':
                            feature_list_strict.append([iso_add_type_1.iloc[i, 0], iso_add_type_2.iloc[j, 0],
                                                 iso_add_type_1.iloc[i, 1], iso_add_type_2.iloc[j, 1],
                                                 iso_add_type_1.iloc[i, 2], iso_add_type_2.iloc[j, 2],
                                                 iso_add_type_1.iloc[i, 3], iso_add_type_2.iloc[j, 3],
                                                 iso_add_type_1.iloc[i, 4], iso_add_type_2.iloc[i, 5]])
                        if iso_add_type_2.iloc[i, 5] == iso_add_type_2.iloc[j, 5] or iso_add_type_2.iloc[i, 5] == 'Nan' or iso_add_type_2.iloc[j, 5] == 'Nan':
                            add_type = iso_add_type_2.iloc[i, 5] if iso_add_type_2.iloc[i, 5] != 'Nan' else iso_add_type_2.iloc[j, 5]
                            feature_list.append([iso_add_type_1.iloc[i, 0], iso_add_type_2.iloc[j, 0],
                                                 iso_add_type_1.iloc[i, 1], iso_add_type_2.iloc[j, 1],
                                                 iso_add_type_1.iloc[i, 2], iso_add_type_2.iloc[j, 2],
                                                 iso_add_type_1.iloc[i, 3], iso_add_type_2.iloc[j, 3],
                                                 iso_add_type_1.iloc[i, 4], add_type])
            elif iso_add_type_2.iloc[j, 1] >= mz_max:
                break
    feature_all = pd.DataFrame(feature_list, columns=['reference', 'target', 'mz_ref', 'mz_tar', 'rt_ref', 'rt_tar',
                                                      'fi_ref', 'fi_tar', 'iso_type', 'add_type'])
    feature_all_strict = pd.DataFrame(feature_list_strict, columns=['reference', 'target', 'mz_ref', 'mz_tar', 'rt_ref', 'rt_tar',
                                                                    'fi_ref', 'fi_tar', 'iso_type', 'add_type'])
    return feature_all, feature_all_strict


# find all disconnected subnetworks in a dataset
def all_networks(iso_add_type, isotope_match, adduct_match, single=False):
    network_list = connected_comp(isotope_match[['reference', 'target']].append(adduct_match[['reference', 'target']], ignore_index=True))
    if single:
        for i in range(len(iso_add_type)):
            if iso_add_type['MZRT_str'].iloc[i] in isotope_match[['reference', 'target']].values or iso_add_type['MZRT_str'].iloc[i] in adduct_match[['reference', 'target']].values:
                pass
            else:
                network_list.append({iso_add_type['MZRT_str'].iloc[i]})
    return network_list


# merge subnetworks if their correlation between every node of the subnetworks is high and rt in the threshold
# todo: merge two networks
def merge_networks(network_list):
    for i in range(len(network_list)-1):
        for j in range(i+1, len(network_list)):
            pass
    return


# todo: add mean rt diff column
# calculate the number of connections between each pair of networks
# return the indices of matching network in dataset 2 for each network in dataset 1
# keep two matches in the second dataset if rt difference is less than d_rt and no overlap match
# multi_match: whether allow more than one subnetworks in dataset B connect to a subnetwork in dataset A
# perfect_match: for a subnetwork in dataset A, it connects to a subnetwork b in dataset B if b has the most number of matches;
#                or it connects to more than one subnetworks b1 and b2 in dataset B if b1 and b2 have the same number of connections and no intersection.
# least_match: the least number of feature matches in a subnetwork connection
def network_connection(feature_all, nl_1, nl_2, d_rt, multi_match=False, least_match=2, perfect_match=False):
    network_match = []
    for i in range(len(nl_1)):
        network_score = np.zeros(len(nl_2))
        network_mzdiff = np.zeros(len(nl_2))
        network_meanrt = np.zeros(len(nl_2))
        network_c = [[] for _ in range(len(nl_2))]
        for j in range(len(nl_1[i])):
            connection_d2 = feature_all['target'][feature_all['reference'] == list(nl_1[i])[j]]
            if not connection_d2.empty:
                for m in range(len(nl_2)):
                    for k in range(len(connection_d2)):
                        if connection_d2.iloc[k] in nl_2[m]:
                            network_score[m] += 1
                            network_mzdiff[m] += np.abs((feature_all['mz_ref']-feature_all['mz_tar'])[np.logical_and(
                                feature_all['reference'] == list(nl_1[i])[j], feature_all['target'] == connection_d2.iloc[k])].iloc[0])
                            network_meanrt[m] += (feature_all['rt_tar'])[np.logical_and(feature_all['reference'] == list(nl_1[i])[j],
                                                                                        feature_all['target'] == connection_d2.iloc[k])].iloc[0]
                            network_c[m].append(j)
        network_ref = []
        network_rt = []
        while np.amax(network_score) >= least_match:
            cc2_id = np.where(network_score == np.amax(network_score))[0]
            network_mzdiff = network_mzdiff / (network_score + 0.000001)
            network_meanrt = network_meanrt / (network_score + 0.000001)
            if len(cc2_id) == 1:
                tar_n = cc2_id[0]
            elif perfect_match and len(list(set(network_c[cc2_id[0]]).intersection(network_c[cc2_id[1]]))) > 0:
                break
            else:
                tar_n = cc2_id[np.argmin(network_mzdiff[cc2_id])]
            if any(network_c[tar_n]) in network_ref:
                continue
            network_rt.append(network_meanrt[tar_n])
            if np.abs(network_meanrt[tar_n] - network_rt[0]) <= d_rt:
                network_ref.append(_ for _ in network_c)
                network_match.append([i, tar_n, network_score[tar_n], len(nl_1[i]), len(nl_2[tar_n]), network_mzdiff[tar_n]])
                if not multi_match:
                    break
            network_score[tar_n] = 0
    network_match_df = pd.DataFrame(network_match, columns=['ref_net', 'tar_net', 'n_match', 'n_node_1', 'n_node_2', 'mean_mz_diff'])
    return network_match_df


# select high probability feature matches from all feature matches using subnetwork connections
def hpfeature_match(feature_all, network_list_1, network_list_2, network_match_df):
    feature_match = []
    for i in range(len(feature_all)):
        refnet = [j for j, s in enumerate(network_list_1) if feature_all.iloc[i, 0] in s]
        tarnet = [j for j, s in enumerate(network_list_2) if feature_all.iloc[i, 1] in s]
        if len(refnet) == 1 and len(tarnet) == 1:
            refind = network_match_df.index[network_match_df['ref_net'] == refnet[0]].tolist()
            tarind = network_match_df.index[network_match_df['tar_net'] == tarnet[0]].tolist()
            if len(refind) > 0 and len(tarind) > 0:
                if refind[0] in tarind:
                    feature_match.append(feature_all.iloc[i].append(network_match_df.iloc[refind[0]]))
    feature_hp_df = pd.DataFrame(feature_match, columns=['reference', 'target', 'mz_ref', 'mz_tar', 'rt_ref', 'rt_tar',
                                                            'fi_ref', 'fi_tar', 'iso_type', 'add_type', 'ref_net', 'tar_net',
                                                            'n_match', 'n_node_1', 'n_node_2', 'mean_mz_diff'])
    return feature_hp_df


# add rt, mz and fi regression results to feature match table
def feature_match_regression(feature_all, feature_hp):
    x_rt = feature_hp['rt_ref'].values
    y_rt = (feature_hp['rt_tar']-feature_hp['rt_ref']).values
    y_rt = y_rt[np.argsort(x_rt)]
    x_rt = np.sort(x_rt)
    lowess = sm.nonparametric.lowess
    z = lowess(y_rt, x_rt, is_sorted=True, frac=0.2)
    xout_rt = z[:, 0]
    yout_rt = z[:, 1]
    f_rt = interp1d(xout_rt, yout_rt, bounds_error=False, fill_value="extrapolate")
    x_rt = feature_hp['rt_ref'].values
    feature_hp['rt_reg'] = f_rt(x_rt).tolist()
    x_all_rt = feature_all['rt_ref'].values
    feature_all['rt_reg'] = f_rt(x_all_rt).tolist()

    x_mz = feature_hp['mz_ref'].values
    y_mz = (feature_hp['mz_tar']-feature_hp['mz_ref']).values
    y_mz = y_mz[np.argsort(x_mz)]
    x_mz = np.sort(x_mz)
    lowess = sm.nonparametric.lowess
    z = lowess(y_mz, x_mz, is_sorted=True, frac=0.2)
    xout_mz = z[:, 0]
    yout_mz = z[:, 1]
    f_mz = interp1d(xout_mz, yout_mz, bounds_error=False, fill_value="extrapolate")
    x_mz = feature_hp['mz_ref'].values
    feature_hp['mz_reg'] = f_mz(x_mz).tolist()
    x_all_mz = feature_all['mz_ref'].values
    feature_all['mz_reg'] = f_mz(x_all_mz).tolist()

    x_fi = np.log10(feature_hp['fi_ref'].values)
    y_fi = np.log10(np.abs((feature_hp['fi_ref'] - feature_hp['fi_tar']).values))
    y_fi = y_fi[np.argsort(x_fi)]
    x_fi = np.sort(x_fi)
    lowess = sm.nonparametric.lowess
    z = lowess(y_fi, x_fi, is_sorted=True, frac=0.1)
    xout_fi = z[:, 0]
    yout_fi = z[:, 1]
    feature_hp['fi_reg'] = yout_fi.tolist()
    f_fi = interp1d(xout_fi, yout_fi, bounds_error=False, fill_value="extrapolate")
    x_fi = np.log10(feature_hp['fi_ref'].values)
    feature_hp['fi_reg'] = f_fi(x_fi).tolist()
    x_all_fi = np.log10(feature_all['fi_ref'].values)
    feature_all['fi_reg'] = f_fi(x_all_fi).tolist()

    return feature_all, feature_hp


# plot step 1 to 3 for rt, mz and fi
def all_match_penal(feature_all, feature_hp, mad=5, w=None, plot=True):
    if w is None:
        w = [1, 1, 0]
    x_rt = feature_hp['rt_ref'].values
    y_rt = (feature_hp['rt_tar'] - feature_hp['rt_ref']).values
    y_rt_reg = feature_hp['rt_reg'].values
    y_rt_reg_ = y_rt_reg[np.argsort(x_rt)]
    x_rt_ = np.sort(x_rt)
    x_all_rt = feature_all['rt_ref'].values
    y_all_rt = (feature_all['rt_tar'] - feature_all['rt_ref']).values
    y_all_rt_reg = feature_all['rt_reg'].values
    y_rt_c = y_rt - y_rt_reg
    y_all_rt_c = y_all_rt - y_all_rt_reg
    y_rt_s = y_rt_c / (stats.median_abs_deviation(y_all_rt_c) * mad)
    y_all_rt_s = y_all_rt_c / (stats.median_abs_deviation(y_all_rt_c) * mad)

    x_mz = feature_hp['mz_ref'].values
    y_mz = (feature_hp['mz_tar'] - feature_hp['mz_ref']).values
    y_mz_reg = feature_hp['mz_reg'].values
    y_mz_reg_ = y_mz_reg[np.argsort(x_mz)]
    x_mz_ = np.sort(x_mz)
    x_all_mz = feature_all['mz_ref'].values
    y_all_mz = (feature_all['mz_tar'] - feature_all['mz_ref']).values
    y_all_mz_reg = feature_all['mz_reg'].values
    y_mz_c = y_mz - y_mz_reg
    y_all_mz_c = y_all_mz - y_all_mz_reg
    y_mz_s = y_mz_c / (stats.median_abs_deviation(y_all_mz_c) * mad)
    y_all_mz_s = y_all_mz_c / (stats.median_abs_deviation(y_all_mz_c) * mad)

    x_fi = np.log10(np.abs(feature_hp['fi_ref'].values))
    y_fi = np.log10(np.abs(feature_hp['fi_tar'].values - feature_hp['fi_ref'].values))
    y_fi_reg = feature_hp['fi_reg'].values
    y_fi_reg_ = y_fi_reg[np.argsort(x_fi)]
    x_fi_ = np.sort(x_fi)
    x_all_fi = np.log10(np.abs(feature_all['fi_ref'].values))
    y_all_fi = np.log10(np.abs((feature_all['fi_tar'] - feature_all['fi_ref']).values))
    y_all_fi_reg = feature_all['fi_reg'].values
    y_fi_c = y_fi - y_fi_reg
    y_all_fi_c = y_all_fi - y_all_fi_reg
    y_fi_s = y_fi_c / (stats.median_abs_deviation(y_all_fi_c) * mad)
    y_all_fi_s = y_all_fi_c / (stats.median_abs_deviation(y_all_fi_c) * mad)

    penalisation = np.sqrt(w[0] * np.square(y_all_rt_s) + w[1] * np.square(y_all_mz_s) + w[2] * np.square(y_all_fi_s))
    feature_all['penalisation'] = penalisation.tolist()

    if plot:
        plt.style.use('seaborn-darkgrid')
        fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(7, 7))

        # step 0
        axs[0, 0].scatter(x_all_rt, y_all_rt, s=2, c='black')
        axs[0, 0].set_xlabel('RT ref', fontsize=8)
        axs[0, 0].set_ylabel('RT diff', fontsize=8)
        axs[0, 0].tick_params(labelsize=7)
        axs[0, 1].scatter(x_all_mz, y_all_mz, s=2, c='black')
        axs[0, 1].set_xlabel('MZ ref', fontsize=8)
        axs[0, 1].set_ylabel('MZ diff', fontsize=8)
        axs[0, 1].tick_params(labelsize=7)
        axs[0, 2].scatter(x_all_fi, y_all_fi, s=2, c='black')
        axs[0, 2].set_xlabel('log10 FI ref', fontsize=8)
        axs[0, 2].set_ylabel('log10 FI diff', fontsize=8)
        axs[0, 2].tick_params(labelsize=7)

        # step 1
        axs[1, 0].scatter(x_all_rt, y_all_rt, s=2, c='black', alpha=0.8, label='other match')
        axs[1, 0].scatter(x_rt, y_rt, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[1, 0].plot(x_rt_, y_rt_reg_, 'r')
        axs[1, 0].set_xlabel('RT ref', fontsize=8)
        axs[1, 0].set_ylabel('RT diff', fontsize=8)
        axs[1, 0].tick_params(labelsize=7)
        axs[1, 1].scatter(x_all_mz, y_all_mz, s=2, c='black', alpha=0.8, label='other match')
        axs[1, 1].scatter(x_mz, y_mz, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[1, 1].plot(x_mz_, y_mz_reg_, 'r')
        axs[1, 1].set_xlabel('MZ ref', fontsize=8)
        axs[1, 1].set_ylabel('MZ diff', fontsize=8)
        axs[1, 1].tick_params(labelsize=7)
        axs[1, 2].scatter(x_all_fi, y_all_fi, s=2, c='black', alpha=0.8, label='other match')
        axs[1, 2].scatter(x_fi, y_fi, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[1, 2].plot(x_fi_, y_fi_reg_, 'r')
        axs[1, 2].set_xlabel('log10 FI ref', fontsize=8)
        axs[1, 2].set_ylabel('log10 FI diff', fontsize=8)
        axs[1, 2].tick_params(labelsize=7)

        # step 2
        axs[2, 0].scatter(x_all_rt, y_all_rt_c, s=2, c='black', alpha=0.8, label='other match')
        axs[2, 0].scatter(x_rt, y_rt_c, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[2, 0].set_xlabel('RT ref', fontsize=8)
        axs[2, 0].set_ylabel('centered RT diff', fontsize=8)
        axs[2, 0].tick_params(labelsize=7)
        axs[2, 1].scatter(x_all_mz, y_all_mz_c, s=2, c='black', alpha=0.8, label='other match')
        axs[2, 1].scatter(x_mz, y_mz_c, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[2, 1].set_xlabel('MZ ref', fontsize=8)
        axs[2, 1].set_ylabel('centered MZ diff', fontsize=8)
        axs[2, 1].tick_params(labelsize=7)
        axs[2, 2].scatter(x_all_fi, y_all_fi_c, s=2, c='black', alpha=0.8, label='other match')
        axs[2, 2].scatter(x_fi, y_fi_c, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[2, 2].set_xlabel('log10 FI ref', fontsize=8)
        axs[2, 2].set_ylabel('centered log10 FI diff', fontsize=8)
        axs[2, 2].tick_params(labelsize=7)

        # step 3
        axs[3, 0].scatter(x_all_rt, y_all_rt_s, s=2, c='black', alpha=0.8, label='other match')
        axs[3, 0].scatter(x_rt, y_rt_s, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[3, 0].plot([np.min(x_all_rt), np.max(x_all_rt)], [1, 1], 'r')
        axs[3, 0].plot([np.min(x_all_rt), np.max(x_all_rt)], [-1, -1], 'r')
        axs[3, 0].set_ylim([-4, 4])
        axs[3, 0].set_xlabel('RT ref', fontsize=8)
        axs[3, 0].set_ylabel('standardised RT diff', fontsize=8)
        axs[3, 0].tick_params(labelsize=7)
        axs[3, 1].scatter(x_all_mz, y_all_mz_s, s=2, c='black', alpha=0.8, label='other match')
        axs[3, 1].scatter(x_mz, y_mz_s, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[3, 1].plot([np.min(x_all_mz), np.max(x_all_mz)], [1, 1], 'r')
        axs[3, 1].plot([np.min(x_all_mz), np.max(x_all_mz)], [-1, -1], 'r')
        axs[3, 1].set_ylim([-4, 4])
        axs[3, 1].set_xlabel('MZ ref', fontsize=8)
        axs[3, 1].set_ylabel('standardised MZ diff', fontsize=8)
        axs[3, 1].tick_params(labelsize=7)
        axs[3, 2].scatter(x_all_fi, y_all_fi_s, s=2, c='black', alpha=0.8, label='other match')
        axs[3, 2].scatter(x_fi, y_fi_s, s=2, c='green', alpha=0.8, label='high possibility match')
        axs[3, 2].plot([np.min(x_all_fi), np.max(x_all_fi)], [1, 1], 'r')
        axs[3, 2].plot([np.min(x_all_fi), np.max(x_all_fi)], [-1, -1], 'r')
        axs[3, 2].set_ylim([-4, 4])
        axs[3, 2].set_xlabel('log10 FI ref', fontsize=8)
        axs[3, 2].set_ylabel('standardised log10 FI diff', fontsize=8)
        axs[3, 2].tick_params(labelsize=7)

        # step 4
        im = axs[4, 0].scatter(x_all_rt, y_all_rt, s=2, c=penalisation, cmap='coolwarm', vmin=0)
        axs[4, 0].set_xlabel('RT ref', fontsize=8)
        axs[4, 0].set_ylabel('RT diff', fontsize=8)
        axs[4, 0].tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=axs[4, 0], ticks=[0, 1, 2, 3, 4])
        cb.ax.tick_params(labelsize='7')
        im = axs[4, 1].scatter(x_all_mz, y_all_mz, s=2, c=penalisation, cmap='coolwarm', vmin=0)
        axs[4, 1].set_xlabel('MZ ref', fontsize=8)
        axs[4, 1].set_ylabel('MZ diff', fontsize=8)
        axs[4, 1].tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=axs[4, 1], ticks=[0, 1, 2, 3, 4])
        cb.ax.tick_params(labelsize='7')
        im = axs[4, 2].scatter(x_all_fi, y_all_fi, s=2, c=penalisation, cmap='coolwarm', vmin=0)
        axs[4, 2].set_xlabel('log10 FI ref', fontsize=8)
        axs[4, 2].set_ylabel('log10 FI diff', fontsize=8)
        axs[4, 2].tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=axs[4, 2], ticks=[0, 1, 2, 3, 4])
        cb.ax.tick_params(labelsize='7')

        fig.tight_layout()
        fig.show()
        plt.savefig('./outfig/workflow.png')
    return feature_all


def reduce_multimatch_ps(feature_all):
    feature_single = pd.DataFrame(columns=list(feature_all.columns))
    feature_single_ = pd.DataFrame(columns=list(feature_all.columns))
    for i in range(len(feature_all)):
        if feature_all.iloc[i, 0] not in list(feature_single_['reference']):
            mm = feature_all[feature_all['reference'] == feature_all.iloc[i, 0]]
            feature_single_ = feature_single_.append(mm.iloc[mm['penalisation'].argmin()])
    for j in range(len(feature_single_)):
        if feature_all.iloc[j, 1] not in list(feature_single['target']):
            mm = feature_all[feature_all['target'] == feature_all.iloc[j, 1]]
            feature_single = feature_single.append(mm.iloc[mm['penalisation'].argmin()])
    return feature_single


def reduce_poormatch(feature_single, mad=5):
    feature_good = feature_single[feature_single['penalisation'] <= stats.median_abs_deviation(feature_single['penalisation']) * mad]
    return feature_good


def plot_goodmatch(feature_all, feature_single, feature_good):
    x_all_rt = feature_all['rt_ref'].values
    y_all_rt = (feature_all['rt_tar'] - feature_all['rt_ref']).values
    x_s_rt = feature_single['rt_ref'].values
    y_s_rt = (feature_single['rt_tar'] - feature_single['rt_ref']).values
    x_g_rt = feature_good['rt_ref'].values
    y_g_rt = (feature_good['rt_tar'] - feature_good['rt_ref']).values

    x_all_mz = feature_all['mz_ref'].values
    y_all_mz = (feature_all['mz_tar'] - feature_all['mz_ref']).values
    x_s_mz = feature_single['mz_ref'].values
    y_s_mz = (feature_single['mz_tar'] - feature_single['mz_ref']).values
    x_g_mz = feature_good['mz_ref'].values
    y_g_mz = (feature_good['mz_tar'] - feature_good['mz_ref']).values

    x_all_fi = np.log10(np.abs(feature_all['fi_ref'].values))
    y_all_fi = np.log10(np.abs(feature_all['fi_tar'].values - feature_all['fi_ref'].values))
    x_s_fi = np.log10(np.abs(feature_single['fi_ref'].values))
    y_s_fi = np.log10(np.abs(feature_single['fi_tar'].values - feature_single['fi_ref'].values))
    x_g_fi = np.log10(np.abs(feature_good['fi_ref'].values))
    y_g_fi = np.log10(np.abs(feature_good['fi_tar'].values - feature_good['fi_ref'].values))

    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(7, 7/5))

    axs[0].scatter(x_all_rt, y_all_rt, s=2, c='blue', label='multi match')
    axs[0].scatter(x_s_rt, y_s_rt, s=2, c='orange', label='poor match')
    axs[0].scatter(x_g_rt, y_g_rt, s=2, c='black', label='good match')
    axs[0].set_xlabel('RT ref', fontsize=8)
    axs[0].set_ylabel('RT diff', fontsize=8)
    axs[0].tick_params(labelsize=7)

    axs[1].scatter(x_all_mz, y_all_mz, s=2, c='blue', label='other match')
    axs[1].scatter(x_s_mz, y_s_mz, s=2, c='orange', label='poor match')
    axs[1].scatter(x_g_mz, y_g_mz, s=2, c='black', label='good match')
    axs[1].set_xlabel('MZ ref', fontsize=8)
    axs[1].set_ylabel('MZ diff', fontsize=8)
    axs[1].tick_params(labelsize=7)

    axs[2].scatter(x_all_fi, y_all_fi, s=2, c='blue', label='other match')
    axs[2].scatter(x_s_fi, y_s_fi, s=2, c='orange', label='poor match')
    axs[2].scatter(x_g_fi, y_g_fi, s=2, c='black', label='good match')
    axs[2].set_xlabel('log10 FI ref', fontsize=8)
    axs[2].set_ylabel('log10 FI diff', fontsize=8)
    axs[2].tick_params(labelsize=7)

    fig.tight_layout()
    fig.show()
    plt.savefig('./outfig/good_match.png')