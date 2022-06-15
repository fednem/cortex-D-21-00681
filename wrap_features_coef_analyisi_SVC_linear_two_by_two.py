# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:36:10 2021

@author: federico.nemmi
"""
import pickle as pkl
import numpy as np
import scipy.stats
# from neurocombat_sklearn import CombatModel
import pandas as pd

with open('SVC_linear_upsampling_results_with_coef_two_by_two_03062021.pkl', 'rb') as f:
    gm_upsampling_groupwise_accuracy_bal_dcd_vs_dys, gm_upsampling_shuff_groupwise_acc_dcd_vs_dys, gm_upsampling_feat_coef_dcd_vs_dys,\
    wm_upsampling_groupwise_accuracy_bal_dcd_vs_dys, wm_upsampling_shuff_groupwise_acc_dcd_vs_dys, wm_upsampling_feat_coef_dcd_vs_dys,\
    falff_upsampling_groupwise_accuracy_bal_dcd_vs_dys, falff_upsampling_shuff_groupwise_acc_dcd_vs_dys, falff_upsampling_feat_coef_dcd_vs_dys,\
    localcorr_upsampling_groupwise_accuracy_bal_dcd_vs_dys, localcorr_upsampling_shuff_groupwise_acc_dcd_vs_dys, localcorr_upsampling_feat_coef_dcd_vs_dys,\
    globalcorr_upsampling_groupwise_accuracy_bal_dcd_vs_dys, globalcorr_upsampling_shuff_groupwise_acc_dcd_vs_dys, globalcorr_upsampling_feat_coef_dcd_vs_dys,\
    structural_upsampling_groupwise_accuracy_bal_dcd_vs_dys, structural_upsampling_shuff_groupwise_acc_dcd_vs_dys, structural_upsampling_feat_coef_dcd_vs_dys,\
    functional_upsampling_groupwise_accuracy_bal_dcd_vs_dys, functional_upsampling_shuff_groupwise_acc_dcd_vs_dys, functional_upsampling_feat_coef_dcd_vs_dys,\
    complete_upsampling_groupwise_accuracy_bal_dcd_vs_dys, complete_upsampling_shuff_groupwise_acc_dcd_vs_dys, complete_upsampling_feat_coef_dcd_vs_dys,\
    gm_upsampling_groupwise_accuracy_bal_dys_vs_com, gm_upsampling_shuff_groupwise_acc_dys_vs_com, gm_upsampling_feat_coef_dys_vs_com,\
    wm_upsampling_groupwise_accuracy_bal_dys_vs_com, wm_upsampling_shuff_groupwise_acc_dys_vs_com, wm_upsampling_feat_coef_dys_vs_com,\
    falff_upsampling_groupwise_accuracy_bal_dys_vs_com, falff_upsampling_shuff_groupwise_acc_dys_vs_com, falff_upsampling_feat_coef_dys_vs_com,\
    localcorr_upsampling_groupwise_accuracy_bal_dys_vs_com, localcorr_upsampling_shuff_groupwise_acc_dys_vs_com, localcorr_upsampling_feat_coef_dys_vs_com,\
    globalcorr_upsampling_groupwise_accuracy_bal_dys_vs_com, globalcorr_upsampling_shuff_groupwise_acc_dys_vs_com, globalcorr_upsampling_feat_coef_dys_vs_com,\
    structural_upsampling_groupwise_accuracy_bal_dys_vs_com, structural_upsampling_shuff_groupwise_acc_dys_vs_com, structural_upsampling_feat_coef_dys_vs_com,\
    functional_upsampling_groupwise_accuracy_bal_dys_vs_com, functional_upsampling_shuff_groupwise_acc_dys_vs_com, functional_upsampling_feat_coef_dys_vs_com,\
    complete_upsampling_groupwise_accuracy_bal_dys_vs_com, complete_upsampling_shuff_groupwise_acc_dys_vs_com, complete_upsampling_feat_coef_dys_vs_com,\
    gm_upsampling_groupwise_accuracy_bal_dcd_vs_com, gm_upsampling_shuff_groupwise_acc_dcd_vs_com, gm_upsampling_feat_coef_dcd_vs_com,\
    wm_upsampling_groupwise_accuracy_bal_dcd_vs_com, wm_upsampling_shuff_groupwise_acc_dcd_vs_com, wm_upsampling_feat_coef_dcd_vs_com,\
    falff_upsampling_groupwise_accuracy_bal_dcd_vs_com, falff_upsampling_shuff_groupwise_acc_dcd_vs_com, falff_upsampling_feat_coef_dcd_vs_com,\
    localcorr_upsampling_groupwise_accuracy_bal_dcd_vs_com, localcorr_upsampling_shuff_groupwise_acc_dcd_vs_com, localcorr_upsampling_feat_coef_dcd_vs_com,\
    globalcorr_upsampling_groupwise_accuracy_bal_dcd_vs_com, globalcorr_upsampling_shuff_groupwise_acc_dcd_vs_com, globalcorr_upsampling_feat_coef_dcd_vs_com,\
    structural_upsampling_groupwise_accuracy_bal_dcd_vs_com, structural_upsampling_shuff_groupwise_acc_dcd_vs_com, structural_upsampling_feat_coef_dcd_vs_com,\
    functional_upsampling_groupwise_accuracy_bal_dcd_vs_com, functional_upsampling_shuff_groupwise_acc_dcd_vs_com, functional_upsampling_feat_coef_dcd_vs_com,\
    complete_upsampling_groupwise_accuracy_bal_dcd_vs_com, complete_upsampling_shuff_groupwise_acc_dcd_vs_com, complete_upsampling_feat_coef_dcd_vs_com = pkl.load(f)



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
#average the coefficient over the folder
#complete
average_2_by_2_coef = complete_feat_coef.mean(2)
#the output of the functions are assigned along the third dimension of the output array
#so the axis are (comparisons, features, output)
dcd_dys_coef = np.apply_along_axis(mean_confidence_interval, 2, complete_upsampling_feat_coef_dcd_vs_dys)
dcd_dys_coef = dcd_dys_coef[1,:,:]
dys_com_coef = np.apply_along_axis(mean_confidence_interval, 2, complete_upsampling_feat_coef_dys_vs_com)
dys_com_coef = dys_com_coef[1,:,:]
dcd_com_coef = np.apply_along_axis(mean_confidence_interval, 2, complete_upsampling_feat_coef_dcd_vs_com)
dcd_com_coef = dcd_com_coef[1,:,:]

#####READ DATA#####
complete_included = pd.read_csv("D:/corona2020/DysTacMap_Data/complete_naming_data.csv")
cmbt = CombatModel()

##complete
complete_included = complete_included[complete_included.group != 1]
complete_values = complete_included.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]]
complete_cmbt = cmbt.fit_transform(complete_values,
                                   complete_included["centre"].values.reshape(-1,1),
                                   complete_included.loc[:,["group", "demographic_sex"]].values,
                                   complete_included.loc[:,["demographic_age"]])
complete_final = complete_included.copy()
complete_final.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]] = complete_cmbt

complete_features_name = ['falffxx-9_-81_-39', 'falffxx40_15_32', 'localcorrxx-24_-49_-48', 'globalcorrxx39_-61_50', 'globalcorrxx25_29_48',
       'gmxx15_24_57', 'gmxx-27_15_6', 'gmxx64_-15_2', 'gmxx36_10_0', 'gmxx32_-18_3', 'wmxx15_21_52',
       'wmxx-42_-56_-38']
coef_means = pd.DataFrame(np.concatenate((dcd_dys_coef[:,0], dys_com_coef[:,0], dcd_com_coef[:,0])), columns = ["median"])
coef_means["features"] = complete_features_name * 3
coef_means["discriminations"] = ["dcd_vs_dys"] * len(complete_features_name) +\
                                ["dys_vs_com"] * len(complete_features_name) +\
                                ["dcd_vs_com"] * len(complete_features_name)
coef_lower_ci = pd.DataFrame(np.concatenate((dcd_dys_coef[:,1], dys_com_coef[:,1], dcd_com_coef[:,1])), columns = ["median"])
coef_lower_ci["features"] = complete_features_name * 3
coef_lower_ci["discriminations"] = ["dcd_vs_dys"] * len(complete_features_name) +\
                                ["dys_vs_com"] * len(complete_features_name) +\
                                ["dcd_vs_com"] * len(complete_features_name)

coef_upper_ci = pd.DataFrame(np.concatenate((dcd_dys_coef[:,2], dys_com_coef[:,2], dcd_com_coef[:,2])), columns = ["median"])
coef_upper_ci["features"] = complete_features_name * 3
coef_upper_ci["discriminations"] = ["dcd_vs_dys"] * len(complete_features_name) +\
                                ["dys_vs_com"] * len(complete_features_name) +\
                                ["dcd_vs_com"] * len(complete_features_name)


coef_means.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_upsample_linear_coef_means_two_by_two.csv")
coef_lower_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_upsample_linear_coef_lower_ci_two_by_two.csv")
coef_upper_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_upsample_linear_coef_upper_ci_two_by_two.csv")



