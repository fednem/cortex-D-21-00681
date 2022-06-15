# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:36:10 2021

@author: federico.nemmi
"""
import pickle as pkl
import numpy as np
import scipy.stats
from neurocombat_sklearn import CombatModel
import pandas as pd

with open('D:/corona2020/Python Scripts_/DysTacMap/SVC_results_with_coef_02122020.pkl', 'rb') as f:
    gm_groupwise_accuracy_bal, gm_rbf_groupwise_accuracy_bal, gm_shuff_groupwise_acc, gm_rbf_shuff_groupwise_acc, gm_feat_coef, \
    wm_groupwise_accuracy_bal, wm_rbf_groupwise_accuracy_bal, wm_shuff_groupwise_acc, wm_rbf_shuff_groupwise_acc, wm_feat_coef, \
    falff_groupwise_accuracy_bal, falff_rbf_groupwise_accuracy_bal, falff_shuff_groupwise_acc, falff_rbf_shuff_groupwise_acc, falff_feat_coef, \
    localcorr_groupwise_accuracy_bal, localcorr_rbf_groupwise_accuracy_bal, localcorr_shuff_groupwise_acc, localcorr_rbf_shuff_groupwise_acc, localcorr_feat_coef, \
    globalcorr_groupwise_accuracy_bal, globalcorr_rbf_groupwise_accuracy_bal, globalcorr_shuff_groupwise_acc, globalcorr_rbf_shuff_groupwise_acc, globalcorr_feat_coef, \
    structural_groupwise_accuracy_bal, structural_rbf_groupwise_accuracy_bal, structural_shuff_groupwise_acc, structural_rbf_shuff_groupwise_acc, structural_feat_coef, \
    functional_groupwise_accuracy_bal, functional_rbf_groupwise_accuracy_bal, functional_shuff_groupwise_acc, functional_rbf_shuff_groupwise_acc, functional_feat_coef, \
    complete_groupwise_accuracy_bal, complete_rbf_groupwise_accuracy_bal, complete_shuff_groupwise_acc, complete_rbf_shuff_groupwise_acc, complete_feat_coef = pkl.load(f)



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
tt = np.apply_along_axis(mean_confidence_interval, 2, complete_feat_coef)


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
coef_means = pd.DataFrame(tt[:,:,0], columns = complete_features_name)
coef_lower_ci = pd.DataFrame(tt[:,:,1], columns = complete_features_name)
coef_upper_ci = pd.DataFrame(tt[:,:,2], columns = complete_features_name)

coef_means.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_linear_coef_means.csv")
coef_lower_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_linear_coef_lower_ci.csv")
coef_upper_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_linear_coef_upper_ci.csv")



with open('D:/corona2020/Python Scripts_/DysTacMap/SVC_linear_upsampling_results_with_coef_02112020.pkl', 'rb') as f:
    gm_upsampling_groupwise_accuracy_bal, gm_upsampling_shuff_groupwise_acc, gm_upsampling_feat_coef, \
                 wm_upsampling_groupwise_accuracy_bal, wm_upsampling_shuff_groupwise_acc, wm_upsampling_feat_coef, \
                 falff_upsampling_groupwise_accuracy_bal, falff_upsampling_shuff_groupwise_acc, falff_upsampling_feat_coef, \
                 localcorr_upsampling_groupwise_accuracy_bal, localcorr_upsampling_shuff_groupwise_acc, localcorr_upsampling_feat_coef, \
                 globalcorr_upsampling_groupwise_accuracy_bal, globalcorr_upsampling_shuff_groupwise_acc, globalcorr_upsampling_feat_coef, \
                 structural_upsampling_groupwise_accuracy_bal, structural_upsampling_shuff_groupwise_acc, structural_upsampling_feat_coef, \
                 functional_upsampling_groupwise_accuracy_bal, functional_upsampling_shuff_groupwise_acc, functional_upsampling_feat_coef, \
                 complete_upsampling_groupwise_accuracy_bal, complete_upsampling_shuff_groupwise_acc, complete_upsampling_feat_coef = pkl.load(f)


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
tt = np.apply_along_axis(mean_confidence_interval, 2, complete_upsampling_feat_coef)

coef_means = pd.DataFrame(tt[:,:,0], columns = complete_features_name)
coef_lower_ci = pd.DataFrame(tt[:,:,1], columns = complete_features_name)
coef_upper_ci = pd.DataFrame(tt[:,:,2], columns = complete_features_name)

coef_means.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_upsampling_linear_coef_means.csv")
coef_lower_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_upsampling_linear_coef_lower_ci.csv")
coef_upper_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/SVC_upsampling_linear_coef_upper_ci.csv")
