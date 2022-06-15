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

with open('D:/corona2020/Python Scripts_/DysTacMap/RF_results_features_importance_19112020.pkl', 'rb') as f:
    gm_feat_importance, wm_feat_importance, falff_feat_importance, \
                 structural_feat_importance, functional_feat_importance, \
                 complete_feat_importance = pkl.load(f)



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
#average the coefficient over the folder
#complete
average_2_by_2_coef = complete_feat_importance.mean(0)
#the output of the functions are assigned along the third dimension of the output array
#so the axis are (comparisons, features, output)
tt = np.apply_along_axis(mean_confidence_interval, 0, complete_feat_importance)


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
coef_means = pd.DataFrame([tt[0,:]], columns = complete_features_name)
coef_lower_ci = pd.DataFrame([tt[1,:]], columns = complete_features_name)
coef_upper_ci = pd.DataFrame([tt[2,:]], columns = complete_features_name)

coef_means.to_csv("D:/corona2020/Python Scripts_/DysTacMap/RF_linear_coef_means.csv")
coef_lower_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/RF_linear_coef_lower_ci.csv")
coef_upper_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/RF_linear_coef_upper_ci.csv")



with open('D:/corona2020/Python Scripts_/DysTacMap/RF_upsampling_results_09022021.pkl', 'rb') as f:
    gm_groupwise_accuracy_bal, gm_feat_importance, \
                 wm_groupwise_accuracy_bal, wm_feat_importance, \
                 falff_groupwise_accuracy_bal, falff_feat_importance, \
                 localcorr_groupwise_accuracy_bal, localcorr_feat_importance, \
                 globalcorr_groupwise_accuracy_bal, globalcorr_feat_importance, \
                 structural_groupwise_accuracy_bal, structural_feat_importance, \
                 functional_groupwise_accuracy_bal, functional_feat_importance, \
                 complete_groupwise_accuracy_bal, complete_feat_importance = pkl.load(f)



#average the coefficient over the folder
#complete
average_2_by_2_coef = complete_feat_importance.mean(0)
#the output of the functions are assigned along the third dimension of the output array
#so the axis are (comparisons, features, output)
tt = np.apply_along_axis(mean_confidence_interval, 0, complete_feat_importance)


coef_means = pd.DataFrame([tt[0,:]], columns = complete_features_name)
coef_lower_ci = pd.DataFrame([tt[1,:]], columns = complete_features_name)
coef_upper_ci = pd.DataFrame([tt[2,:]], columns = complete_features_name)

coef_means.to_csv("D:/corona2020/Python Scripts_/DysTacMap/RF_upsampling_linear_coef_means.csv")
coef_lower_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/RF_upsampling_linear_coef_lower_ci.csv")
coef_upper_ci.to_csv("D:/corona2020/Python Scripts_/DysTacMap/RF_upsampling_linear_coef_upper_ci.csv")



