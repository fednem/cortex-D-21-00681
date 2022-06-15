# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:22:42 2020

@author: federico.nemmi
"""
from neurocombat_sklearn import CombatModel
from helper_functions import get_residuals, SVC_CV, SVC_CV_shuff, calculate_groupwise_significance, SVC_rbf_CV, SVC_rbf_CV_shuff
import pandas as pd
import pickle as pkl

localcorr_naming_data_included = pd.read_csv("D:/corona2020/DysTacMap_Data/localCorr_naming_data.csv")
globalcorr_naming_data_included = pd.read_csv("D:/corona2020/DysTacMap_Data/globalCorr_naming_data.csv")
falff_naming_data_included = pd.read_csv("D:/corona2020/DysTacMap_Data/falff_naming_data.csv")
gm_naming_data_included = pd.read_csv("D:/corona2020/DysTacMap_Data/gm_naming_data.csv")
wm_naming_data_included = pd.read_csv("D:/corona2020/DysTacMap_Data/wm_naming_data.csv")
functional_included = pd.read_csv("D:/corona2020/DysTacMap_Data/functional_naming_data.csv")
structural_included = pd.read_csv("D:/corona2020/DysTacMap_Data/structural_naming_data.csv")
complete_included = pd.read_csv("D:/corona2020/DysTacMap_Data/complete_naming_data.csv")


#####REMOVE BATCH EFFECT WITH COMBAT#####
###divide variables among batch, continuous and categorical
##fmri
falff_naming_data_included = falff_naming_data_included[falff_naming_data_included.group != 1]
rs_batch = falff_naming_data_included.centre.values
rs_continuous = falff_naming_data_included.loc[:, ['demographic_age']].values
rs_categorical = falff_naming_data_included.loc[:, ['demographic_sex', "group"]].values
##t1
gm_naming_data_included = gm_naming_data_included[gm_naming_data_included.group != 1]
struct_batch = gm_naming_data_included.centre.values
struct_continuous = gm_naming_data_included.loc[:, ['demographic_age']].values
struct_categorical = gm_naming_data_included.loc[:, ['demographic_sex', "group"]].values
##falff
falff_values = falff_naming_data_included.iloc[:,[0,1]].values
cmbt = CombatModel()
falff_cmbt = cmbt.fit_transform(falff_values, 
                                rs_batch.reshape(-1,1), 
                                rs_categorical,
                                rs_continuous)
falff_final = falff_naming_data_included.copy()
falff_final.iloc[:,[0,1]] = falff_cmbt
##localCorr
localcorr_naming_data_included = localcorr_naming_data_included[localcorr_naming_data_included.group != 1]
localcorr_values = localcorr_naming_data_included.iloc[:,[0]].values
#cmbt = CombatModel()
#localcorr_cmbt = cmbt.fit_transform(localcorr_values, 
                                # rs_batch.reshape(-1,1), 
                                # rs_categorical,
                                # rs_continuous)
localcorr_final = localcorr_naming_data_included.copy()
#localcorr_final.iloc[:,[0]] = localcorr_cmbt
##globalCorr
globalcorr_naming_data_included = globalcorr_naming_data_included[globalcorr_naming_data_included.group != 1]
globalcorr_values = globalcorr_naming_data_included.iloc[:,[0, 1]].values
cmbt = CombatModel()
globalcorr_cmbt = cmbt.fit_transform(globalcorr_values, 
                                rs_batch.reshape(-1,1), 
                                rs_categorical,
                                rs_continuous)
globalcorr_final = globalcorr_naming_data_included.copy()
globalcorr_final.iloc[:,[0, 1]] = globalcorr_cmbt
##gm
gm_values = gm_naming_data_included.iloc[:,[0,1,2,3,4]]
cmbt = CombatModel()
gm_cmbt = cmbt.fit_transform(gm_values, 
                                struct_batch.reshape(-1,1), 
                                struct_categorical,
                                struct_continuous)
gm_final = gm_naming_data_included.copy()
gm_final.iloc[:,[0,1,2,3,4]] = gm_cmbt
##wm
wm_naming_data_included = wm_naming_data_included[wm_naming_data_included.group != 1]
wm_values = wm_naming_data_included.iloc[:,[0,1]]
cmbt = CombatModel()
wm_cmbt = cmbt.fit_transform(wm_values, 
                                struct_batch.reshape(-1,1), 
                                struct_categorical,
                                struct_continuous)
wm_final = wm_naming_data_included.copy()
wm_final.iloc[:,[0,1]] = wm_cmbt
##structural
structural_included = structural_included[structural_included.group != 1]
structural_values = structural_included.iloc[:,[0,1,2,3,4,6,7]]
cmbt = CombatModel()
structural_cmbt = cmbt.fit_transform(structural_values, 
                                struct_batch.reshape(-1,1), 
                                struct_categorical,
                                struct_continuous)
structural_final = structural_included.copy()
structural_final.iloc[:,[0,1,2,3,4,6,7]] = structural_cmbt
##functional
functional_included = functional_included[functional_included.group != 1]
functional_values = functional_included.iloc[:,[0,1,3,4,5]]
cmbt = CombatModel()
functional_cmbt = cmbt.fit_transform(functional_values, 
                                rs_batch.reshape(-1,1), 
                                rs_categorical,
                                rs_continuous)
functional_final = functional_included.copy()
functional_final.iloc[:,[0,1,3,4,5]] = functional_cmbt
##complete
complete_included = complete_included[complete_included.group != 1]
complete_values = complete_included.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]]
complete_cmbt = cmbt.fit_transform(complete_values,
                                   complete_included["centre"].values.reshape(-1,1),
                                   complete_included.loc[:,["group", "demographic_sex"]].values,
                                   complete_included.loc[:,["demographic_age"]])
complete_final = complete_included.copy()
complete_final.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]] = complete_cmbt
#####FIT MODELS#####
##gm fitting
gm_labels = gm_final["group"]

gm_otp_mean_bal, gm_otp_real_bal, gm_groupwise_accuracy_bal, gm_balanced_accuracy_bal, gm_feat_coef  = SVC_CV(gm_final[['15_24_57', '-27_15_6', '64_-15_2', '36_10_0', '32_-18_3']].values,
                                                                                                gm_labels, r_seed = 100, balancing = "balanced")

gm_suff_mean_acc, gm_shuff_bal_mean_acc, gm_shuff_groupwise_acc = SVC_CV_shuff(gm_final[['15_24_57', '-27_15_6', '64_-15_2', '36_10_0', '32_-18_3']].values,
                                                                                                gm_labels, r_seed = 100, balancing = "balanced")
gm_rbf_otp_mean_bal, gm_rbf_otp_real_bal, gm_rbf_groupwise_accuracy_bal, gm_rbf_balanced_accuracy_bal  = SVC_rbf_CV(gm_final[['15_24_57', '-27_15_6', '64_-15_2', '36_10_0', '32_-18_3']].values,
                                                                                                gm_labels, r_seed = 100, balancing = "balanced")

gm_rbf_suff_mean_acc, gm_rbf_shuff_bal_mean_acc, gm_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(gm_final[['15_24_57', '-27_15_6', '64_-15_2', '36_10_0', '32_-18_3']].values,
                                                                                                gm_labels, r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(gm_groupwise_accuracy_bal.mean(0), gm_shuff_groupwise_acc)
calculate_groupwise_significance(gm_rbf_groupwise_accuracy_bal.mean(0), gm_rbf_shuff_groupwise_acc)
##wm fitting
wm_labels = wm_final["group"]

wm_otp_mean_bal, wm_otp_real_bal, wm_groupwise_accuracy_bal, wm_balanced_accuracy_bal, wm_feat_coef  = SVC_CV(wm_final.iloc[:,[0,1]].values,
                                   wm_labels, r_seed = 100, balancing = "balanced")

wm_suff_mean_acc, wm_shuff_bal_mean_acc, wm_shuff_groupwise_acc = SVC_CV_shuff(wm_final.iloc[:,[0,1]].values,
                                                                                                wm_labels, r_seed = 100, balancing = "balanced")
wm_rbf_otp_mean_bal, wm_rbf_otp_real_bal, wm_rbf_groupwise_accuracy_bal, wm_rbf_balanced_accuracy_bal  = SVC_rbf_CV(wm_final.iloc[:,[0,1]].values,
                                   wm_labels, r_seed = 100, balancing = "balanced")

wm_rbf_suff_mean_acc, wm_rbf_shuff_bal_mean_acc, wm_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(wm_final.iloc[:,[0,1]].values,
                                                                                                wm_labels, r_seed = 100, balancing = "balanced")


calculate_groupwise_significance(wm_groupwise_accuracy_bal.mean(0), wm_shuff_groupwise_acc)
calculate_groupwise_significance(wm_rbf_groupwise_accuracy_bal.mean(0), wm_rbf_shuff_groupwise_acc)
##falff fitting
falff_labels = falff_final["group"]

falff_otp_mean_bal, falff_otp_real_bal, falff_groupwise_accuracy_bal, falff_balanced_accuracy_bal, falff_feat_coef  = SVC_CV(falff_final.iloc[:,[0,1]].values,
                                   falff_labels, r_seed = 100, balancing = "balanced")

falff_suff_mean_acc, falff_shuff_bal_mean_acc, falff_shuff_groupwise_acc = SVC_CV_shuff(falff_final.iloc[:,[0,1]].values,
                                                                                                falff_labels, r_seed = 100, balancing = "balanced")
falff_rbf_otp_mean_bal, falff_rbf_otp_real_bal, falff_rbf_groupwise_accuracy_bal, falff_rbf_balanced_accuracy_bal  = SVC_rbf_CV(falff_final.iloc[:,[0,1]].values,
                                   falff_labels, r_seed = 100, balancing = "balanced")

falff_rbf_suff_mean_acc, falff_rbf_shuff_bal_mean_acc, falff_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(falff_final.iloc[:,[0,1]].values,
                                                                                                falff_labels, r_seed = 100, balancing = "balanced")


calculate_groupwise_significance(falff_groupwise_accuracy_bal.mean(0), falff_shuff_groupwise_acc)
calculate_groupwise_significance(falff_rbf_groupwise_accuracy_bal.mean(0), falff_rbf_shuff_groupwise_acc)

##globalcorr fitting
globalcorr_labels = globalcorr_final["group"]

globalcorr_otp_mean_bal, globalcorr_otp_real_bal, globalcorr_groupwise_accuracy_bal, globalcorr_balanced_accuracy_bal, globalcorr_feat_coef  = SVC_CV(globalcorr_final.iloc[:,[0,1]].values,
                                   globalcorr_labels, r_seed = 100, balancing = "balanced")

globalcorr_suff_mean_acc, globalcorr_shuff_bal_mean_acc, globalcorr_shuff_groupwise_acc = SVC_CV_shuff(globalcorr_final.iloc[:,[0,1]].values,
                                                                                                globalcorr_labels, r_seed = 100, balancing = "balanced")

globalcorr_rbf_otp_mean_bal, globalcorr_rbf_otp_real_bal, globalcorr_rbf_groupwise_accuracy_bal, globalcorr_rbf_balanced_accuracy_bal  = SVC_rbf_CV(globalcorr_final.iloc[:,[0,1]].values,
                                   globalcorr_labels, r_seed = 100, balancing = "balanced")

globalcorr_rbf_suff_mean_acc, globalcorr_rbf_shuff_bal_mean_acc, globalcorr_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(globalcorr_final.iloc[:,[0,1]].values,
                                                                                                globalcorr_labels, r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(globalcorr_groupwise_accuracy_bal.mean(0), globalcorr_shuff_groupwise_acc)
calculate_groupwise_significance(globalcorr_rbf_groupwise_accuracy_bal.mean(0), globalcorr_rbf_shuff_groupwise_acc)

##localcorr fitting
localcorr_labels = localcorr_final["group"]

localcorr_otp_mean_bal, localcorr_otp_real_bal, localcorr_groupwise_accuracy_bal, localcorr_balanced_accuracy_bal, localcorr_feat_coef  = SVC_CV(localcorr_final.iloc[:,[0]].values,
                                   localcorr_labels, r_seed = 100, balancing = "balanced")

localcorr_suff_mean_acc, localcorr_shuff_bal_mean_acc, localcorr_shuff_groupwise_acc = SVC_CV_shuff(localcorr_final.iloc[:,[0]].values,
                                                                                                localcorr_labels, r_seed = 100, balancing = "balanced")

localcorr_rbf_otp_mean_bal, localcorr_rbf_otp_real_bal, localcorr_rbf_groupwise_accuracy_bal, localcorr_rbf_balanced_accuracy_bal  = SVC_rbf_CV(localcorr_final.iloc[:,[0]].values,
                                   localcorr_labels, r_seed = 100, balancing = "balanced")

localcorr_rbf_suff_mean_acc, localcorr_rbf_shuff_bal_mean_acc, localcorr_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(localcorr_final.iloc[:,[0]].values,
                                                                                                localcorr_labels, r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(localcorr_groupwise_accuracy_bal.mean(0), localcorr_shuff_groupwise_acc)
calculate_groupwise_significance(localcorr_rbf_groupwise_accuracy_bal.mean(0), localcorr_rbf_shuff_groupwise_acc)


##structural fitting
structural_labels = structural_final["group"]

structural_otp_mean_bal, structural_otp_real_bal, structural_groupwise_accuracy_bal, structural_balanced_accuracy_bal, structural_feat_coef  = SVC_CV(structural_final.iloc[:,[0,1,2,3,4,6,7]].values,
                                   structural_labels, r_seed = 100, balancing = "balanced")

structural_suff_mean_acc, structural_shuff_bal_mean_acc, structural_shuff_groupwise_acc = SVC_CV_shuff(structural_final.iloc[:,[0,1,2,3,4,6,7]].values,
                                                                                                structural_labels, r_seed = 100, balancing = "balanced")
structural_rbf_otp_mean_bal, structural_rbf_otp_real_bal, structural_rbf_groupwise_accuracy_bal, structural_rbf_balanced_accuracy_bal  = SVC_rbf_CV(structural_final.iloc[:,[0,1,2,3,4,6,7]].values,
                                   structural_labels, r_seed = 100, balancing = "balanced")

structural_rbf_suff_mean_acc, structural_rbf_shuff_bal_mean_acc, structural_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(structural_final.iloc[:,[0,1,2,3,4,6,7]].values,
                                                                                                structural_labels, r_seed = 100, balancing = "balanced")


calculate_groupwise_significance(structural_groupwise_accuracy_bal.mean(0), structural_shuff_groupwise_acc)
calculate_groupwise_significance(structural_rbf_groupwise_accuracy_bal.mean(0), structural_rbf_shuff_groupwise_acc)
##functional fitting
functional_labels = functional_final["group"]

functional_otp_mean_bal, functional_otp_real_bal, functional_groupwise_accuracy_bal, functional_balanced_accuracy_bal, functional_feat_coef  = SVC_CV(functional_final.iloc[:,[0,1,3,4,5]].values,
                                   functional_labels, r_seed = 100, balancing = "balanced")

functional_suff_mean_acc, functional_shuff_bal_mean_acc, functional_shuff_groupwise_acc = SVC_CV_shuff(functional_final.iloc[:,[0,1,3,4,5]].values,
                                                                                                functional_labels, r_seed = 100, balancing = "balanced")

functional_rbf_otp_mean_bal, functional_rbf_otp_real_bal, functional_rbf_groupwise_accuracy_bal, functional_rbf_balanced_accuracy_bal  = SVC_rbf_CV(functional_final.iloc[:,[0,1,3,4,5]].values,
                                   functional_labels, r_seed = 100, balancing = "balanced")

functional_rbf_suff_mean_acc, functional_rbf_shuff_bal_mean_acc, functional_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(functional_final.iloc[:,[0,1,3,4,5]].values,
                                                                                                functional_labels, r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(functional_groupwise_accuracy_bal.mean(0), functional_shuff_groupwise_acc)
calculate_groupwise_significance(functional_rbf_groupwise_accuracy_bal.mean(0), functional_rbf_shuff_groupwise_acc)

##complete fitting
complete_labels = complete_final["group"]

complete_otp_mean_bal, complete_otp_real_bal, complete_groupwise_accuracy_bal, complete_balanced_accuracy_bal, complete_feat_coef  = SVC_CV(complete_final.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]].values,
                                   complete_labels, r_seed = 100, balancing = "balanced")

complete_rbf_otp_mean_bal, complete_rbf_otp_real_bal, complete_rbf_groupwise_accuracy_bal, complete_rbf_balanced_accuracy_bal  = SVC_rbf_CV(complete_final.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]].values,
                                   complete_labels, r_seed = 100, balancing = "balanced")


complete_suff_mean_acc, complete_shuff_bal_mean_acc, complete_shuff_groupwise_acc = SVC_CV_shuff(complete_final.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]].values,
                                                                                                complete_labels, r_seed = 100, balancing = "balanced")
complete_rbf_suff_mean_acc, complete_rbf_shuff_bal_mean_acc, complete_rbf_shuff_groupwise_acc = SVC_rbf_CV_shuff(complete_final.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13]].values,
                                                                                                complete_labels, r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(complete_groupwise_accuracy_bal.mean(0), complete_shuff_groupwise_acc)
calculate_groupwise_significance(complete_rbf_groupwise_accuracy_bal.mean(0), complete_rbf_shuff_groupwise_acc)

with open('SVC_results_with_coef_02122020.pkl', 'wb') as f:
    pkl.dump([gm_groupwise_accuracy_bal, gm_rbf_groupwise_accuracy_bal, gm_shuff_groupwise_acc, gm_rbf_shuff_groupwise_acc, gm_feat_coef,
                 wm_groupwise_accuracy_bal, wm_rbf_groupwise_accuracy_bal, wm_shuff_groupwise_acc, wm_rbf_shuff_groupwise_acc, wm_feat_coef,
                 falff_groupwise_accuracy_bal, falff_rbf_groupwise_accuracy_bal, falff_shuff_groupwise_acc, falff_rbf_shuff_groupwise_acc, falff_feat_coef,
                 localcorr_groupwise_accuracy_bal, localcorr_rbf_groupwise_accuracy_bal, localcorr_shuff_groupwise_acc, localcorr_rbf_shuff_groupwise_acc, localcorr_feat_coef,
                 globalcorr_groupwise_accuracy_bal, globalcorr_rbf_groupwise_accuracy_bal, globalcorr_shuff_groupwise_acc, globalcorr_rbf_shuff_groupwise_acc, globalcorr_feat_coef,
                 structural_groupwise_accuracy_bal, structural_rbf_groupwise_accuracy_bal, structural_shuff_groupwise_acc, structural_rbf_shuff_groupwise_acc, structural_feat_coef,
                 functional_groupwise_accuracy_bal, functional_rbf_groupwise_accuracy_bal, functional_shuff_groupwise_acc, functional_rbf_shuff_groupwise_acc, functional_feat_coef,
                 complete_groupwise_accuracy_bal, complete_rbf_groupwise_accuracy_bal, complete_shuff_groupwise_acc, complete_rbf_shuff_groupwise_acc, complete_feat_coef], f)