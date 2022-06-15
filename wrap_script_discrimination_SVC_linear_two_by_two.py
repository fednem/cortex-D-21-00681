# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:22:42 2020

@author: federico.nemmi
"""
from neurocombat_sklearn import CombatModel
from helper_functions import calculate_groupwise_significance, SVC_CV, SVC_CV_shuff
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
#DCD vs DYS 
gm_labels = gm_final["group"]
gm_labels = gm_labels[gm_labels <4]
gm_in_matrix = gm_final.loc[gm_final["group"] < 4,['15_24_57', '-27_15_6', '64_-15_2', '36_10_0', '32_-18_3']].values
gm_otp_mean_bal_dcd_vs_dys,\
    gm_otp_real_bal_dcd_vs_dys,\
        gm_groupwise_accuracy_bal_dcd_vs_dys,\
            gm_balanced_accuracy_bal_dcd_vs_dys,\
                gm_feat_coef_dcd_vs_dys  = SVC_CV(gm_in_matrix,
                                                             gm_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

gm_suff_mean_acc_dcd_vs_dys,\
    gm_shuff_bal_mean_acc_dcd_vs_dys,\
        gm_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(gm_in_matrix,
                                                                    gm_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(gm_groupwise_accuracy_bal_dcd_vs_dys, gm_shuff_groupwise_acc_dcd_vs_dys)

#DYS vs COM 
gm_labels = gm_final["group"]
gm_labels = gm_labels[gm_labels >2]
gm_in_matrix = gm_final.loc[gm_final["group"] > 2,['15_24_57', '-27_15_6', '64_-15_2', '36_10_0', '32_-18_3']].values
gm_otp_mean_bal_dys_vs_com,\
    gm_otp_real_bal_dys_vs_com,\
        gm_groupwise_accuracy_bal_dys_vs_com,\
            gm_balanced_accuracy_bal_dys_vs_com,\
                gm_feat_coef_dys_vs_com  = SVC_CV(gm_in_matrix,
                                                             gm_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

gm_suff_mean_acc_dys_vs_com,\
    gm_shuff_bal_mean_acc_dys_vs_com,\
        gm_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(gm_in_matrix,
                                                                    gm_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(gm_groupwise_accuracy_bal_dys_vs_com, gm_shuff_groupwise_acc_dys_vs_com)


#DCD vs COM 
gm_labels = gm_final["group"]
gm_labels = gm_labels[gm_labels != 3]
gm_in_matrix = gm_final.loc[gm_final["group"] != 3, ['15_24_57', '-27_15_6', '64_-15_2', '36_10_0', '32_-18_3']].values
gm_otp_mean_bal_dcd_vs_com,\
    gm_otp_real_bal_dcd_vs_com,\
        gm_groupwise_accuracy_bal_dcd_vs_com,\
            gm_balanced_accuracy_bal_dcd_vs_com,\
                gm_feat_coef_dcd_vs_com  = SVC_CV(gm_in_matrix,
                                                             gm_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

gm_suff_mean_acc_dcd_vs_com,\
    gm_shuff_bal_mean_acc_dcd_vs_com,\
        gm_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(gm_in_matrix,
                                                                    gm_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(gm_groupwise_accuracy_bal_dcd_vs_com, gm_shuff_groupwise_acc_dcd_vs_com)

##wm fitting
wm_labels = wm_final["group"]
#DCD vs DYS 
wm_labels = wm_final["group"]
wm_labels = wm_labels[wm_labels <4]
wm_in_matrix = wm_final.loc[wm_final["group"] < 4,["15_21_52", "-42_-56_-38"]].values
wm_otp_mean_bal_dcd_vs_dys,\
    wm_otp_real_bal_dcd_vs_dys,\
        wm_groupwise_accuracy_bal_dcd_vs_dys,\
            wm_balanced_accuracy_bal_dcd_vs_dys,\
                wm_feat_coef_dcd_vs_dys  = SVC_CV(wm_in_matrix,
                                                             wm_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

wm_suff_mean_acc_dcd_vs_dys,\
    wm_shuff_bal_mean_acc_dcd_vs_dys,\
        wm_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(wm_in_matrix,
                                                                    wm_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(wm_groupwise_accuracy_bal_dcd_vs_dys, wm_shuff_groupwise_acc_dcd_vs_dys)


#DYS vs COM 
wm_labels = wm_final["group"]
wm_labels = wm_labels[wm_labels >2]
wm_in_matrix = wm_final.loc[wm_final["group"] > 2,["15_21_52", "-42_-56_-38"]].values
wm_otp_mean_bal_dys_vs_com,\
    wm_otp_real_bal_dys_vs_com,\
        wm_groupwise_accuracy_bal_dys_vs_com,\
            wm_balanced_accuracy_bal_dys_vs_com,\
                wm_feat_coef_dys_vs_com  = SVC_CV(wm_in_matrix,
                                                             wm_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

wm_suff_mean_acc_dys_vs_com,\
    wm_shuff_bal_mean_acc_dys_vs_com,\
        wm_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(wm_in_matrix,
                                                                    wm_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(wm_groupwise_accuracy_bal_dys_vs_com, wm_shuff_groupwise_acc_dys_vs_com)



#DCD vs COM 
wm_labels = wm_final["group"]
wm_labels = wm_labels[wm_labels != 3]
wm_in_matrix = wm_final.loc[wm_final["group"] != 3,["15_21_52", "-42_-56_-38"]].values
wm_otp_mean_bal_dcd_vs_com,\
    wm_otp_real_bal_dcd_vs_com,\
        wm_groupwise_accuracy_bal_dcd_vs_com,\
            wm_balanced_accuracy_bal_dcd_vs_com,\
                wm_feat_coef_dcd_vs_com  = SVC_CV(wm_in_matrix,
                                                             wm_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

wm_suff_mean_acc_dcd_vs_com,\
    wm_shuff_bal_mean_acc_dcd_vs_com,\
        wm_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(wm_in_matrix,
                                                                    wm_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(wm_groupwise_accuracy_bal_dcd_vs_com, wm_shuff_groupwise_acc_dcd_vs_com)


##falff fitting
falff_labels = falff_final["group"]
#DCD vs DYS 
falff_labels = falff_final["group"]
falff_labels = falff_labels[falff_labels <4]
falff_in_matrix = falff_final.loc[falff_final["group"] < 4,["-9_-81_-39", "40_15_32"]].values
falff_otp_mean_bal_dcd_vs_dys,\
    falff_otp_real_bal_dcd_vs_dys,\
        falff_groupwise_accuracy_bal_dcd_vs_dys,\
            falff_balanced_accuracy_bal_dcd_vs_dys,\
                falff_feat_coef_dcd_vs_dys  = SVC_CV(falff_in_matrix,
                                                             falff_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

falff_suff_mean_acc_dcd_vs_dys,\
    falff_shuff_bal_mean_acc_dcd_vs_dys,\
        falff_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(falff_in_matrix,
                                                                    falff_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(falff_groupwise_accuracy_bal_dcd_vs_dys, falff_shuff_groupwise_acc_dcd_vs_dys)


#DYS vs COM 
falff_labels = falff_final["group"]
falff_labels = falff_labels[falff_labels >2]
falff_in_matrix = falff_final.loc[falff_final["group"] > 2,["-9_-81_-39", "40_15_32"]].values
falff_otp_mean_bal_dys_vs_com,\
    falff_otp_real_bal_dys_vs_com,\
        falff_groupwise_accuracy_bal_dys_vs_com,\
            falff_balanced_accuracy_bal_dys_vs_com,\
                falff_feat_coef_dys_vs_com  = SVC_CV(falff_in_matrix,
                                                             falff_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

falff_suff_mean_acc_dys_vs_com,\
    falff_shuff_bal_mean_acc_dys_vs_com,\
        falff_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(falff_in_matrix,
                                                                    falff_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(falff_groupwise_accuracy_bal_dys_vs_com, falff_shuff_groupwise_acc_dys_vs_com)



#DCD vs COM 
falff_labels = falff_final["group"]
falff_labels = falff_labels[falff_labels != 3]
falff_in_matrix = falff_final.loc[falff_final["group"] != 3,["-9_-81_-39", "40_15_32"]].values
falff_otp_mean_bal_dcd_vs_com,\
    falff_otp_real_bal_dcd_vs_com,\
        falff_groupwise_accuracy_bal_dcd_vs_com,\
            falff_balanced_accuracy_bal_dcd_vs_com,\
                falff_feat_coef_dcd_vs_com  = SVC_CV(falff_in_matrix,
                                                             falff_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

falff_suff_mean_acc_dcd_vs_com,\
    falff_shuff_bal_mean_acc_dcd_vs_com,\
        falff_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(falff_in_matrix,
                                                                    falff_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(falff_groupwise_accuracy_bal_dcd_vs_com, falff_shuff_groupwise_acc_dcd_vs_com)



##globalcorr fitting
globalcorr_labels = globalcorr_final["group"]
#DCD vs DYS 
globalcorr_labels = globalcorr_final["group"]
globalcorr_labels = globalcorr_labels[globalcorr_labels <4]
globalcorr_in_matrix = globalcorr_final.loc[globalcorr_final["group"] < 4,["39_-61_50", "25_29_48"]].values
globalcorr_otp_mean_bal_dcd_vs_dys,\
    globalcorr_otp_real_bal_dcd_vs_dys,\
        globalcorr_groupwise_accuracy_bal_dcd_vs_dys,\
            globalcorr_balanced_accuracy_bal_dcd_vs_dys,\
                globalcorr_feat_coef_dcd_vs_dys  = SVC_CV(globalcorr_in_matrix,
                                                             globalcorr_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

globalcorr_suff_mean_acc_dcd_vs_dys,\
    globalcorr_shuff_bal_mean_acc_dcd_vs_dys,\
        globalcorr_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(globalcorr_in_matrix,
                                                                    globalcorr_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(globalcorr_groupwise_accuracy_bal_dcd_vs_dys, globalcorr_shuff_groupwise_acc_dcd_vs_dys)


#DYS vs COM 
globalcorr_labels = globalcorr_final["group"]
globalcorr_labels = globalcorr_labels[globalcorr_labels >2]
globalcorr_in_matrix = globalcorr_final.loc[globalcorr_final["group"] > 2,["39_-61_50", "25_29_48"]].values
globalcorr_otp_mean_bal_dys_vs_com,\
    globalcorr_otp_real_bal_dys_vs_com,\
        globalcorr_groupwise_accuracy_bal_dys_vs_com,\
            globalcorr_balanced_accuracy_bal_dys_vs_com,\
                globalcorr_feat_coef_dys_vs_com  = SVC_CV(globalcorr_in_matrix,
                                                             globalcorr_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

globalcorr_suff_mean_acc_dys_vs_com,\
    globalcorr_shuff_bal_mean_acc_dys_vs_com,\
        globalcorr_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(globalcorr_in_matrix,
                                                                    globalcorr_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(globalcorr_groupwise_accuracy_bal_dys_vs_com, globalcorr_shuff_groupwise_acc_dys_vs_com)



#DCD vs COM 
globalcorr_labels = globalcorr_final["group"]
globalcorr_labels = globalcorr_labels[globalcorr_labels != 3]
globalcorr_in_matrix = globalcorr_final.loc[globalcorr_final["group"] != 3,["39_-61_50", "25_29_48"]].values
globalcorr_otp_mean_bal_dcd_vs_com,\
    globalcorr_otp_real_bal_dcd_vs_com,\
        globalcorr_groupwise_accuracy_bal_dcd_vs_com,\
            globalcorr_balanced_accuracy_bal_dcd_vs_com,\
                globalcorr_feat_coef_dcd_vs_com  = SVC_CV(globalcorr_in_matrix,
                                                             globalcorr_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

globalcorr_suff_mean_acc_dcd_vs_com,\
    globalcorr_shuff_bal_mean_acc_dcd_vs_com,\
        globalcorr_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(globalcorr_in_matrix,
                                                                    globalcorr_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(globalcorr_groupwise_accuracy_bal_dcd_vs_com, globalcorr_shuff_groupwise_acc_dcd_vs_com)


##localcorr fitting
localcorr_labels = localcorr_final["group"]
#DCD vs DYS 
localcorr_labels = localcorr_final["group"]
localcorr_labels = localcorr_labels[localcorr_labels <4]
localcorr_in_matrix = localcorr_final.loc[localcorr_final["group"] < 4,['-24_-49_-48']].values
localcorr_otp_mean_bal_dcd_vs_dys,\
    localcorr_otp_real_bal_dcd_vs_dys,\
        localcorr_groupwise_accuracy_bal_dcd_vs_dys,\
            localcorr_balanced_accuracy_bal_dcd_vs_dys,\
                localcorr_feat_coef_dcd_vs_dys  = SVC_CV(localcorr_in_matrix,
                                                             localcorr_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

localcorr_suff_mean_acc_dcd_vs_dys,\
    localcorr_shuff_bal_mean_acc_dcd_vs_dys,\
        localcorr_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(localcorr_in_matrix,
                                                                    localcorr_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(localcorr_groupwise_accuracy_bal_dcd_vs_dys, localcorr_shuff_groupwise_acc_dcd_vs_dys)


#DYS vs COM 
localcorr_labels = localcorr_final["group"]
localcorr_labels = localcorr_labels[localcorr_labels >2]
localcorr_in_matrix = localcorr_final.loc[localcorr_final["group"] > 2,['-24_-49_-48']].values
localcorr_otp_mean_bal_dys_vs_com,\
    localcorr_otp_real_bal_dys_vs_com,\
        localcorr_groupwise_accuracy_bal_dys_vs_com,\
            localcorr_balanced_accuracy_bal_dys_vs_com,\
                localcorr_feat_coef_dys_vs_com  = SVC_CV(localcorr_in_matrix,
                                                             localcorr_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

localcorr_suff_mean_acc_dys_vs_com,\
    localcorr_shuff_bal_mean_acc_dys_vs_com,\
        localcorr_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(localcorr_in_matrix,
                                                                    localcorr_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(localcorr_groupwise_accuracy_bal_dys_vs_com, localcorr_shuff_groupwise_acc_dys_vs_com)



#DCD vs COM 
localcorr_labels = localcorr_final["group"]
localcorr_labels = localcorr_labels[localcorr_labels != 3]
localcorr_in_matrix = localcorr_final.loc[localcorr_final["group"] != 3,['-24_-49_-48']].values
localcorr_otp_mean_bal_dcd_vs_com,\
    localcorr_otp_real_bal_dcd_vs_com,\
        localcorr_groupwise_accuracy_bal_dcd_vs_com,\
            localcorr_balanced_accuracy_bal_dcd_vs_com,\
                localcorr_feat_coef_dcd_vs_com  = SVC_CV(localcorr_in_matrix,
                                                             localcorr_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

localcorr_suff_mean_acc_dcd_vs_com,\
    localcorr_shuff_bal_mean_acc_dcd_vs_com,\
        localcorr_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(localcorr_in_matrix,
                                                                    localcorr_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(localcorr_groupwise_accuracy_bal_dcd_vs_com, localcorr_shuff_groupwise_acc_dcd_vs_com)


##structural fitting
structural_labels = structural_final["group"]
#DCD vs DYS 
structural_labels = structural_final["group"]
structural_labels = structural_labels[structural_labels <4]
structural_in_matrix = structural_final.loc[structural_final["group"] < 4,['15_24_57', 
                                                                           '-27_15_6', 
                                                                           '64_-15_2', 
                                                                           '36_10_0', 
                                                                           '32_-18_3',
                                                                           '15_21_52',
                                                                           '-42_-56_-38']].values
structural_otp_mean_bal_dcd_vs_dys,\
    structural_otp_real_bal_dcd_vs_dys,\
        structural_groupwise_accuracy_bal_dcd_vs_dys,\
            structural_balanced_accuracy_bal_dcd_vs_dys,\
                structural_feat_coef_dcd_vs_dys  = SVC_CV(structural_in_matrix,
                                                             structural_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

structural_suff_mean_acc_dcd_vs_dys,\
    structural_shuff_bal_mean_acc_dcd_vs_dys,\
        structural_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(structural_in_matrix,
                                                                    structural_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(structural_groupwise_accuracy_bal_dcd_vs_dys, structural_shuff_groupwise_acc_dcd_vs_dys)


#DYS vs COM 
structural_labels = structural_final["group"]
structural_labels = structural_labels[structural_labels >2]
structural_in_matrix = structural_final.loc[structural_final["group"] > 2,['15_24_57', 
                                                                           '-27_15_6', 
                                                                           '64_-15_2', 
                                                                           '36_10_0', 
                                                                           '32_-18_3',
                                                                           '15_21_52',
                                                                           '-42_-56_-38']].values
structural_otp_mean_bal_dys_vs_com,\
    structural_otp_real_bal_dys_vs_com,\
        structural_groupwise_accuracy_bal_dys_vs_com,\
            structural_balanced_accuracy_bal_dys_vs_com,\
                structural_feat_coef_dys_vs_com  = SVC_CV(structural_in_matrix,
                                                             structural_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

structural_suff_mean_acc_dys_vs_com,\
    structural_shuff_bal_mean_acc_dys_vs_com,\
        structural_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(structural_in_matrix,
                                                                    structural_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(structural_groupwise_accuracy_bal_dys_vs_com, structural_shuff_groupwise_acc_dys_vs_com)



#DCD vs COM 
structural_labels = structural_final["group"]
structural_labels = structural_labels[structural_labels != 3]
structural_in_matrix = structural_final.loc[structural_final["group"] != 3,['15_24_57', 
                                                                           '-27_15_6', 
                                                                           '64_-15_2', 
                                                                           '36_10_0', 
                                                                           '32_-18_3',
                                                                           '15_21_52',
                                                                           '-42_-56_-38']].values
structural_otp_mean_bal_dcd_vs_com,\
    structural_otp_real_bal_dcd_vs_com,\
        structural_groupwise_accuracy_bal_dcd_vs_com,\
            structural_balanced_accuracy_bal_dcd_vs_com,\
                structural_feat_coef_dcd_vs_com  = SVC_CV(structural_in_matrix,
                                                             structural_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

structural_suff_mean_acc_dcd_vs_com,\
    structural_shuff_bal_mean_acc_dcd_vs_com,\
        structural_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(structural_in_matrix,
                                                                    structural_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(structural_groupwise_accuracy_bal_dcd_vs_com, structural_shuff_groupwise_acc_dcd_vs_com)


##functional fitting
functional_labels = functional_final["group"]
#DCD vs DYS 
functional_labels = functional_final["group"]
functional_labels = functional_labels[functional_labels <4]
functional_in_matrix = functional_final.loc[functional_final["group"] < 4,['-9_-81_-39',
                                                                           '40_15_32',
                                                                           '-24_-49_-48',
                                                                           '39_-61_50',
                                                                           '25_29_48']].values
functional_otp_mean_bal_dcd_vs_dys,\
    functional_otp_real_bal_dcd_vs_dys,\
        functional_groupwise_accuracy_bal_dcd_vs_dys,\
            functional_balanced_accuracy_bal_dcd_vs_dys,\
                functional_feat_coef_dcd_vs_dys  = SVC_CV(functional_in_matrix,
                                                             functional_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

functional_suff_mean_acc_dcd_vs_dys,\
    functional_shuff_bal_mean_acc_dcd_vs_dys,\
        functional_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(functional_in_matrix,
                                                                    functional_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(functional_groupwise_accuracy_bal_dcd_vs_dys, functional_shuff_groupwise_acc_dcd_vs_dys)


#DYS vs COM 
functional_labels = functional_final["group"]
functional_labels = functional_labels[functional_labels >2]
functional_in_matrix = functional_final.loc[functional_final["group"] > 2,['-9_-81_-39',
                                                                           '40_15_32',
                                                                           '-24_-49_-48',
                                                                           '39_-61_50',
                                                                           '25_29_48']].values
functional_otp_mean_bal_dys_vs_com,\
    functional_otp_real_bal_dys_vs_com,\
        functional_groupwise_accuracy_bal_dys_vs_com,\
            functional_balanced_accuracy_bal_dys_vs_com,\
                functional_feat_coef_dys_vs_com  = SVC_CV(functional_in_matrix,
                                                             functional_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

functional_suff_mean_acc_dys_vs_com,\
    functional_shuff_bal_mean_acc_dys_vs_com,\
        functional_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(functional_in_matrix,
                                                                    functional_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(functional_groupwise_accuracy_bal_dys_vs_com, functional_shuff_groupwise_acc_dys_vs_com)



#DCD vs COM 
functional_labels = functional_final["group"]
functional_labels = functional_labels[functional_labels != 3]
functional_in_matrix = functional_final.loc[functional_final["group"] != 3,['-9_-81_-39',
                                                                           '40_15_32',
                                                                           '-24_-49_-48',
                                                                           '39_-61_50',
                                                                           '25_29_48']].values
functional_otp_mean_bal_dcd_vs_com,\
    functional_otp_real_bal_dcd_vs_com,\
        functional_groupwise_accuracy_bal_dcd_vs_com,\
            functional_balanced_accuracy_bal_dcd_vs_com,\
                functional_feat_coef_dcd_vs_com  = SVC_CV(functional_in_matrix,
                                                             functional_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

functional_suff_mean_acc_dcd_vs_com,\
    functional_shuff_bal_mean_acc_dcd_vs_com,\
        functional_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(functional_in_matrix,
                                                                    functional_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(functional_groupwise_accuracy_bal_dcd_vs_com, functional_shuff_groupwise_acc_dcd_vs_com)

##complete fitting
complete_labels = complete_final["group"]
#DCD vs DYS 
complete_labels = complete_final["group"]
complete_labels = complete_labels[complete_labels <4]
complete_in_matrix = complete_final.loc[complete_final["group"] < 4,['-9_-81_-39', 
                                                                     '40_15_32', 
                                                                     '-24_-49_-48', 
                                                                     '39_-61_50', 
                                                                     '25_29_48',
                                                                     '15_24_57',
                                                                     '-27_15_6',
                                                                     '64_-15_2',
                                                                     '36_10_0',
                                                                     '32_-18_3',
                                                                     '15_21_52',
                                                                     '-42_-56_-38']].values
complete_otp_mean_bal_dcd_vs_dys,\
    complete_otp_real_bal_dcd_vs_dys,\
        complete_groupwise_accuracy_bal_dcd_vs_dys,\
            complete_balanced_accuracy_bal_dcd_vs_dys,\
                complete_feat_coef_dcd_vs_dys  = SVC_CV(complete_in_matrix,
                                                             complete_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

complete_suff_mean_acc_dcd_vs_dys,\
    complete_shuff_bal_mean_acc_dcd_vs_dys,\
        complete_shuff_groupwise_acc_dcd_vs_dys = SVC_CV_shuff(complete_in_matrix,
                                                                    complete_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(complete_groupwise_accuracy_bal_dcd_vs_dys, complete_shuff_groupwise_acc_dcd_vs_dys)


#DYS vs COM 
complete_labels = complete_final["group"]
complete_labels = complete_labels[complete_labels >2]
complete_in_matrix = complete_final.loc[complete_final["group"] > 2,['-9_-81_-39', 
                                                                     '40_15_32', 
                                                                     '-24_-49_-48', 
                                                                     '39_-61_50', 
                                                                     '25_29_48',
                                                                     '15_24_57',
                                                                     '-27_15_6',
                                                                     '64_-15_2',
                                                                     '36_10_0',
                                                                     '32_-18_3',
                                                                     '15_21_52',
                                                                     '-42_-56_-38']].values
complete_otp_mean_bal_dys_vs_com,\
    complete_otp_real_bal_dys_vs_com,\
        complete_groupwise_accuracy_bal_dys_vs_com,\
            complete_balanced_accuracy_bal_dys_vs_com,\
                complete_feat_coef_dys_vs_com  = SVC_CV(complete_in_matrix,
                                                             complete_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

complete_suff_mean_acc_dys_vs_com,\
    complete_shuff_bal_mean_acc_dys_vs_com,\
        complete_shuff_groupwise_acc_dys_vs_com = SVC_CV_shuff(complete_in_matrix,
                                                                    complete_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(complete_groupwise_accuracy_bal_dys_vs_com, complete_shuff_groupwise_acc_dys_vs_com)



#DCD vs COM 
complete_labels = complete_final["group"]
complete_labels = complete_labels[complete_labels != 3]
complete_in_matrix = complete_final.loc[complete_final["group"] != 3,['-9_-81_-39', 
                                                                     '40_15_32', 
                                                                     '-24_-49_-48', 
                                                                     '39_-61_50', 
                                                                     '25_29_48',
                                                                     '15_24_57',
                                                                     '-27_15_6',
                                                                     '64_-15_2',
                                                                     '36_10_0',
                                                                     '32_-18_3',
                                                                     '15_21_52',
                                                                     '-42_-56_-38']].values
complete_otp_mean_bal_dcd_vs_com,\
    complete_otp_real_bal_dcd_vs_com,\
        complete_groupwise_accuracy_bal_dcd_vs_com,\
            complete_balanced_accuracy_bal_dcd_vs_com,\
                complete_feat_coef_dcd_vs_com  = SVC_CV(complete_in_matrix,
                                                             complete_labels, 
                                                             r_seed = 100, 
                                                             balancing = "balanced")

complete_suff_mean_acc_dcd_vs_com,\
    complete_shuff_bal_mean_acc_dcd_vs_com,\
        complete_shuff_groupwise_acc_dcd_vs_com = SVC_CV_shuff(complete_in_matrix,
                                                                    complete_labels, 
                                                                    r_seed = 100, balancing = "balanced")

calculate_groupwise_significance(complete_groupwise_accuracy_bal_dcd_vs_com, complete_shuff_groupwise_acc_dcd_vs_com)

with open('SVC_linear_results_with_coef_two_by_two_03062021.pkl', 'wb') as f:
    pkl.dump([gm_groupwise_accuracy_bal_dcd_vs_dys, gm_shuff_groupwise_acc_dcd_vs_dys, gm_feat_coef_dcd_vs_dys,
              wm_groupwise_accuracy_bal_dcd_vs_dys, wm_shuff_groupwise_acc_dcd_vs_dys, wm_feat_coef_dcd_vs_dys,
              falff_groupwise_accuracy_bal_dcd_vs_dys, falff_shuff_groupwise_acc_dcd_vs_dys, falff_feat_coef_dcd_vs_dys,
              localcorr_groupwise_accuracy_bal_dcd_vs_dys, localcorr_shuff_groupwise_acc_dcd_vs_dys, localcorr_feat_coef_dcd_vs_dys,
              globalcorr_groupwise_accuracy_bal_dcd_vs_dys, globalcorr_shuff_groupwise_acc_dcd_vs_dys, globalcorr_feat_coef_dcd_vs_dys,
              structural_groupwise_accuracy_bal_dcd_vs_dys, structural_shuff_groupwise_acc_dcd_vs_dys, structural_feat_coef_dcd_vs_dys,
              functional_groupwise_accuracy_bal_dcd_vs_dys, functional_shuff_groupwise_acc_dcd_vs_dys, functional_feat_coef_dcd_vs_dys,
              complete_groupwise_accuracy_bal_dcd_vs_dys, complete_shuff_groupwise_acc_dcd_vs_dys, complete_feat_coef_dcd_vs_dys,
              gm_groupwise_accuracy_bal_dys_vs_com, gm_shuff_groupwise_acc_dys_vs_com, gm_feat_coef_dys_vs_com,
              wm_groupwise_accuracy_bal_dys_vs_com, wm_shuff_groupwise_acc_dys_vs_com, wm_feat_coef_dys_vs_com,
              falff_groupwise_accuracy_bal_dys_vs_com, falff_shuff_groupwise_acc_dys_vs_com, falff_feat_coef_dys_vs_com,
              localcorr_groupwise_accuracy_bal_dys_vs_com, localcorr_shuff_groupwise_acc_dys_vs_com, localcorr_feat_coef_dys_vs_com,
              globalcorr_groupwise_accuracy_bal_dys_vs_com, globalcorr_shuff_groupwise_acc_dys_vs_com, globalcorr_feat_coef_dys_vs_com,
              structural_groupwise_accuracy_bal_dys_vs_com, structural_shuff_groupwise_acc_dys_vs_com, structural_feat_coef_dys_vs_com,
              functional_groupwise_accuracy_bal_dys_vs_com, functional_shuff_groupwise_acc_dys_vs_com, functional_feat_coef_dys_vs_com,
              complete_groupwise_accuracy_bal_dys_vs_com, complete_shuff_groupwise_acc_dys_vs_com, complete_feat_coef_dys_vs_com,
              gm_groupwise_accuracy_bal_dcd_vs_com, gm_shuff_groupwise_acc_dcd_vs_com, gm_feat_coef_dcd_vs_com,
              wm_groupwise_accuracy_bal_dcd_vs_com, wm_shuff_groupwise_acc_dcd_vs_com, wm_feat_coef_dcd_vs_com,
              falff_groupwise_accuracy_bal_dcd_vs_com, falff_shuff_groupwise_acc_dcd_vs_com, falff_feat_coef_dcd_vs_com,
              localcorr_groupwise_accuracy_bal_dcd_vs_com, localcorr_shuff_groupwise_acc_dcd_vs_com, localcorr_feat_coef_dcd_vs_com,
              globalcorr_groupwise_accuracy_bal_dcd_vs_com, globalcorr_shuff_groupwise_acc_dcd_vs_com, globalcorr_feat_coef_dcd_vs_com,
              structural_groupwise_accuracy_bal_dcd_vs_com, structural_shuff_groupwise_acc_dcd_vs_com, structural_feat_coef_dcd_vs_com,
              functional_groupwise_accuracy_bal_dcd_vs_com, functional_shuff_groupwise_acc_dcd_vs_com, functional_feat_coef_dcd_vs_com,
              complete_groupwise_accuracy_bal_dcd_vs_com, complete_shuff_groupwise_acc_dcd_vs_com, complete_feat_coef_dcd_vs_com], f)
