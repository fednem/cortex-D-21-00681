# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:53:00 2020

@author: federico.nemmi
"""

import pickle as pkl
import numpy as np
from helper_functions import calculate_groupwise_significance
import scipy.stats
with open("D:/corona2020/Python Scripts_/DysTacMap/SVC_results_18112020.pkl", "rb") as f:  
    gm_groupwise_accuracy_bal, gm_rbf_groupwise_accuracy_bal, gm_shuff_groupwise_acc, gm_rbf_shuff_groupwise_acc, \
    wm_groupwise_accuracy_bal, wm_rbf_groupwise_accuracy_bal, wm_shuff_groupwise_acc, wm_rbf_shuff_groupwise_acc, \
    falff_groupwise_accuracy_bal, falff_rbf_groupwise_accuracy_bal, falff_shuff_groupwise_acc, falff_rbf_shuff_groupwise_acc, \
    localcorr_groupwise_accuracy_bal, localcorr_rbf_groupwise_accuracy_bal, localcorr_shuff_groupwise_acc, localcorr_rbf_shuff_groupwise_acc, \
    globalcorr_groupwise_accuracy_bal, globalcorr_rbf_groupwise_accuracy_bal, globalcorr_shuff_groupwise_acc, globalcorr_rbf_shuff_groupwise_acc, \
    structural_groupwise_accuracy_bal, structural_rbf_groupwise_accuracy_bal, structural_shuff_groupwise_acc, structural_rbf_shuff_groupwise_acc, \
    functional_groupwise_accuracy_bal, functional_rbf_groupwise_accuracy_bal, functional_shuff_groupwise_acc, functional_rbf_shuff_groupwise_acc, \
    complete_groupwise_accuracy_bal, complete_rbf_groupwise_accuracy_bal, complete_shuff_groupwise_acc, complete_rbf_shuff_groupwise_acc = pkl.load(f)

gm_signif = calculate_groupwise_significance(gm_groupwise_accuracy_bal, gm_shuff_groupwise_acc)

gm_rbf_signif = calculate_groupwise_significance(gm_rbf_groupwise_accuracy_bal, gm_rbf_shuff_groupwise_acc)


wm_signif = calculate_groupwise_significance(wm_groupwise_accuracy_bal, wm_shuff_groupwise_acc)
wm_rbf_signif = calculate_groupwise_significance(wm_rbf_groupwise_accuracy_bal, wm_rbf_shuff_groupwise_acc)


falff_signif = calculate_groupwise_significance(falff_groupwise_accuracy_bal, falff_shuff_groupwise_acc)
falff_rbf_signif = calculate_groupwise_significance(falff_rbf_groupwise_accuracy_bal, falff_rbf_shuff_groupwise_acc)


localcorr_signif = calculate_groupwise_significance(localcorr_groupwise_accuracy_bal, localcorr_shuff_groupwise_acc)
localcorr_rbf_signif = calculate_groupwise_significance(localcorr_rbf_groupwise_accuracy_bal, localcorr_rbf_shuff_groupwise_acc)


globalcorr_signif = calculate_groupwise_significance(globalcorr_groupwise_accuracy_bal, globalcorr_shuff_groupwise_acc)
globalcorr_rbf_signif = calculate_groupwise_significance(globalcorr_rbf_groupwise_accuracy_bal, globalcorr_rbf_shuff_groupwise_acc)


structural_signif = calculate_groupwise_significance(structural_groupwise_accuracy_bal, structural_shuff_groupwise_acc)
structural_rbf_signif = calculate_groupwise_significance(structural_rbf_groupwise_accuracy_bal, structural_rbf_shuff_groupwise_acc)


functional_signif = calculate_groupwise_significance(functional_groupwise_accuracy_bal, functional_shuff_groupwise_acc)
functional_rbf_signif = calculate_groupwise_significance(functional_rbf_groupwise_accuracy_bal, functional_rbf_shuff_groupwise_acc)


complete_signif = calculate_groupwise_significance(complete_groupwise_accuracy_bal, complete_shuff_groupwise_acc)
complete_rbf_signif = calculate_groupwise_significance(complete_rbf_groupwise_accuracy_bal, complete_rbf_shuff_groupwise_acc)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

np.apply_along_axis(mean_confidence_interval, 0, gm_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, wm_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, falff_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, globalcorr_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, localcorr_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, structural_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, functional_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, complete_groupwise_accuracy_bal)