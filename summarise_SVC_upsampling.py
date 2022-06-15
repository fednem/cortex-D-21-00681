# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:53:00 2020

@author: federico.nemmi
"""

import pickle as pkl
import numpy as np
import scipy.stats
from helper_functions import calculate_groupwise_significance
with open("D:/corona2020/Python Scripts_/DysTacMap/SVC_linear_upsampling_results_02112020.pkl", "rb") as f:  
    gm_upsampling_groupwise_accuracy_bal, gm_upsampling_shuff_groupwise_acc, \
    wm_upsampling_groupwise_accuracy_bal, wm_upsampling_shuff_groupwise_acc, \
    falff_upsampling_groupwise_accuracy_bal, falff_upsampling_shuff_groupwise_acc, \
    localcorr_upsampling_groupwise_accuracy_bal, localcorr_upsampling_shuff_groupwise_acc, \
    globalcorr_upsampling_groupwise_accuracy_bal, globalcorr_upsampling_shuff_groupwise_acc, \
    structural_upsampling_groupwise_accuracy_bal, structural_upsampling_shuff_groupwise_acc, \
    functional_upsampling_groupwise_accuracy_bal, functional_upsampling_shuff_groupwise_acc, \
    complete_upsampling_groupwise_accuracy_bal, complete_upsampling_shuff_groupwise_acc = pkl.load(f)



gm_upsampling_signif = calculate_groupwise_significance(gm_upsampling_groupwise_accuracy_bal, gm_upsampling_shuff_groupwise_acc)



wm_upsampling_signif = calculate_groupwise_significance(wm_upsampling_groupwise_accuracy_bal, wm_upsampling_shuff_groupwise_acc)



falff_upsampling_signif = calculate_groupwise_significance(falff_upsampling_groupwise_accuracy_bal, falff_upsampling_shuff_groupwise_acc)



localcorr_upsampling_signif = calculate_groupwise_significance(localcorr_upsampling_groupwise_accuracy_bal, localcorr_upsampling_shuff_groupwise_acc)



globalcorr_upsampling_signif = calculate_groupwise_significance(globalcorr_upsampling_groupwise_accuracy_bal, globalcorr_upsampling_shuff_groupwise_acc)



structural_upsampling_signif = calculate_groupwise_significance(structural_upsampling_groupwise_accuracy_bal, structural_upsampling_shuff_groupwise_acc)



functional_upsampling_signif = calculate_groupwise_significance(functional_upsampling_groupwise_accuracy_bal, functional_upsampling_shuff_groupwise_acc)



complete_upsampling_signif = calculate_groupwise_significance(complete_upsampling_groupwise_accuracy_bal, complete_upsampling_shuff_groupwise_acc)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

np.apply_along_axis(mean_confidence_interval, 0, gm_upsampling_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, wm_upsampling_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, falff_upsampling_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, globalcorr_upsampling_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, localcorr_upsampling_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, structural_upsampling_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, functional_upsampling_groupwise_accuracy_bal)
np.apply_along_axis(mean_confidence_interval, 0, complete_upsampling_groupwise_accuracy_bal)