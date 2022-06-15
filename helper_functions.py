# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:25:16 2018

@author: federico nemmi
"""
import scipy as sp
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from scipy.stats import itemfreq
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif, SelectFromModel
from sklearn.svm import LinearSVC, NuSVC
from scipy.stats import ttest_1samp
from collections import namedtuple
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

def calculate_groupwise_significance(true_groupwise, shuffled_groupwise):
    p_value = ((shuffled_groupwise > true_groupwise.mean(0)).sum(1) != 0).sum()/shuffled_groupwise.shape[0]
    return(true_groupwise.mean(0), p_value)
    
def get_residuals(df, dependent_variables, independent_variables):
    dependent_variables_as_array = df[dependent_variables].values
    independent_variables_as_array = df[independent_variables].values
    linreg = LinearRegression()
    linreg.fit(independent_variables_as_array, dependent_variables_as_array)
    residuals = dependent_variables_as_array - linreg.predict(independent_variables_as_array)
    return(residuals)



def f_score_recursive (features, labels):
    f_score = f_classif(features, labels)[0].argsort()[::-1]
    clf = GaussianNB()
    accs = []
    for n in np.arange(features.shape[1]):
        clf.fit(features[:,f_score[:n + 1]], labels)
        pred = clf.predict(features[:,:n + 1])
        conf = confusion_matrix(labels, pred)
        bal_acc = (np.diag(conf)/conf.sum(1)).mean()
        accs.append(bal_acc)
        best_combo_index = np.array(accs).argsort()[::-1][0]
        if best_combo_index == 0:
            best_combo = f_score[0]
        else:
            best_combo = f_score[0:np.array(accs).argsort()[::-1][0]]
    
    return(best_combo)
        
        
    

def SVC_CV(features, 
           labels, 
           r_seed, 
           n_fold = 10, 
           n_rep = 10,
           balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    otp_real = np.empty([n_fold * n_rep, 1])
    
    group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
    
    balanced_accuracy = np.empty([n_fold * n_rep, 1])
    
    feat_coef = np.empty(labels.unique().shape + (features.shape[1],) + (n_fold * n_rep,))
    
    
    rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                   n_repeats = n_rep, 
                                   random_state = r_seed)
    #loop over il numero di bootstrapping
    n = 0
    for train_index, test_index in rsfk.split(features, labels):
        print ("Iteration advancement = {0:3.2f}%".format((n/(n_fold * n_rep))*100))
        #il while loop è qui per accertarsi che ci sia almeno un esemplare di ciascuna categoria nel resampled sample
        X_train = features[train_index,:]
        X_test = features[test_index,:]
        y_train = labels.values[train_index]
        y_test = labels.values[test_index]
                
        #fitta il navie gaussian e predici
        clf = LinearSVC(class_weight = balancing)
        clf.fit(X_train, y_train)
        feat_coef[:,:,n] = clf.coef_
        pred = clf.predict(X_test)
        #calcola l accuracy
        otp_real[n] = accuracy_score(y_test, pred)
        #calcola l accuracy per ciascuna classe
        conf = confusion_matrix(y_test, pred)
        temp_conf = np.diag(conf)/conf.sum(1)
        group_wise_accuracy[n,:] = temp_conf.transpose()
        #calcola la balanced accuracy
        balanced_accuracy[n] = temp_conf.mean()
        n = n + 1
        
        
    otp_mean = otp_real.mean()
    
    
    
    
    
    return otp_mean, otp_real, group_wise_accuracy, balanced_accuracy, feat_coef



def SVC_rbf_CV(features, 
           labels, 
           r_seed, 
           n_fold = 10, 
           n_rep = 10,
           balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    otp_real = np.empty([n_fold * n_rep, 1])
    
    group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
    
    balanced_accuracy = np.empty([n_fold * n_rep, 1])
    
    
    
    
    
    rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                   n_repeats = n_rep, 
                                   random_state = r_seed)
    #loop over il numero di bootstrapping
    n = 0
    for train_index, test_index in rsfk.split(features, labels):
        print ("Iteration advancement = {0:3.2f}%".format((n/(n_fold * n_rep))*100))
        #il while loop è qui per accertarsi che ci sia almeno un esemplare di ciascuna categoria nel resampled sample
        X_train = features[train_index,:]
        X_test = features[test_index,:]
        y_train = labels.values[train_index]
        y_test = labels.values[test_index]
                
        #fitta il navie gaussian e predici
        clf = NuSVC(class_weight = balancing)
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_test)
        #calcola l accuracy
        otp_real[n] = accuracy_score(y_test, pred)
        #calcola l accuracy per ciascuna classe
        conf = confusion_matrix(y_test, pred)
        temp_conf = np.diag(conf)/conf.sum(1)
        group_wise_accuracy[n,:] = temp_conf.transpose()
        #calcola la balanced accuracy
        balanced_accuracy[n] = temp_conf.mean()
        n = n + 1
        
        
    otp_mean = otp_real.mean()
    
    
    
    
    
    return otp_mean, otp_real, group_wise_accuracy, balanced_accuracy


def SVC_rbf_upsampling_CV(features, 
           labels, 
           r_seed, 
           n_fold = 10, 
           n_rep = 10,
           balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    otp_real = np.empty([n_fold * n_rep, 1])
    
    group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
    
    balanced_accuracy = np.empty([n_fold * n_rep, 1])
    
    
    feat_coef = np.empty(labels.unique().shape + (features.shape[1],) + (n_fold * n_rep,))
    
    
    rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                   n_repeats = n_rep, 
                                   random_state = r_seed)
    #loop over il numero di bootstrapping
    n = 0
    for train_index, test_index in rsfk.split(features, labels):
        print ("Iteration advancement = {0:3.2f}%".format((n/(n_fold * n_rep))*100))
        #il while loop è qui per accertarsi che ci sia almeno un esemplare di ciascuna categoria nel resampled sample
        X_train = features[train_index,:]
        X_test = features[test_index,:]
        y_train = labels.values[train_index]
        y_test = labels.values[test_index]
        labels_train, labels_count = np.unique(y_train, return_counts = True)
        label_max = labels_train[labels_count.argmax()] 
        label_max_indexes = np.where(y_train == label_max)[0]
        n_index_max = (y_train == label_max).sum()
        second_label = labels_train[labels_count.argsort()[0]]
        second_label_indexes = np.where(y_train == second_label)[0]
        third_label = labels_train[labels_count.argsort()[1]]
        third_label_indexes = np.where(y_train == third_label)[0]
        upsampled_second_label_indexes = resample(second_label_indexes,
                                                  replace = True,
                                                  n_samples = n_index_max)
        upsampled_third_label_indexes = resample(third_label_indexes,
                                                  replace = True,
                                                  n_samples = n_index_max)
        all_indexes = np.concatenate([label_max_indexes, 
                                      upsampled_second_label_indexes, 
                                      upsampled_third_label_indexes])
        X_train_upsampled = X_train[all_indexes,:]
        y_train_upsampled = np.concatenate([np.repeat(label_max, n_index_max),
                                       np.repeat(second_label, n_index_max),
                                       np.repeat(third_label, n_index_max)])
                                                 
        
        #fitta il navie gaussian e predici
        clf = NuSVC(class_weight = balancing)
        clf.fit(X_train_upsampled, y_train_upsampled)
        feat_coef[:,:,n] = clf.coef_
        pred = clf.predict(X_test)
        #calcola l accuracy
        otp_real[n] = accuracy_score(y_test, pred)
        #calcola l accuracy per ciascuna classe
        conf = confusion_matrix(y_test, pred)
        temp_conf = np.diag(conf)/conf.sum(1)
        group_wise_accuracy[n,:] = temp_conf.transpose()
        #calcola la balanced accuracy
        balanced_accuracy[n] = temp_conf.mean()
        n = n + 1
        
        
    otp_mean = otp_real.mean()
    
    
    
    
    
    return otp_mean, otp_real, group_wise_accuracy, balanced_accuracy, feat_coef




def SVC_upsampling_CV(features, 
           labels, 
           r_seed, 
           n_fold = 10, 
           n_rep = 10,
           balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    otp_real = np.empty([n_fold * n_rep, 1])
    
    group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
    
    balanced_accuracy = np.empty([n_fold * n_rep, 1])
    
    feat_coef = np.empty(labels.unique().shape + (features.shape[1],) + (n_fold * n_rep,))
    
    
    
    rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                   n_repeats = n_rep, 
                                   random_state = r_seed)
    #loop over il numero di bootstrapping
    n = 0
    for train_index, test_index in rsfk.split(features, labels):
        print ("Iteration advancement = {0:3.2f}%".format((n/(n_fold * n_rep))*100))
        #il while loop è qui per accertarsi che ci sia almeno un esemplare di ciascuna categoria nel resampled sample
        X_train = features[train_index,:]
        X_test = features[test_index,:]
        y_train = labels.values[train_index]
        y_test = labels.values[test_index]
        labels_train, labels_count = np.unique(y_train, return_counts = True)
        label_max = labels_train[labels_count.argmax()] 
        label_max_indexes = np.where(y_train == label_max)[0]
        n_index_max = (y_train == label_max).sum()
        second_label = labels_train[labels_count.argsort()[0]]
        second_label_indexes = np.where(y_train == second_label)[0]
        third_label = labels_train[labels_count.argsort()[1]]
        third_label_indexes = np.where(y_train == third_label)[0]
        upsampled_second_label_indexes = resample(second_label_indexes,
                                                  replace = True,
                                                  n_samples = n_index_max)
        upsampled_third_label_indexes = resample(third_label_indexes,
                                                  replace = True,
                                                  n_samples = n_index_max)
        all_indexes = np.concatenate([label_max_indexes, 
                                      upsampled_second_label_indexes, 
                                      upsampled_third_label_indexes])
        X_train_upsampled = X_train[all_indexes,:]
        y_train_upsampled = np.concatenate([np.repeat(label_max, n_index_max),
                                       np.repeat(second_label, n_index_max),
                                       np.repeat(third_label, n_index_max)])
                                                 
        
        #fitta il navie gaussian e predici
        clf = LinearSVC(class_weight = balancing)
        clf.fit(X_train_upsampled, y_train_upsampled)
        feat_coef[:,:,n] = clf.coef_
        pred = clf.predict(X_test)
        #calcola l accuracy
        otp_real[n] = accuracy_score(y_test, pred)
        #calcola l accuracy per ciascuna classe
        conf = confusion_matrix(y_test, pred)
        temp_conf = np.diag(conf)/conf.sum(1)
        group_wise_accuracy[n,:] = temp_conf.transpose()
        #calcola la balanced accuracy
        balanced_accuracy[n] = temp_conf.mean()
        n = n + 1
        
        
    otp_mean = otp_real.mean()
    
    
    
    
    
    return otp_mean, otp_real, group_wise_accuracy, balanced_accuracy, feat_coef

def RF_CV(features, 
           labels, 
           r_seed, 
           n_fold = 10, 
           n_rep = 10,
           n_trees = 1000):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    otp_real = np.empty([n_fold * n_rep, 1])
    
    group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
    
    balanced_accuracy = np.empty([n_fold * n_rep, 1])
    
    feature_importance = np.empty([n_fold * n_rep, features.shape[1]])
    
    
    
    
    rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                   n_repeats = n_rep, 
                                   random_state = r_seed)
    #loop over il numero di bootstrapping
    n = 0
    for train_index, test_index in rsfk.split(features, labels):
        print ("Iteration advancement = {0:3.2f}%".format((n/(n_fold * n_rep))*100))
        #il while loop è qui per accertarsi che ci sia almeno un esemplare di ciascuna categoria nel resampled sample
        X_train = features[train_index,:]
        X_test = features[test_index,:]
        y_train = labels.values[train_index]
        y_test = labels.values[test_index]
                
        #fitta il navie gaussian e predici
        clf = RandomForestClassifier(n_estimators = n_trees,
                                     class_weight = "balanced_subsample",
                                     n_jobs = 3)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        #calcola l accuracy
        otp_real[n] = accuracy_score(y_test, pred)
        #calcola l accuracy per ciascuna classe
        conf = confusion_matrix(y_test, pred)
        temp_conf = np.diag(conf)/conf.sum(1)
        group_wise_accuracy[n,:] = temp_conf.transpose()
        #calcola la balanced accuracy
        balanced_accuracy[n] = temp_conf.mean()
        feature_importance[n,:] = clf.feature_importances_
        n = n + 1
        
        
    otp_mean = otp_real.mean()
    
    
    
    
    
    return otp_mean, otp_real, group_wise_accuracy, balanced_accuracy, feature_importance

def RF_CV_shuff(features, 
                 labels, 
                 r_seed, 
                 n_fold = 10, 
                 n_rep = 10, 
                 outern_rep = 1000,
                 n_trees = 1000):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    
    outern_mean_acc = np.empty([outern_rep, 1])
    outern_bal_mean_acc = np.empty([outern_rep, 1])
    outern_groupwise_acc = np.empty([outern_rep, len(labels.unique())])

    for rep in np.arange(outern_rep):
        
        print ("Iteration advancement = {0:3.2f}%".format(((rep + 1)/(outern_rep))*100))
        otp_real = np.empty([n_fold * n_rep, 1])
        
        group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
        
        balanced_accuracy = np.empty([n_fold * n_rep, 1])
        
        
        shuff_labels = shuffle(labels)
        
        rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                       n_repeats = n_rep, 
                                       random_state = r_seed)
        #loop over il numero di bootstrapping
        n = 0
        for train_index, test_index in rsfk.split(features, shuff_labels):
            
            
            X_train = features[train_index,:]
            X_test = features[test_index,:]
            y_train = shuff_labels.values[train_index]
            y_test = shuff_labels.values[test_index]
                    
            #fitta il navie gaussian e predici
            clf = RandomForestClassifier(n_estimators = n_trees,
                                     class_weight = "balanced_subsample",
                                     n_jobs = 3)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            #calcola l accuracy
            otp_real[n] = accuracy_score(y_test, pred)
            #calcola l accuracy per ciascuna classe
            conf = confusion_matrix(y_test, pred)
            temp_conf = np.diag(conf)/conf.sum(1)
            group_wise_accuracy[n,:] = temp_conf.transpose()
            #calcola la balanced accuracy
            balanced_accuracy[n] = temp_conf.mean()
            n = n + 1
            
            
        otp_mean = otp_real.mean()
        balanced_accuracy_mean = balanced_accuracy.mean()
        groupwise_accuracy_mean = group_wise_accuracy.mean(0)
        outern_mean_acc[rep] = otp_mean
        outern_bal_mean_acc[rep] = balanced_accuracy_mean
        outern_groupwise_acc[rep] = groupwise_accuracy_mean
        
        
    return outern_mean_acc, outern_bal_mean_acc, outern_groupwise_acc



def SVC_CV_shuff(features, 
                 labels, 
                 r_seed, 
                 n_fold = 10, 
                 n_rep = 10, 
                 outern_rep = 1000,
                 balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    
    outern_mean_acc = np.empty([outern_rep, 1])
    outern_bal_mean_acc = np.empty([outern_rep, 1])
    outern_groupwise_acc = np.empty([outern_rep, len(labels.unique())])

    for rep in np.arange(outern_rep):
        
        print ("Iteration advancement = {0:3.2f}%".format(((rep + 1)/(outern_rep))*100))
        otp_real = np.empty([n_fold * n_rep, 1])
        
        group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
        
        balanced_accuracy = np.empty([n_fold * n_rep, 1])
        
        
        shuff_labels = shuffle(labels)
        
        rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                       n_repeats = n_rep, 
                                       random_state = r_seed)
        #loop over il numero di bootstrapping
        n = 0
        for train_index, test_index in rsfk.split(features, shuff_labels):
            
            
            X_train = features[train_index,:]
            X_test = features[test_index,:]
            y_train = shuff_labels.values[train_index]
            y_test = shuff_labels.values[test_index]
                    
            #fitta il navie gaussian e predici
            clf = LinearSVC(class_weight = balancing)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            #calcola l accuracy
            otp_real[n] = accuracy_score(y_test, pred)
            #calcola l accuracy per ciascuna classe
            conf = confusion_matrix(y_test, pred)
            temp_conf = np.diag(conf)/conf.sum(1)
            group_wise_accuracy[n,:] = temp_conf.transpose()
            #calcola la balanced accuracy
            balanced_accuracy[n] = temp_conf.mean()
            n = n + 1
            
            
        otp_mean = otp_real.mean()
        balanced_accuracy_mean = balanced_accuracy.mean()
        groupwise_accuracy_mean = group_wise_accuracy.mean(0)
        outern_mean_acc[rep] = otp_mean
        outern_bal_mean_acc[rep] = balanced_accuracy_mean
        outern_groupwise_acc[rep] = groupwise_accuracy_mean
        
        
    return outern_mean_acc, outern_bal_mean_acc, outern_groupwise_acc

def SVC_rbf_CV_shuff(features, 
                 labels, 
                 r_seed, 
                 n_fold = 10, 
                 n_rep = 10, 
                 outern_rep = 1000,
                 balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    
    outern_mean_acc = np.empty([outern_rep, 1])
    outern_bal_mean_acc = np.empty([outern_rep, 1])
    outern_groupwise_acc = np.empty([outern_rep, len(labels.unique())])

    for rep in np.arange(outern_rep):
        
        print ("Iteration advancement = {0:3.2f}%".format(((rep + 1)/(outern_rep))*100))
        otp_real = np.empty([n_fold * n_rep, 1])
        
        group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
        
        balanced_accuracy = np.empty([n_fold * n_rep, 1])
        
        
        shuff_labels = shuffle(labels)
        
        rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                       n_repeats = n_rep, 
                                       random_state = r_seed)
        #loop over il numero di bootstrapping
        n = 0
        for train_index, test_index in rsfk.split(features, shuff_labels):
            
            
            X_train = features[train_index,:]
            X_test = features[test_index,:]
            y_train = shuff_labels.values[train_index]
            y_test = shuff_labels.values[test_index]
                    
            #fitta il navie gaussian e predici
            clf = NuSVC(class_weight = balancing)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            #calcola l accuracy
            otp_real[n] = accuracy_score(y_test, pred)
            #calcola l accuracy per ciascuna classe
            conf = confusion_matrix(y_test, pred)
            temp_conf = np.diag(conf)/conf.sum(1)
            group_wise_accuracy[n,:] = temp_conf.transpose()
            #calcola la balanced accuracy
            balanced_accuracy[n] = temp_conf.mean()
            n = n + 1
            
            
        otp_mean = otp_real.mean()
        balanced_accuracy_mean = balanced_accuracy.mean()
        groupwise_accuracy_mean = group_wise_accuracy.mean(0)
        outern_mean_acc[rep] = otp_mean
        outern_bal_mean_acc[rep] = balanced_accuracy_mean
        outern_groupwise_acc[rep] = groupwise_accuracy_mean
        
        
    return outern_mean_acc, outern_bal_mean_acc, outern_groupwise_acc


def SVC_rbf_upsampling_CV_shuff(features, 
                 labels, 
                 r_seed, 
                 n_fold = 10, 
                 n_rep = 10, 
                 outern_rep = 1000,
                 balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    
    outern_mean_acc = np.empty([outern_rep, 1])
    outern_bal_mean_acc = np.empty([outern_rep, 1])
    outern_groupwise_acc = np.empty([outern_rep, len(labels.unique())])

    for rep in np.arange(outern_rep):
        
        print ("Iteration advancement = {0:3.2f}%".format(((rep + 1)/(outern_rep))*100))
        otp_real = np.empty([n_fold * n_rep, 1])
        
        group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
        
        balanced_accuracy = np.empty([n_fold * n_rep, 1])
        
        
        shuff_labels = shuffle(labels)
        
        rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                       n_repeats = n_rep, 
                                       random_state = r_seed)
        #loop over il numero di bootstrapping
        n = 0
        for train_index, test_index in rsfk.split(features, shuff_labels):
            
            
            X_train = features[train_index,:]
            X_test = features[test_index,:]
            y_train = shuff_labels.values[train_index]
            y_test = shuff_labels.values[test_index]
            labels_train, labels_count = np.unique(y_train, return_counts = True)
            label_max = labels_train[labels_count.argmax()] 
            label_max_indexes = np.where(y_train == label_max)[0]
            n_index_max = (y_train == label_max).sum()
            second_label = labels_train[labels_count.argsort()[0]]
            second_label_indexes = np.where(y_train == second_label)[0]
            third_label = labels_train[labels_count.argsort()[1]]
            third_label_indexes = np.where(y_train == third_label)[0]
            upsampled_second_label_indexes = resample(second_label_indexes,
                                                      replace = True,
                                                      n_samples = n_index_max)
            upsampled_third_label_indexes = resample(third_label_indexes,
                                                      replace = True,
                                                      n_samples = n_index_max)
            all_indexes = np.concatenate([label_max_indexes, 
                                          upsampled_second_label_indexes, 
                                          upsampled_third_label_indexes])
            X_train_upsampled = X_train[all_indexes,:]
            y_train_upsampled = np.concatenate([np.repeat(label_max, n_index_max),
                                           np.repeat(second_label, n_index_max),
                                           np.repeat(third_label, n_index_max)])
                                                 
        
                    
            #fitta il navie gaussian e predici
            clf = NuSVC(class_weight = balancing)
            clf.fit(X_train_upsampled, y_train_upsampled)
            pred = clf.predict(X_test)
            #calcola l accuracy
            otp_real[n] = accuracy_score(y_test, pred)
            #calcola l accuracy per ciascuna classe
            conf = confusion_matrix(y_test, pred)
            temp_conf = np.diag(conf)/conf.sum(1)
            group_wise_accuracy[n,:] = temp_conf.transpose()
            #calcola la balanced accuracy
            balanced_accuracy[n] = temp_conf.mean()
            n = n + 1
            
            
        otp_mean = otp_real.mean()
        balanced_accuracy_mean = balanced_accuracy.mean()
        groupwise_accuracy_mean = group_wise_accuracy.mean(0)
        outern_mean_acc[rep] = otp_mean
        outern_bal_mean_acc[rep] = balanced_accuracy_mean
        outern_groupwise_acc[rep] = groupwise_accuracy_mean
        
        
    return outern_mean_acc, outern_bal_mean_acc, outern_groupwise_acc
        

def SVC_upsampling_CV_shuff(features, 
                 labels, 
                 r_seed, 
                 n_fold = 10, 
                 n_rep = 10, 
                 outern_rep = 1000,
                 balancing = "balanced"):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    
    outern_mean_acc = np.empty([outern_rep, 1])
    outern_bal_mean_acc = np.empty([outern_rep, 1])
    outern_groupwise_acc = np.empty([outern_rep, len(labels.unique())])

    for rep in np.arange(outern_rep):
        
        print ("Iteration advancement = {0:3.2f}%".format(((rep + 1)/(outern_rep))*100))
        otp_real = np.empty([n_fold * n_rep, 1])
        
        group_wise_accuracy = np.empty([n_fold * n_rep, len(labels.unique())])
        
        balanced_accuracy = np.empty([n_fold * n_rep, 1])
        
        
        shuff_labels = shuffle(labels)
        
        rsfk = RepeatedStratifiedKFold(n_splits = n_fold, 
                                       n_repeats = n_rep, 
                                       random_state = r_seed)
        #loop over il numero di bootstrapping
        n = 0
        for train_index, test_index in rsfk.split(features, shuff_labels):
            
            
            X_train = features[train_index,:]
            X_test = features[test_index,:]
            y_train = shuff_labels.values[train_index]
            y_test = shuff_labels.values[test_index]
            labels_train, labels_count = np.unique(y_train, return_counts = True)
            label_max = labels_train[labels_count.argmax()] 
            label_max_indexes = np.where(y_train == label_max)[0]
            n_index_max = (y_train == label_max).sum()
            second_label = labels_train[labels_count.argsort()[0]]
            second_label_indexes = np.where(y_train == second_label)[0]
            third_label = labels_train[labels_count.argsort()[1]]
            third_label_indexes = np.where(y_train == third_label)[0]
            upsampled_second_label_indexes = resample(second_label_indexes,
                                                      replace = True,
                                                      n_samples = n_index_max)
            upsampled_third_label_indexes = resample(third_label_indexes,
                                                      replace = True,
                                                      n_samples = n_index_max)
            all_indexes = np.concatenate([label_max_indexes, 
                                          upsampled_second_label_indexes, 
                                          upsampled_third_label_indexes])
            X_train_upsampled = X_train[all_indexes,:]
            y_train_upsampled = np.concatenate([np.repeat(label_max, n_index_max),
                                           np.repeat(second_label, n_index_max),
                                           np.repeat(third_label, n_index_max)])
                                                 
        
                    
            #fitta il navie gaussian e predici
            clf = LinearSVC(class_weight = balancing)
            clf.fit(X_train_upsampled, y_train_upsampled)
            pred = clf.predict(X_test)
            #calcola l accuracy
            otp_real[n] = accuracy_score(y_test, pred)
            #calcola l accuracy per ciascuna classe
            conf = confusion_matrix(y_test, pred)
            temp_conf = np.diag(conf)/conf.sum(1)
            group_wise_accuracy[n,:] = temp_conf.transpose()
            #calcola la balanced accuracy
            balanced_accuracy[n] = temp_conf.mean()
            n = n + 1
            
            
        otp_mean = otp_real.mean()
        balanced_accuracy_mean = balanced_accuracy.mean()
        groupwise_accuracy_mean = group_wise_accuracy.mean(0)
        outern_mean_acc[rep] = otp_mean
        outern_bal_mean_acc[rep] = balanced_accuracy_mean
        outern_groupwise_acc[rep] = groupwise_accuracy_mean
        
        
    return outern_mean_acc, outern_bal_mean_acc, outern_groupwise_acc

def plot_real_and_shuffled_results(real_mean, real_std, shuffled_mean, shuffled_std, bar_width = .35):
    n_bar = 1
    index = np.arange(n_bar)
    bar_width = .35
    fig, ax = plt.subplots()
    rects1 = ax.bar(index, real_mean, bar_width, color='r', yerr=real_std)
    rects2 = ax.bar(index + bar_width, shuffled_mean, bar_width, color='y', yerr=shuffled_std)
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification performance')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels((''))
    ax.legend((rects1[0], rects2[0]), ('Real Data', 'Shuffled Data'))
    
    
def permuted_between_class_prediction_significance(within_features, within_labels, original_label_for_between_class,
                                                   between_features, r_seed, non_permuted_accuracy,
                                                   n_perms = 5000):
    otp = np.empty([n_perms,1])
    
    for n in range(1,n_perms):
        features_shuffled = shuffle(within_features, random_state = r_seed + n)
        clf = GaussianNB()
        clf.fit(features_shuffled, within_labels)
        pred = clf.predict(between_features)
        acc = accuracy_score(np.repeat(original_label_for_between_class, between_features.shape[0]), pred)
        otp[n] = acc
        
    bigger_than_accuracy = len([val for val in otp if val >= non_permuted_accuracy])/n_perms
    return bigger_than_accuracy, otp
        



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def logReg_nested_CV(x_train, y_train, params_to_cv, cv_scheme = 10, n_jobs = 10):
    logReg = LogisticRegression(penalty = "l1", solver = "saga", multi_class = "multinomial",  max_iter = 5000)
    clf = GridSearchCV(logReg, params_to_cv, cv = cv_scheme, n_jobs = 10)
    clf.fit(x_train, y_train)
    return clf.best_estimator_
    




#calcola acuracy e balanced accuracy con bootstrap
#NB devi importare tutti  i valori dal modulo params
def logReg_real_and_shuffled_bootstrap__(features, labels, subj_id, r_seed, n_perms = 5000, test_size = .25):
    import params as p
    print(p.params_to_cv)
    print(p.n_jobs)
    print(p.cv_scheme)
    #crea tutti gli array vuoit che verranno popolati nella funzione
    otp_real = np.empty([n_perms, 1])
    otp_shuffled = np.empty([n_perms, 1])
    group_wise_accuracy = np.empty([n_perms, len(labels.unique())])
    group_wise_accuracy_shuffled = np.empty([n_perms, len(labels.unique())])
    balanced_accuracy = np.empty([n_perms, 1])
    balanced_accuracy_shuffled = np.empty([n_perms, 1])
    coef = np.ndarray((n_perms, len(labels.unique()), features.shape[1]))
    
    #loop over il numero di bootstrapping
    for n in range(0,n_perms):
        print(str(n))
        #il while loop è qui per accertarsi che ci sia almeno un esemplare di ciascuna categoria nel resampled sample
        test = 0
        while test == 0:
            res_features, res_labels, res_subj = resample(features, labels, subj_id, random_state = r_seed + n)
            test = all(itemfreq(res_labels)[:,1] > 0) * 1
        #dividi in train and testing sets
        
        X_train = res_features
        y_train = res_labels
        oob_subj = [s for s in subj_id if s not in res_subj]
        X_test = np.array([features[n,:] for n in np.arange(features.shape[0]) if n in oob_subj])
        y_test = [labels.iloc[n] for n in np.arange(features.shape[0]) if n in oob_subj]
        
        #fitta il cross validatori per logReg
        best_model = logReg_nested_CV(X_train, y_train, p.params_to_cv, p.cv_scheme, p.n_jobs)
        #fitta il best model con il sample intero
        best_model.fit(X_train, y_train)
        pred = best_model.predict(X_test)
        #calcola l accuracy
        otp_real[n] = accuracy_score(y_test, pred)
        #calcola l accuracy per ciascuna classe
        conf = confusion_matrix(y_test, pred)
        temp_conf = np.diag(conf)/sum(conf,1)
        group_wise_accuracy[n,:] = temp_conf.transpose()
        #calcola la balanced accuracy
        balanced_accuracy[n] = temp_conf.mean()
        coef[n,:,:] = best_model.coef_
    
    

    
        X_train = shuffle(X_train, random_state = r_seed + n)
        best_model = logReg_nested_CV(X_train, y_train, p.params_to_cv, p.cv_scheme, p.n_jobs)
        #fitta il best model con il sample intero
        best_model.fit(X_train, y_train)
        pred = best_model.predict(X_test)

        otp_shuffled[n] = accuracy_score(y_test, pred)
        conf = confusion_matrix(y_test, pred)
        temp_conf_shuffled = np.diag(conf)/sum(conf,1)
        group_wise_accuracy_shuffled[n,:] = temp_conf_shuffled.transpose()
        balanced_accuracy_shuffled[n] = temp_conf_shuffled.mean()
        
    otp_mean = otp_real.mean()
    otp_std = otp_real.std()    
    otp_shuffled_mean = otp_shuffled.mean()
    otp_shuffled_std = otp_shuffled.std()
    
    
    
    return otp_mean, otp_std, otp_shuffled_mean, otp_shuffled_std, otp_real, otp_shuffled, group_wise_accuracy, group_wise_accuracy_shuffled, balanced_accuracy, balanced_accuracy_shuffled, coef


def check_significance_within_accuracies(res_array, expected_value):
    all_significance = namedtuple("all_significance", "test t p")
    t_test = ttest_1samp(res_array, expected_value)
    test = np.logical_and(t_test[0] > 0, t_test[1] < .05)
    res = all_significance(test, t_test[0], t_test[1])
    return(res)
    


        