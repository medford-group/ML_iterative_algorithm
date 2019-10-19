#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:49:26 2019

@author: fuzhu
"""

"""
from Iteration import iteration
I = iteration()
origin_data = 'New_formation_ASS_X.csv' ## The name of structures
new_data = 'New_New_features_Ead.csv' ### the Features+Ead
i, MaxdE_testset, MAE_trainingset, MAE_testset, std_Tr, Num_outliers_percent = I.analyze_data(origin_data,new_data,Num=37,sigma_f=1,alpha=0.07,length_scale=10,max_cycle=200,
                                                                                              std=True,random_state=False,threshold_std=0.35,threshold_dE=1,kk=1)
print ("########iteration#######", i)
print ("MAE_test : %s" % MAE_testset)
"""

"""
from Iteration_self import iteration
origin_data = 'New_formation_ASS_X.csv' ## The name of structures
new_data = 'New_New_features_Ead.csv' ### the Features+Ead

I = iteration(origin_data,new_data,Num=37,alpha=0.07,length_scale=10,max_cycle=200,
              std=True,random_state=False,threshold_std=0.35,threshold_dE=1,kk=1)

i, MaxdE_testset, MAE_testset, std_dE_trainset, Num_outliers = I.analyze_data()
print ("########iteration#######", i)
print ("MAE_test : %s" % MAE_testset)
"""


from Iteration_outliers_self import iteration
origin_data = 'New_formation_ASS_X.csv' ## The name of structures
new_data = 'New_New_features_Ead.csv' ### the Features+Ead

I = iteration(origin_data,new_data,Num=37,alpha=0.07,length_scale=10,max_cycle=200,
              std=True,random_state=False,threshold_std=0.35,threshold_dE=1,kk=1,ntimes=3,outlier_identifier=False)

i, MaxdE_testset, MAE_testset, std_dE_trainset, Num_outliers = I.analyze_data()
print ("########iteration#######", i)
print ("MAE_test : %s" % MAE_testset)