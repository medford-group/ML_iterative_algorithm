#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:56:58 2019

@author: fuzhu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:36:05 2018

@author: fuzhu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import random
import sys
sys.setrecursionlimit(1000000)

class iteration:
    def __init__(self,origin_data,new_data,Num,alpha=0.2,length_scale=1,n_restarts_optimizer=10,
                 max_cycle=500,std=True,random_state=False,use_all_std=False,threshold_std=0.7,
                 threshold_dE=1,kk=1,outliers=5,ntimes=2):
        self.origin_data = origin_data  # label or structures
        self.new_data = new_data   # data for machine learning regression
        self.Num = Num # initial training set
        self.alpha = alpha  # hyperparameter in GPR
        self.length_scale = length_scale
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_cycle = max_cycle
        self.std = std
        self.random_state = random_state
        self.use_all_std = use_all_std
        self.threshold_std = threshold_std
        self.threshold_dE = threshold_dE
        self.kk = kk
        self.outliers = outliers
        self.ntimes = ntimes
    def analyze_data(self):
        
        #new_data is from feature construction
        # Num is the number of low coverage configurations
        # this output the final configurations!
        df_origin = pd.read_csv(self.origin_data)
        df_origin = df_origin.dropna(axis=0,how='any')
        print (df_origin.shape)
        col_var = df_origin.columns.get_values().tolist()  # 
        col_var.remove(col_var[-1])
        print ('columns_name:',col_var)
        print ("origin_data.shape:",df_origin.shape)
        training_set_AdsSlabSite = df_origin.iloc[:self.Num,:-1].values.tolist()
        #training_set_observations = df_origin.iloc[:Num,-1].tolist()
        test_set_AdsSlabSite = df_origin.iloc[self.Num:,:-1].values.tolist()
        #test_set_observations = df_origin.iloc[Num:,-1].tolist()
        
        # Initialize the parameters:
        df = pd.read_csv(self.new_data)
        print ("new_data.shape:",df.shape)
        X = df.values[:,:-1]  # we predict energy finally
        Y = df.values[:,-1]
        index_of_samples = range(len(df))
        training_set_variables = df.iloc[:self.Num,:-1].values.tolist()
        training_set_observations = df.iloc[:self.Num,-1].tolist()
        test_set_variables = df.iloc[self.Num:,:-1].values.tolist()
        test_set_observations = df.iloc[self.Num:,-1].tolist()
        MaxStd_testset= 5
        MaxdE_testset = 5
        print ("The number of initial training/test set and total samples are [%s, %s, %s]" % (len(training_set_observations),len(test_set_observations),len(index_of_samples)))
        SIGMA = []  # collect the uncertainty of data
        StruOfAdd = [] ####$$$$$$$$
        MaxPredStd = []  #$$$$$$$$$                                       
        for i in range(self.max_cycle):
            run_std = self.std
            if self.std is True:
                if MaxStd_testset > self.threshold_std:
                    management, sigma = self.gpr(X,training_set_variables,training_set_observations,test_set_variables,
                                                 test_set_observations,run_std,training_set_AdsSlabSite,test_set_AdsSlabSite)
                    
                    MaxStd_testset = management['MaxStd_testset']
                    MaxPredStd.append(MaxStd_testset) #$$$$$$
                    std_index = management['std_index']
                    MaxdE_testset = management['MaxdE_testset']
                    training_set_AdsSlabSite = management['training_set_AdsSlabSite']
                    test_set_AdsSlabSite = management['test_set_AdsSlabSite']
                    SIGMA.append(sigma)
                    print ("This is %s_th iteration, MaxdE of test_set is ** %s eV **" % (i,round(MaxdE_testset,3)))
                    print ("This is %s_th iteration, MaxStd of test_set is ** %s eV **" % (i,round(MaxStd_testset,3)))
                    print ('')
                    
                    elements = [test_set_variables[k] for k in std_index]
                    for elem in elements:
                        indx = test_set_variables.index(elem)
                        
                        ##
                        """
                        ads1 = test_set_AdsSlabSite[indx][0]
                        site1 = test_set_AdsSlabSite[indx][1]
                        ads2 = test_set_AdsSlabSite[indx][2]
                        site2 = test_set_AdsSlabSite[indx][3]  
                        ##
                        """
                        
                        training_set_variables.append(elem)
                        training_set_observations.append(test_set_observations[indx])
                        test_set_variables.remove(elem)
                        test_set_observations.remove(test_set_observations[indx])
                        
                        training_set_AdsSlabSite.append(test_set_AdsSlabSite[indx])
                        StruOfAdd.append(test_set_AdsSlabSite[indx]) ###$$$$$$$$$$$$$$$$$
                        test_set_AdsSlabSite.remove(test_set_AdsSlabSite[indx])
                        
                        """
                        ###
                        for list_i in test_set_AdsSlabSite:
                            if (ads2 == list_i[0]) and (site2 == list_i[1]) and (ads1 == list_i[2]) and (site1 == list_i[3]):    
                                ind_same_str = test_set_AdsSlabSite.index(list_i)
                                
                                training_set_AdsSlabSite.append(test_set_AdsSlabSite[ind_same_str])
                                random_testset_variables = test_set_AdsSlabSite[ind_same_str]
                                test_set_AdsSlabSite.remove(random_testset_variables)
                                
                                
                                training_set_variables.append(test_set_variables[ind_same_str])
                                training_set_observations.append(test_set_observations[ind_same_str])
                                
                                random_testset_variables = test_set_variables[ind_same_str]
                                random_testset_observations = test_set_observations[ind_same_str]
                                test_set_variables.remove(random_testset_variables)
                                test_set_observations.remove(random_testset_observations)   
                            else:
                                pass
                        """
                        ##3

                else:
                    #NumOf_dft = len(training_set_variables)
                    print (" # # # # # # # # Iterations Have Convereged # # # # # # # # #")
                    print ("########### Summary ##########:")
                    print ("The number of samples = %s" % len(index_of_samples))
                    print ("The number of converged samples = %s" % len(training_set_variables))
                    print ("After %s times iteration, MaxdE of test_set is ** %s eV **" % ((i-1),round(MaxdE_testset,3)))
                    print ("After %s times iteration, MaxStd of test_set is ** %s eV **" % ((i-1),round(MaxStd_testset,3)))
                                        
                    self.save_AdsSlabSite(col_var,management)
                    self.save_predicted_values(self.origin_data,management)
                    MAE_trainingset, MAE_testset, std_dE_trainset = self.plot_deltaE(management)  ### std_dE_trainset is used to select MPE
                    self.plot_DFT_ML(management)
                    self.plot_emerge(management)
                    self.find_outlier(df,management)
                    self.plot_algorithm(SIGMA)
                    StruOfAdd.append('None')
                    if len(MaxPredStd) < len(StruOfAdd):
                        N = len(StruOfAdd) - len(MaxPredStd)
                        MaxPredStd += N*[MaxPredStd[-1]]
                    fg = pd.DataFrame()
                    fg['max_std'] = MaxPredStd
                    fg['str'] = StruOfAdd
                    indx = range(len(StruOfAdd))
                    fg.to_csv('added_stru.csv',index=indx)
                    
                    print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    print ("You need to check 'added_stru.csv' file to find the bad structures, the index are:")
                    bad_data = self.bad_data_index(MaxPredStd)
                    print (bad_data)
                    if self.outliers == 1:
                        Num_outliers = self.Num_outlier(Y,management)  # dE > threshold_dE
                    if self.outliers == 2:
                        Num_outliers = self.Num_outlier_percent(Y,management)  # dE > std
                        print ("outliers == %s is for the all data, the definition of outlier is dE > threshold_dE!" % self.outliers)
                    if self.outliers == 3:
                        Num_outliers = self.New_Num_outliers_percent(Y,management)  # (dft-ml)/std > 2
                        print ("outliers == %s is for the all data!, the definition of outlier is dE/threshold_std > 2!" % self.outliers)
                    if self.outliers == 4:
                        Num_outliers = self.New_Num_outliers_percent_trainingset(management)  # (dft-ml)/std > 2
                        print ("outliers == %s is for the training set, the definition of outlier is dE/threshold_std > 2!" % self.outliers)
                    if self.outliers == 5:
                        Num_outliers = self.New_Num_outliers_percent_testset(management)  # (dft-ml)/std > 2
                        print ("outliers == %s is for the test set, the definition of outlier is dE/threshold_std > 2!" % self.outliers)

                    self.save_variables(training_set_variables,training_set_observations)
                    #----------
                    if self.kk > 1 or self.use_all_std is True:
                        i = len(training_set_variables)
                        print ("Here the i is the total numer of the training data: %s " % i)
                    break

            else: 
                if MaxdE_testset > self.threshold_dE:
                    management, sigma = self.gpr(X,training_set_variables,training_set_observations,test_set_variables,
                                                 test_set_observations,run_std,training_set_AdsSlabSite,test_set_AdsSlabSite)
                    MaxdE_testset = management['MaxdE_testset']
                    MaxStd_testset = management['MaxStd_testset']
                    deltaE_index = management['deltaE_index']
                    training_set_AdsSlabSite = management['training_set_AdsSlabSite']
                    test_set_AdsSlabSite = management['test_set_AdsSlabSite']
                    SIGMA.append(sigma)

                    print ("This is %s_th iteration, MaxdE of test_set is ** %s eV **" % (i,round(MaxdE_testset,3)))
                    print ('')

                    elements = [test_set_variables[k] for k in deltaE_index]
                    for elem in elements:
                        indx = test_set_variables.index(elem)
                        
                        """
                        ads1 = test_set_AdsSlabSite[indx][0]
                        site1 = test_set_AdsSlabSite[indx][1]
                        ads2 = test_set_AdsSlabSite[indx][2]
                        site2 = test_set_AdsSlabSite[indx][3]                        
                        """
                        
                        training_set_variables.append(elem)
                        training_set_observations.append(test_set_observations[indx])
                        test_set_variables.remove(elem)
                        test_set_observations.remove(test_set_observations[indx])
                        
                        training_set_AdsSlabSite.append(test_set_AdsSlabSite[indx])
                        test_set_AdsSlabSite.remove(test_set_AdsSlabSite[indx])
                        
                        """
                        for list_i in test_set_AdsSlabSite:
                            if (ads2 == list_i[0]) and (site2 == list_i[1]) and (ads1 == list_i[2]) and (site1 == list_i[3]):    
                                ind_same_str = test_set_AdsSlabSite.index(list_i)
                                
                                training_set_AdsSlabSite.append(test_set_AdsSlabSite[ind_same_str])
                                random_testset_variables = test_set_AdsSlabSite[ind_same_str]
                                test_set_AdsSlabSite.remove(random_testset_variables)
                                
                                
                                training_set_variables.append(test_set_variables[ind_same_str])
                                training_set_observations.append(test_set_observations[ind_same_str])
                                
                                random_testset_variables = test_set_variables[ind_same_str]
                                random_testset_observations = test_set_observations[ind_same_str]
                                test_set_variables.remove(random_testset_variables)
                                test_set_observations.remove(random_testset_observations)   
                            else:
                                pass
                        """
                        
                else:
                    #NumOf_dft = len(training_set_variables)
                    print ("# # # # # # # # # # Iterations have convereged # # # # # # # # # #")
                    print ("## ## ## ## ## ## Summary ## ## ## ## ## ##")
                    print ("The number of samples = %s" % len(index_of_samples))
                    print ("The number of converged samples = %s" % len(training_set_observations))
                    print ("After %s times iteration, MaxdE of test_set is ** %s **" % ((i-1),round(MaxdE_testset,3)))
                    print ("After %s times iteration, MaxStd of test_set is ** %s eV **" % ((i-1),round(MaxStd_testset,3)))
                    
                    self.save_AdsSlabSite(col_var,management)
                    self.save_predicted_values(self.origin_data,management)
                    MAE_trainingset, MAE_testset, std_dE_trainset = self.plot_deltaE(management)
                    self.plot_DFT_ML(management)
                    self.plot_emerge(management)
                    self.find_outlier(df,management)
                    self.plot_algorithm_dE(SIGMA)
                    Num_outliers = []  
                    break
        
        return i, MaxdE_testset, MAE_testset, std_dE_trainset, Num_outliers
        
        
    def gpr(self,X,training_set_variables,training_set_observations,test_set_variables,test_set_observations,
            run_std,training_set_AdsSlabSite,test_set_AdsSlabSite):
        kernel_rbf = 1 * RBF(self.length_scale,length_scale_bounds=(1e-5,1e5))
        model = GaussianProcessRegressor(kernel=kernel_rbf,alpha=self.alpha,n_restarts_optimizer=self.n_restarts_optimizer,random_state=None)  
        model.fit(training_set_variables,training_set_observations)
        
        # just checking the training set
        predict_observations_trainingset, std_trainingset = model.predict(training_set_variables,return_std=True)
        deltaE_trainingset = [training_set_observations[i] - predict_observations_trainingset[i] for i in range(len(predict_observations_trainingset))]
        print ("maximum and miximum values for the E_DFT - E_ML on training set = (%s, %s)" % (round(max(deltaE_trainingset),3),round(min(deltaE_trainingset),3)))

        predict_observations_testset, std_testset = model.predict(test_set_variables,return_std=True)
        deltaE_testset = [test_set_observations[i] - predict_observations_testset[i] for i in range(len(predict_observations_testset))]
        print ("maximum and miximum values for the E_DFT - E_ML on test set = (%s, %s)" % (round(max(deltaE_testset),3),round(min(deltaE_testset),3)))
        print ("length of (trainingset, testset) = (%s, %s)" % (len(training_set_observations),len(test_set_observations)))
        
        Y_predicted, std = model.predict(X,return_std=True)
        sigma = std
        # consider error bar (Uncertainty)
        if run_std is True:
            abs_deltaE = [abs(i) for i in deltaE_testset]
            MaxdE_testset = max(abs_deltaE)
            # first method: large uncertainty
            MaxStd_testset = max(std_testset)
            std_index = []
            if MaxStd_testset > self.threshold_std:
                if self.use_all_std is False:
                    if self.random_state is True:
                        std = [std_i for std_i in list(set(std_testset)) if std_i > self.threshold_std]
                        if len(std) > 10:
                            k = self.kk
                            list_std = random.sample(population=std,k=k)
                        else:
                            k = 1
                            list_std = random.sample(population=std,k=k)
                    else:
                        std = sorted([std_i for std_i in list(set(std_testset)) if std_i > self.threshold_std])                
                        if len(std) > 10:
                            k = self.kk
                            list_std = std[-k:]
                        else:
                            k = 1
                            list_std = std[-k:]
                else:
                    std = [std_i for std_i in list(set(std_testset)) if std_i > self.threshold_std]
                    list_std = std
                    
                for std_i in list_std:
                    index_i = list(std_testset).index(std_i)
                    std_index.append(index_i)
                management = {'MaxStd_testset':MaxStd_testset, 'std_index':std_index, 'MaxdE_testset':MaxdE_testset,'test_set_observations':test_set_observations,
                              'deltaE_trainingset':deltaE_trainingset, 'deltaE_testset':deltaE_testset, 'training_set_observations':training_set_observations,
                              'predict_observations_trainingset':predict_observations_trainingset,'test_set_observations':test_set_observations,
                              'predict_observations_testset':predict_observations_testset,'std_trainingset':std_trainingset,'std_testset':std_testset,
                              'test_set_AdsSlabSite':test_set_AdsSlabSite,'training_set_AdsSlabSite':training_set_AdsSlabSite,'Y_predicted':Y_predicted,'std':std}

            else:
                management = {'MaxStd_testset':MaxStd_testset, 'std_index':std_index, 'MaxdE_testset':MaxdE_testset,'test_set_observations':test_set_observations,
                              'deltaE_trainingset':deltaE_trainingset, 'deltaE_testset':deltaE_testset, 'training_set_observations':training_set_observations,
                              'predict_observations_trainingset':predict_observations_trainingset,'test_set_observations':test_set_observations,
                              'predict_observations_testset':predict_observations_testset,'std_trainingset':std_trainingset,'std_testset':std_testset,
                              'test_set_AdsSlabSite':test_set_AdsSlabSite,'training_set_AdsSlabSite':training_set_AdsSlabSite,'Y_predicted':Y_predicted,'std':std}                
         
           
        # consider energy difference
        else:
            # second method: large energy difference between DFT and ML energy
            MaxStd_testset = max(std_testset)
            abs_deltaE = [abs(i) for i in deltaE_testset]
            MaxdE_testset = max(abs_deltaE)
            deltaE_index = []
            if MaxdE_testset > self.threshold_dE:
                if self.random_state is True:
                    deltaE = [i for i in list(set(deltaE_testset)) if abs(i) > self.threshold_dE]
                    if len(deltaE) > 10:
                        k = self.kk
                        list_dE = random.sample(population=deltaE,k=k)
                    else:
                        k = 1
                        list_dE = random.sample(population=deltaE,k=k)
                else:
                    deltaE = sorted([i for i in list(set(deltaE_testset)) if abs(i) > self.threshold_dE],key=abs)
                    if len(deltaE) > 10:
                        k = self.kk
                        list_dE = deltaE[-k:]
                    else:
                        k = 1
                        list_dE = deltaE[-k:]   
                for de in list_dE:
                    index_i = deltaE_testset.index(de)
                    deltaE_index.append(index_i)
            
                management = {'MaxStd_testset':MaxStd_testset,'MaxdE_testset':MaxdE_testset,'deltaE_index':deltaE_index,'deltaE_trainingset':deltaE_trainingset,
                              'deltaE_testset':deltaE_testset, 'training_set_observations':training_set_observations,
                              'predict_observations_trainingset':predict_observations_trainingset,'test_set_observations':test_set_observations,'test_set_variables':test_set_variables,
                              'predict_observations_testset':predict_observations_testset,'std_trainingset':std_trainingset,'std_testset':std_testset,
                              'test_set_AdsSlabSite':test_set_AdsSlabSite,'training_set_AdsSlabSite':training_set_AdsSlabSite,'Y_predicted':Y_predicted,'std':std}

            else:
                management = {'MaxStd_testset':MaxStd_testset,'MaxdE_testset':MaxdE_testset,'deltaE_index':deltaE_index,'deltaE_trainingset':deltaE_trainingset,
                              'deltaE_testset':deltaE_testset, 'training_set_observations':training_set_observations,
                              'predict_observations_trainingset':predict_observations_trainingset,'test_set_observations':test_set_observations,'test_set_variables':test_set_variables,
                              'predict_observations_testset':predict_observations_testset,'std_trainingset':std_trainingset,'std_testset':std_testset,
                              'test_set_AdsSlabSite':test_set_AdsSlabSite,'training_set_AdsSlabSite':training_set_AdsSlabSite,'Y_predicted':Y_predicted,'std':std}
                                
        return management, sigma
        
    def plot_deltaE(self,management):
        # Mean Absolute Error
        deltaE_trainingset = management['deltaE_trainingset']
        from scipy.stats import norm
        mu, std_dE_trainset = norm.fit(deltaE_trainingset)
        
        abs_deltaE = [abs(i) for i in deltaE_trainingset]
        MAE_trainingset = sum(abs_deltaE)/len(deltaE_trainingset)
        print ("The Mean Absolute Error of Training Set is = %s !" % round(MAE_trainingset,3))
        fig, ax = plt.subplots()
        ax.hist(deltaE_trainingset,color='b',edgecolor='black',bins=50,rwidth=5,density=True)
        ax.set_xlabel('E$_{DFT}$ - E$_{ML}$ (eV)',fontsize=16)
        ax.set_ylabel('Normalized frequency',fontsize=16)
        ax.tick_params(axis='both',direction='in',labelsize=14)
        ax.set_xlim(-1.5,1.5)
        ax.text(0.4,1.5,"MAE = %s eV" % round(MAE_trainingset,3),fontsize=14)
        fig.tight_layout()
        plt.savefig('dE_trainingset.pdf')
        plt.savefig('dE_trainingset.png')
        #plt.show()
        deltaE_testset = management['deltaE_testset']
        abs_deltaE = [abs(i) for i in deltaE_testset]
        MAE_testset = sum(abs_deltaE)/len(deltaE_testset)
        print ("The Mean Absolute Error of Test Set is = %s !" % round(MAE_testset,3))
        fig, ax = plt.subplots()
        ax.hist(deltaE_testset,color='r',edgecolor='black',bins=50,rwidth=5,density=True)
        ax.set_xlabel('E$_{DFT}$ - E$_{ML}$ (eV)',fontsize=16)
        ax.set_ylabel('Normalized frequency',fontsize=16)
        ax.tick_params(axis='both',direction='in',labelsize=14)
        ax.set_xlim(-1.5,1.5)
        ax.text(0.4,0.6,"MAE = %s eV" % round(MAE_testset,3),fontsize=14)
        fig.tight_layout()
        plt.savefig('dE_testset.pdf')
        plt.savefig('dE_testset.png')
        #plt.show()
        return MAE_trainingset, MAE_testset, std_dE_trainset
        
    def plot_DFT_ML(self,management):
        # DFT and ML energies
        training_set_observations = management['training_set_observations']
        predict_observations_trainingset = management['predict_observations_trainingset']
        test_set_observations = management['test_set_observations']
        predict_observations_testset = management['predict_observations_testset']
        std_trainingset = management['std_trainingset']
        std_testset = management['std_testset']
        Mean_Std_training = round(np.mean(std_trainingset),3)
        Mean_Std_test = round(np.mean(std_testset),3)
        print ("The Mean_Std for trainingset is = %s !" % Mean_Std_training)
        print ("The Mean_Std for testset is = %s !" % Mean_Std_test)
        fig, ax = plt.subplots()
        ax.errorbar(training_set_observations,predict_observations_trainingset,std_trainingset,fmt='bo')
        ax.errorbar(test_set_observations,predict_observations_testset,std_testset,fmt='ro')
        ax.legend(['train','test'],loc='best',fontsize=14)
        ax.set_xlabel('E$_{DFT}$ (eV)',fontsize=16)
        ax.set_ylabel('E$_{ML}$ (eV)',fontsize=16)
        t = np.linspace(-3,8.5,30)
        y = t
        ax.plot(t,y,'--',)
        #plt.xlim(-2.5,8.5)
        #plt.ylim(-2.5,8.5)
        ax.set_xlim(-2,8)
        ax.set_ylim(-2,8)
        ax.tick_params(axis='both',direction='in',labelsize=14)
        fig.tight_layout()
        plt.savefig('gpr.pdf')
        plt.savefig('gpr.png')
        #plt.show()
        Number_of_trainingset = len(training_set_observations)
        Number_of_testset = len(test_set_observations)
        return Number_of_trainingset, Number_of_testset
        
    def save_AdsSlabSite(self,col_var,management):
        training_set_observations = management['training_set_observations']
        test_set_observations = management['test_set_observations']
        training_set_AdsSlabSite = management['training_set_AdsSlabSite']
        test_set_AdsSlabSite = management['test_set_AdsSlabSite']
        
        #col_var = ['Ads','Slab','Site']
        AdsSlabSite_trainingset = pd.DataFrame(training_set_AdsSlabSite,columns=col_var)
        AdsSlabSite_trainingset.insert(loc=len(col_var),column='Ead',value=training_set_observations)
        AdsSlabSite_trainingset.to_csv('best_trainingset.csv',index=False)
        
        AdsSlabSite_testset = pd.DataFrame(test_set_AdsSlabSite,columns=col_var)
        AdsSlabSite_testset.insert(loc=len(col_var),column='Ead',value=test_set_observations)
        AdsSlabSite_testset.to_csv('best_testset.csv',index=False)
        Number_of_training_set_AdsSlabSite = len(training_set_AdsSlabSite) 
        Number_of_test_set_AdsSlabSite = len(test_set_AdsSlabSite)
        return Number_of_training_set_AdsSlabSite, Number_of_test_set_AdsSlabSite

    def save_variables(self,training_set_variables,training_set_observations):
        columns = len(training_set_variables[0])
        df = pd.DataFrame(training_set_variables,columns=['f_%s' % i for i in range(columns)])
        df.insert(loc=columns,column='Ead',value=training_set_observations)
        df.to_csv('best_trainingset_data.csv',index=False)
        
    def save_predicted_values(self,origin_data,management):
        E_ml = management['Y_predicted']
        Std = management['std']
        dff = pd.read_csv(origin_data)
        dff = dff.dropna(axis=0,how='any')
        columns = dff.columns
        dff.insert(loc=len(columns),column='energyML',value=E_ml)
        columns = dff.columns
        dff.insert(loc=len(columns),column='Std',value=Std)
        dff.to_csv('Analysis_data.csv',index=False)

    def find_outlier(self,df,management):
        Y_ml = management['Y_predicted']
        Y_dft = df[df.columns[-1]]
        delta_Y = [abs(Y_dft[i] - Y_ml[i]) for i in range(len(Y_dft))]
        Idx = []
        for value in delta_Y:
            if value >= self.threshold_dE:
                idx = delta_Y.index(value)
                Idx.append(idx)
                print ('Check structures:')   # # % df_origin.iloc[idx] will output structures
                print ("(index, E_dft, delta_E):", tuple([idx,Y_dft[idx],value]))  #  (tuple([idx,df.iloc[idx,:].tolist()]))
            else:
                pass
        print ("Length of Outliers:", len(Idx))
        return Idx        

    def plot_emerge(self,management):
        # DFT and ML energies
        training_set_observations = management['training_set_observations']
        predict_observations_trainingset = management['predict_observations_trainingset']
        test_set_observations = management['test_set_observations']
        predict_observations_testset = management['predict_observations_testset']
        std_trainingset = management['std_trainingset']
        std_testset = management['std_testset']
        Mean_Std_training = round(np.mean(std_trainingset),3)
        Mean_Std_test = round(np.mean(std_testset),3)
        print ("The Mean_Std for trainingset is = %s !" % Mean_Std_training)
        print ("The Mean_Std for testset is = %s !" % Mean_Std_test)
        
        deltaE_testset = management['deltaE_testset']
        abs_deltaE = [abs(i) for i in deltaE_testset]
        MAE_testset = sum(abs_deltaE)/len(deltaE_testset)
        print ("The Mean Absolute Error of Test Set is = %s !" % round(MAE_testset,3))
        
        fig = plt.figure(tight_layout=True,figsize=(6,8))
        left = 0.025
        top = 0.9
        ax1 = plt.subplot(211)
        ax1.errorbar(training_set_observations,predict_observations_trainingset,std_trainingset,fmt='bo')
        ax1.errorbar(test_set_observations,predict_observations_testset,std_testset,fmt='ro')
        ax1.legend(['training','test'],loc='lower right',fontsize=16)
        ax1.set_xlabel('E$_{DFT}$ (eV)',fontsize=20)
        ax1.set_ylabel('E$_{ML}$ (eV)',fontsize=20)
        t = np.linspace(-4,10,30)
        y = t
        ax1.plot(t,y,'--',)
        #ax1.set_xlim(-2.5,8.5)
        #ax1.set_ylim(-2.5,8.5)
        plt.xlim(-2.5,9)
        plt.ylim(-2.5,9)
        ax1.tick_params(axis='both',direction='in',labelsize=18)

        ax2 = plt.subplot(212)
        ax2.hist(deltaE_testset,color='r',edgecolor='black',bins=50,rwidth=5,density=True)
        ax2.set_xlabel('E$_{DFT}$ - E$_{ML}$ (eV)',fontsize=20)
        ax2.set_ylabel('Normalized frequency',fontsize=20)
        ax2.tick_params(axis='both',direction='in',labelsize=18)
        ax2.set_xlim(-1.5,1.5)
        ax2.text(0.25,1.25,"MAE = %s eV" % round(MAE_testset,3),fontsize=16)
        ax1.text(left,top,'(a)',fontsize=16,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes)
        ax2.text(left,top,'(b)',fontsize=16,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes)
        
        # Plot the PDF.
        from scipy.stats import norm
        mu, std = norm.fit(deltaE_testset)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax2.plot(x, p, 'k', linewidth=1.5)
        ax2.annotate('$\sigma$ = %s' % round(std,2),xy=(-0.4,norm.pdf(-0.4,mu,std)),xytext=(-1.25,0.75),arrowprops=dict(facecolor='black',width=0.4,headwidth=6),fontsize=16)  # 6
        
        plt.savefig('Figure.pdf')
        plt.savefig('Figure.png')
        
        
        #***********************---------------------------------------
        fig, ax = plt.subplots(1,2,figsize=(12,6))

        ax[0].errorbar(training_set_observations,predict_observations_trainingset,std_trainingset,fmt='bo')
        ax[0].errorbar(test_set_observations,predict_observations_testset,std_testset,fmt='ro')
        ax[0].legend(['training','test'],loc='lower right',fontsize=16)
        ax[0].set_xlabel('E$_{DFT}$ (eV)\n(a)',fontsize=20)
        ax[0].set_ylabel('E$_{ML}$ (eV)',fontsize=20)
        t = np.linspace(-4,10,30)
        y = t
        ax[0].plot(t,y,'--',)
        ax[0].set_xlim(-2.5,9.3)
        ax[0].set_ylim(-2.5,9.3)
        ax[0].tick_params(axis='both',direction='in',labelsize=18)

        ax[1].hist(deltaE_testset,color='r',edgecolor='black',bins=50,rwidth=5,density=True)
        ax[1].set_xlabel('E$_{DFT}$ - E$_{ML}$ (eV)\n(b)',fontsize=20)
        ax[1].set_ylabel('Normalized frequency',fontsize=20)
        ax[1].tick_params(axis='both',direction='in',labelsize=18)
        ax[1].set_xlim(-1.5,1.5)
        ax[1].text(0.25,1.25,"MAE = %s eV" % round(MAE_testset,3),fontsize=16)
        #ax[1].text(left,top,'(a)',fontsize=16,horizontalalignment='left',verticalalignment='bottom',transform=ax1.transAxes)
        #ax[1].text(left,top,'(b)',fontsize=16,horizontalalignment='left',verticalalignment='bottom',transform=ax2.transAxes)
        
        # Plot the PDF.
        from scipy.stats import norm
        mu, std = norm.fit(deltaE_testset)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax[1].plot(x, p, 'k', linewidth=1.5)
        ax[1].annotate('$\sigma$ = %s' % round(std,2),xy=(-0.4,norm.pdf(-0.4,mu,std)),xytext=(-1.25,0.75),arrowprops=dict(facecolor='black',width=0.4,headwidth=6),fontsize=16)  # 6
        fig.tight_layout()
        plt.savefig('ML_DFT.pdf')
        plt.savefig('ML_DFT.png')

    def plot_algorithm(self,SIGMA):
        N = len(SIGMA)
        fig, ax = plt.subplots()
        ax.tick_params(axis='both',direction='in',labelsize=14)
        for n in range(N):
            y = SIGMA[n]
            x = len(y) * [n]
            Max_y = max(y)
            ax.scatter(x, y, color='g')  # s
            ax.scatter(n, Max_y, color='r') # one point
        ax.set_xlabel('Iteration #',fontsize=16)
        ax.set_ylabel('Predicted error (eV)', fontsize=16)
        
        xx = np.linspace(-0.2,N-1+0.2,50)
        yy = 50 * [self.threshold_std]
        ax.plot(xx,yy,linestyle='--',color='black',linewidth=2)
        ax.legend(['MPE = %s' % round(self.threshold_std,3)], loc='best',fontsize=14)
        fig.tight_layout()
        plt.savefig('ML_algorithm_1.pdf')
        plt.savefig('ML_algorithm_1.png')
        
    def Num_outlier(self,Y, management):
        training_set_observations = management['training_set_observations']
        predict_observations_trainingset = management['predict_observations_trainingset']
        delE_trainingset = [abs(training_set_observations[k] - predict_observations_trainingset[k]) for k in range(len(predict_observations_trainingset))]
        test_set_observations = management['test_set_observations']
        predict_observations_testset = management['predict_observations_testset']
        delE_testset = [abs(test_set_observations[k] - predict_observations_testset[k]) for k in range(len(predict_observations_testset))]

        #std_trainingset = management['std_trainingset']
        #std_testset = management['std_testset']
        
        fig, ax = plt.subplots()
        ax.tick_params(axis='both',direction='in',labelsize=14)
        c = []
        for e in delE_trainingset:
            if e > self.threshold_dE:
                c.append('r')
            else:
                c.append('g')
        
        ax.scatter(training_set_observations,delE_trainingset,marker='o',s=10,color=c)
        
        c = []
        for e in delE_testset:
            if e > self.threshold_dE:
                c.append('r')
            else:
                c.append('g')        
        ax.scatter(test_set_observations,delE_testset,marker='*',color=c)
        ax.legend(['training','test'],loc='best',fontsize=14)
        left, right = ax.get_xlim()
        xx = np.linspace(left,right,50)
        yy = 50 * [self.threshold_dE]
        ax.plot(xx,yy,linestyle='--',color='black',linewidth=1.5)
        
        ax.set_xlabel('DFT energy (eV)',fontsize=16)
        ax.set_ylabel('$\Delta$E (eV)', fontsize=16)
        fig.tight_layout()
        plt.savefig('show_outlier.pdf')
        plt.savefig('show_outlier.png')
        
        y_ml = management['Y_predicted']
        delE = [abs(Y[i] - y_ml[i]) for i in range(len(y_ml))]
        outliers = [e for e in delE if e > self.threshold_dE ]    ### e > threshold_dE
        Num_outliers = len(outliers)/len(Y) * 100 # unit is %
        return Num_outliers        

    def Num_outlier_percent(self,Y, management):
        training_set_observations = management['training_set_observations']
        predict_observations_trainingset = management['predict_observations_trainingset']
        delE_trainingset = [abs(training_set_observations[k] - predict_observations_trainingset[k]) for k in range(len(predict_observations_trainingset))]
        test_set_observations = management['test_set_observations']
        predict_observations_testset = management['predict_observations_testset']
        delE_testset = [abs(test_set_observations[k] - predict_observations_testset[k]) for k in range(len(predict_observations_testset))]

        std_trainingset = management['std_trainingset']
        std_testset = management['std_testset']
        
        fig, ax = plt.subplots()
        ax.tick_params(axis='both',direction='in',labelsize=14)
        c = []
        for k, e in enumerate(delE_trainingset):
            if e > std_trainingset[k]:
                c.append('r')
            else:
                c.append('g')
        
        ax.scatter(std_trainingset,delE_trainingset,marker='o',s=10,color=c)
        
        c = []
        for k, e in enumerate(delE_testset):
            if e > std_testset[k]:
                c.append('r')
            else:
                c.append('g')        
        ax.scatter(std_testset,delE_testset,marker='*',color=c)
        ax.legend(['training','test'],loc='best',fontsize=14)

        left, right = ax.get_ylim()
        xx = np.linspace(left,right,50)
        yy = xx
        ax.plot(xx,yy,linestyle='--',color='black',linewidth=1.5)
        
        ax.set_xlabel('Predicted Std (eV)',fontsize=16)
        ax.set_ylabel('$\Delta$E (eV)', fontsize=16)
        fig.tight_layout()
        plt.savefig('show_outlier_2.pdf')
        plt.savefig('show_outlier_2.png')
        
        y_ml = management['Y_predicted']
        delE = [abs(Y[i] - y_ml[i]) for i in range(len(y_ml))]
        new_outliers = [e for e in delE if e > self.threshold_dE]   #here the outlier is abs(delta_E) > threshold_dE.
        
        Num_outliers_percent = len(new_outliers) / float(len(Y)) * 100 # unit is %
        print ("-------------------------------------")
        print ("Num_outliers_percent [e > threshold_dE] : %s" % Num_outliers_percent)
        return Num_outliers_percent
        
    def New_Num_outliers_percent(self,Y,management):
        E_ml = management['Y_predicted']
        #Std = management['std']
        outliers = []
        delE = [abs(Y[i] - E_ml[i])/self.threshold_std for i in range(len(E_ml))]
        for de in delE:
            if de > self.ntimes:
                outliers.append(de)
            
        Num_outliers_percent = len(outliers) / float(len(Y)) * 100 # unit is %
        print ("-------------------------------------")
        print ("Num_outliers_percent (all data) [|E_dft - E_ml|/threshold_std > ntimes]: %s" % Num_outliers_percent)
        return Num_outliers_percent


    def New_Num_outliers_percent_trainingset(self,management):
        training_set_observations = management['training_set_observations']
        predict_observations_trainingset = management['predict_observations_trainingset']
        training_set_AdsSlabSite = management['training_set_AdsSlabSite']
        #std_trainingset = management['std_trainingset']
        outliers = []
        str_outliers = []
        
        delE = [abs(training_set_observations[i] - predict_observations_trainingset[i])/self.threshold_std for i in range(len(training_set_observations))]
        for k, de in enumerate(delE):
            if de > self.ntimes:
                outliers.append(de)
                str_outliers.append(training_set_AdsSlabSite[k])
                
        columns = [str(k) for k in range(1, 1+len(str_outliers[0]))]
        df = pd.DataFrame(str_outliers,columns=columns)
        df.to_csv('Check_test_set_structures.csv',index=False)
            
        Num_outliers_percent = len(outliers) / float(len(training_set_observations)) * 100 # unit is %
        print ("-------------------------------------")
        print ("Num_outliers_percent [|E_dft - E_ml|/threshold_std > ntimes]: %s" % Num_outliers_percent)
        return Num_outliers_percent

    def New_Num_outliers_percent_testset(self,management):
        test_set_observations = management['test_set_observations']
        predict_observations_testset = management['predict_observations_testset']
        test_set_AdsSlabSite = management['test_set_AdsSlabSite']
        #std_testset = management['std_testset']
        outliers = []
        
        str_outliers = []
        #--------
        dft_good, dft_bad, ml_good, ml_bad = [], [], [], []
        #--------
        delE = [abs(test_set_observations[i] - predict_observations_testset[i])/self.threshold_std for i in range(len(test_set_observations))]
        for k, de in enumerate(delE):
            if de > self.ntimes:
                outliers.append(de)
                str_outliers.append(test_set_AdsSlabSite[k])
                
                dft_bad.append(test_set_observations[k])
                ml_bad.append(predict_observations_testset[k])
            else:
                dft_good.append(test_set_observations[k])
                ml_good.append(predict_observations_testset[k])
        
        columns = [str(k) for k in range(1, 1+len(str_outliers[0]))]
        df = pd.DataFrame(str_outliers,columns=columns)
        df.to_csv('Check_test_set_structures.csv',index=False)
            
        Num_outliers_percent = len(outliers) / float(len(test_set_observations)) * 100 # unit is %
        print ("-------------------------------------")
        print ("Num_outliers_percent [|E_dft - E_ml|/threshold_std > ntimes]: %s" % Num_outliers_percent)
        
        iteration.plot_outliers_testset(self,dft_good, ml_good, dft_bad, ml_bad,)
        return Num_outliers_percent

    def plot_outliers_testset(self,dft_good, ml_good, dft_bad, ml_bad):
        fig, ax = plt.subplots(figsize=(6.5,6))
        ax.scatter(dft_good,ml_good,marker='o',color='r',s=10)
        ax.scatter(dft_bad,ml_bad,marker='x',color='r',s=20)
        ax.set_xlabel('E$_{DFT}$ (eV)\n',fontsize=20)
        ax.set_ylabel('E$_{ML}$ (eV)',fontsize=20)
        t = np.linspace(-10,15,30)
        y = t
        ax.plot(t,y,'--',)
        
        y_down = t - self.ntimes * self.threshold_std
        y_up = t + self.ntimes * self.threshold_std
        ax.plot(t,y_down,'--',color='orange')
        ax.plot(t,y_up,'--',color='orange')
        
        ax.set_xlim(-2,8)
        ax.set_ylim(-2,8)
        ax.tick_params(axis='both',direction='in',labelsize=18)
        fig.tight_layout()
        plt.savefig('plot_outliers.png')
        plt.savefig('plot_outliers.pdf')


    def bad_data_index(self,MaxPredStd):
        bad_data = []
        for k in range(1, len(MaxPredStd)):
            mps_k = MaxPredStd[k]
            mps_k_1 = MaxPredStd[k-1]
            if mps_k > mps_k_1:
                bad_data.append(k-1)
        return bad_data
        
    def plot_algorithm_dE(self,SIGMA):
        N = len(SIGMA)
        fig, ax = plt.subplots()
        ax.tick_params(axis='both',direction='in',labelsize=14)
        for n in range(N):
            y = SIGMA[n]
            x = len(y) * [n]
            Max_y = max(y)
            ax.scatter(x, y, color='g')  # s
            ax.scatter(n, Max_y, color='r') # one point
        ax.set_xlabel('Iteration #',fontsize=16)
        ax.set_ylabel('$\Delta E$ (eV)', fontsize=16)
        
        xx = np.linspace(-0.2,N-1+0.2,50)
        yy = 50 * [self.threshold_dE]
        ax.plot(xx,yy,linestyle='--',color='black',linewidth=2)
        ax.legend(['$Max_{_}\DetalE$ = %s' % self.threshold_dE], loc='best',fontsize=14)
        fig.tight_layout()
        plt.savefig('ML_algorithm_1.pdf')
        plt.savefig('ML_algorithm_1.png')    
