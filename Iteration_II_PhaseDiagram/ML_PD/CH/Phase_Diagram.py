# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:09:42 2019

@author: fuzhu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import random
import pickle


"""
#usage:
import pandas as pd
from Phase_Diagram import Run_phase_diagram

data_ASS = 'for_formation_AdsSlabSite_corr.csv'
data_ml = 'New_features_Ead.csv' # 

# pre-processing
df = pd.read_csv(data_ASS)
dff = df.dropna(axis=0,how='any')
columns = df.columns.tolist()
Ele = df.values.tolist()
index = []
for ele in Ele:
    site2 = ele[3]
    if site2 == 'top_other':
        ind = Ele.index(ele)
        index.append(ind)
#print (index)       
df.drop(index=index)
df.to_csv('A.csv',index=False)
df = pd.read_csv(data_ml)
dff = df.dropna(axis=0,how='any')
dff.drop(index=index)
dff.to_csv('B.csv',index=False)
data_ASS = 'A.csv'
data_ml = 'B.csv' # 
atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2],'O':[-2,2]}  # 4d
#atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2]}  # 3d
#atomType_chemPot = {'H':[-2,2],'C':[0,4]}  # 2d
N_int = 4
rPD = Run_phase_diagram(data_ml=data_ml,data_ASS=data_ASS,atomType_chemPot=atomType_chemPot,N_int=N_int,initial_training_number=43,kk=1)
if len(atomType_chemPot) == 2:
    rPD.run_2d()
if len(atomType_chemPot) == 3:
    rPD.run_3d()
if len(atomType_chemPot) == 4:
    rPD.run_4d()

"""

class Run_phase_diagram:
    
    def __init__(self,data_ml,data_ASS,atomType_chemPot,N_int,ref_atomType_chemPot=None,n=2,factor=0,max_cycle=300,
                 alpha=0.1,n_restarts_optimizer=10,initial_training_number=43,random_state=False,kk=5):
        """
        atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2]} # 3d
        ref_atomType_chemPot = {'O':0}  # 3d
        
        atomType_chemPot = {'H':[-2,2],'C':[0,4]} # 2d
        ref_atomType_chemPot = {'O':0,'N':0}  # 2d
        
        if ref_atomType_chemPot = None, the ref is to 0
        """
        self.data_ml = data_ml
        self.data_ASS = data_ASS
        self.atomType_chemPot = atomType_chemPot
        self.ref_atomType_chemPot = ref_atomType_chemPot
        self.N_int = N_int
        self.n = n
        self.factor = factor
        self.max_cycle = max_cycle
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.initial_training_number = initial_training_number
        self.random_state = random_state
        self.kk = kk

    def str_name(self):
        return ''.join(self.atomType_chemPot.keys())
        
    def mu_X(self):
        keys = list(self.atomType_chemPot.keys())
        values = list(self.atomType_chemPot.values())
        length = len(keys)
        N_int = self.N_int
        
        if length == 1:
            atomA = keys[0]
            muA = np.linspace(values[0][0],values[0][1],N_int)
            return atomA, muA       
        
        if length == 2:
            atomA = keys[0]
            muA = np.linspace(values[0][0],values[0][1],N_int)
            atomB = keys[1]
            muB = np.linspace(values[1][0],values[1][1],N_int)
            return atomA, muA, atomB, muB
        
        if length == 3:
            atomA = keys[0]
            muA = np.linspace(values[0][0],values[0][1],N_int)
            atomB = keys[1]
            muB = np.linspace(values[1][0],values[1][1],N_int)
            atomC = keys[2]
            muC = np.linspace(values[2][0],values[2][1],N_int)
            return atomA, muA, atomB, muB, atomC, muC
        
        if length == 4:
            atomA = keys[0]
            muA = np.linspace(values[0][0],values[0][1],N_int)
            atomB = keys[1]
            muB = np.linspace(values[1][0],values[1][1],N_int)
            atomC = keys[2]
            muC = np.linspace(values[2][0],values[2][1],N_int)
            atomD = keys[1]
            muD = np.linspace(values[3][0],values[3][1],N_int)
            return atomA, muA, atomB, muB, atomC, muC, atomD, muD

    def ref_mu_X(self):
        if self.ref_atomType_chemPot is None:
            pass
        else:
            keys = list(self.ref_atomType_chemPot.keys())
            values = list(self.ref_atomType_chemPot.values())
            
            length = len(self.atomType_chemPot)

            if length == 1:
                atomB = keys[0]
                stateB = values[0]
                atomC = keys[1]
                stateC = values[1]
                atomD = keys[2]
                stateD = values[2]
                return atomB, stateB, atomC, stateC, atomD, stateD
            
            if length == 2:
                atomC = keys[0]
                stateC = values[0]
                atomD = keys[1]
                stateD = values[1]
                return atomC, stateC, atomD, stateD
                
            if length == 3:
                atomD = keys[0]
                stateD = values[0]
                return atomD, stateD

    def run_1d(self):
        
        atomA, muA = self.mu_X()
        chemical_potential_A = atomA
        
        if self.ref_atomType_chemPot is None:
            chemical_potential_B = None
            delta_mu_atomB = None
            chemical_potential_C = None
            delta_mu_atomC = None
            chemical_potential_D = None
            delta_mu_atomD = None        
        else:
            atomB, stateB, atomC, stateC, atomD, stateD = self.ref_mu_X()
            chemical_potential_B = atomB
            delta_mu_atomB = stateB
            chemical_potential_C = atomC
            delta_mu_atomC = stateC
            chemical_potential_D = atomD
            delta_mu_atomD = stateD
        
        Num_high_cover = []
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures 
        All_Str = []   # str that has been added to the initial training set
        for stateA in muA:
            delta_mu_atomA = stateA
            i, minE, stableStr, Add_str = self.launch_iteration(chemical_potential_A,delta_mu_atomA,
                                                                chemical_potential_B,delta_mu_atomB,
                                                                chemical_potential_C,delta_mu_atomC,
                                                                chemical_potential_D,delta_mu_atomD)
            Num_high_cover.append(i)
            Matrix_energy.append(minE)
            Matrix_structure.append(stableStr)
            All_Str.append(Add_str)

        new_matrix = []
        for mat in Matrix_structure:
            del mat[-2:]
            new_elem = '-'.join(mat)
            new_matrix.append(new_elem)
                
        name = self.str_name()
        #print (new_matrix)
        with open('matrix_structures_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Matrix_energy, data)
        
        with open('num_high_coverage_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Num_high_cover, data)

        with open('All_Str_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(All_Str, data)

    def run_2d(self):
        atomA, muA, atomB, muB = self.mu_X()
        chemical_potential_A = atomA
        chemical_potential_B = atomB
        
        if self.ref_atomType_chemPot is None:
            chemical_potential_C = None
            delta_mu_atomC = None
            chemical_potential_D = None
            delta_mu_atomD = None        
        else:
            atomC, stateC, atomD, stateD = self.ref_mu_X()
            chemical_potential_C = atomC
            delta_mu_atomC = stateC
            chemical_potential_D = atomD
            delta_mu_atomD = stateD
        
        Num_high_cover = []
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures
        All_Str = []   # str that has been added to the initial training set
        for i in range(self.N_int):
            Matrix_energy.append([])
            Matrix_structure.append([])
            Num_high_cover.append([])
            All_Str.append([])
        for k, stateA in enumerate(muA):
            delta_mu_atomA = stateA
            for stateB in muB:
                delta_mu_atomB = stateB
                i, minE, stableStr, Add_str = self.launch_iteration(chemical_potential_A,delta_mu_atomA,
                                                                    chemical_potential_B,delta_mu_atomB,
                                                                    chemical_potential_C,delta_mu_atomC,
                                                                    chemical_potential_D,delta_mu_atomD)
                Matrix_energy[k].append(minE)
                Matrix_structure[k].append(stableStr)
                Num_high_cover[k].append(i)
                All_Str[k].append(Add_str)


        new_matrix = []
        for i in range(self.N_int):
            new_matrix.append([])
            
        for k, mat in enumerate(Matrix_structure):
            for elem in mat:
                del elem[-2:]
                new_elem = '-'.join(elem)
                new_matrix[k].append(new_elem)
                
        name = self.str_name()
        #print (new_matrix)
        with open('matrix_structures_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Matrix_energy, data)
        
        with open('num_high_coverage_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Num_high_cover, data)
        
        with open('All_Str_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(All_Str, data)
            
    def run_3d(self):
        atomA, muA, atomB, muB, atomC, muC = self.mu_X()
        chemical_potential_A = atomA
        chemical_potential_B = atomB
        chemical_potential_C = atomC
        
        if self.ref_atomType_chemPot is None:
            chemical_potential_D = None
            delta_mu_atomD = None
        else:
            atomD, stateD = self.ref_mu_X()
            chemical_potential_D = atomD
            delta_mu_atomD = stateD
        
        Num_high_cover = []
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures
        for i in range(self.N_int):
            Matrix_energy.append([])
            Matrix_structure.append([])
            Num_high_cover.append([])
            for j in range(self.N_int):
                Matrix_energy[i].append([])
                Matrix_structure[i].append([])
                Num_high_cover[i].append([])
                
        for k, stateA in enumerate(muA):
            delta_mu_atomA=stateA
            for z, stateB in enumerate(muB):
                delta_mu_atomB=stateB
                for stateC in muC:
                    delta_mu_atomC=stateC
                    i, minE, stableStr, Add_str = self.launch_iteration(chemical_potential_A,delta_mu_atomA,
                                                                        chemical_potential_B,delta_mu_atomB,
                                                                        chemical_potential_C,delta_mu_atomC,
                                                                        chemical_potential_D,delta_mu_atomD)
                    Matrix_energy[k][z].append(minE)
                    Matrix_structure[k][z].append(stableStr)
                    Num_high_cover[k][z].append(i)


        new_matrix = []
        for i in range(self.N_int):
            new_matrix.append([])
            for j in range(self.N_int):
                new_matrix[i].append([])
        
        for k, mat in enumerate(Matrix_structure):
            for z, elems in enumerate(mat):
                for elem in elems:
                    del elem[-2:]
                    new_elem = '-'.join(elem)
                    new_matrix[k][z].append(new_elem)
        
        name = self.str_name()            
        #print (new_matrix)
        with open('matrix_structures_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Matrix_energy, data)
        
        with open('num_high_coverage_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Num_high_cover, data)  

    def run_4d(self):
        atomA, muA, atomB, muB, atomC, muC, atomD, muD = self.mu_X()
        chemical_potential_A = atomA
        chemical_potential_B = atomB
        chemical_potential_C = atomC
        chemical_potential_D = atomD
        
        Num_high_cover = []
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures
        for k in range(self.N_int):
            Matrix_energy.append([])
            Matrix_structure.append([])
            Num_high_cover.append([])
            for z in range(self.N_int):
                Matrix_energy[k].append([])
                Matrix_structure[k].append([])
                Num_high_cover[k].append([])
                for w in range(self.N_int):
                    Matrix_energy[k][z].append([])
                    Matrix_structure[k][z].append([])
                    Num_high_cover[k][z].append([])
                
        for k, stateA in enumerate(muA):
            delta_mu_atomA=stateA
            for z, stateB in enumerate(muB):
                delta_mu_atomB=stateB
                for w, stateC in enumerate(muC):
                    delta_mu_atomC=stateC
                    for stateD in muD:
                        delta_mu_atomD = stateD
                        i, minE, stableStr, Add_str = self.launch_iteration(chemical_potential_A,delta_mu_atomA,
                                                                            chemical_potential_B,delta_mu_atomB,
                                                                            chemical_potential_C,delta_mu_atomC,
                                                                            chemical_potential_D,delta_mu_atomD)
                        Matrix_energy[k][z][w].append(minE)
                        Matrix_structure[k][z][w].append(stableStr)
                        Num_high_cover[k][z][w].append(i)


        new_matrix = []
        for k in range(self.N_int):
            new_matrix.append([])
            for z in range(self.N_int):
                new_matrix[k].append([])
                for w in range(self.N_int):
                    new_matrix[k][z].append([])

        
        for k, A in enumerate(Matrix_structure):
            for z, B in enumerate(A):
                for w, C in enumerate(B):
                    for D in C:
                        del D[-2:]
                        new_ele = '-'.join(D)
                        new_matrix[k][z][w].append(new_ele)
        
        name = self.str_name()            
        #print (new_matrix)
        with open('matrix_structures_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Matrix_energy, data)
        
        with open('num_high_coverage_%s_%s_%s.pickle' % (name,self.N_int,self.n),'wb') as data:
            pickle.dump(Num_high_cover, data)

    
    """
    [Ead_ml(A@B) + Ead_df(B) - n * sigma(A@B) - n_i(A,B)*deltaMu] <= min([Ead_dft(A@B) + Ead_dft(B) - n_i(A,B)*deltaMu])
    
    The passing data is:
    data_ml = 'ml_data_f.csv'
    data_ASS = 'for_formation_AdsSlabSite.csv'
    
    """
    def launch_iteration(self,chemical_potential_A, delta_mu_atomA,
                         chemical_potential_B, delta_mu_atomB,
                         chemical_potential_C, delta_mu_atomC,
                         chemical_potential_D, delta_mu_atomD):
        #-----------------------------------------------------------------------------------------
        #initialize the analysized data
        # read the columns names of ads_slab_site
        data_ml = self.data_ml
        data_ASS= self.data_ASS
        initial_training_number = self.initial_training_number

        df_origin = pd.read_csv(data_ASS)
        df_origin = df_origin.dropna(axis=0,how='any')
        #print (df_origin.shape)
        col_var = df_origin.columns.get_values().tolist()
        #col_var.remove(col_var[-1])
        # in order to extract the index of ads_slab_site, this is cosistent with ml_data.csv data
        training_set_AdsSlabSite = df_origin.iloc[:initial_training_number].values.tolist()
        test_set_AdsSlabSite = df_origin.iloc[initial_training_number:].values.tolist()
        # read the ML data: ml_data.csv
        df = pd.read_csv(data_ml)
        #X = df.values[:,:-1]
        index_of_samples = len(df)
        training_set_variables = df.iloc[:initial_training_number,:-1].values.tolist()
        training_set_observations = df.iloc[:initial_training_number,-1].tolist()
        test_set_variables = df.iloc[initial_training_number:,:-1].values.tolist()
        test_set_observations = df.iloc[initial_training_number:,-1].tolist()
        #-----------------------------------------------------------------------------------------
        #launch the first time ML prediction
        management, min_dft_trainingset, min_E_ml_nstd = self.initial_grp(training_set_variables,training_set_observations,
                                                                          test_set_variables,test_set_observations,
                                                                          training_set_AdsSlabSite,test_set_AdsSlabSite)
        min_dft = min_dft_trainingset
        min_ml = min_E_ml_nstd
        Add_str = [] #############  stats candidates' structure
        #*****************************************************************************************
        for i in range(self.max_cycle):
            if min_ml <= min_dft:
                training_set_variables = management['training_set_variables']
                test_set_variables = management['test_set_variables']
                training_set_observations = management['training_set_observations']
                test_set_observations = management['test_set_observations']
                training_set_AdsSlabSite = management['training_set_AdsSlabSite']
                test_set_AdsSlabSite = management['test_set_AdsSlabSite']
                
                self.save_AdsSlabSite(col_var,management)
                
                var, sizeOf_dft = self.calc_formation_energy(chemical_potential_A,delta_mu_atomA,
                                                             chemical_potential_B,delta_mu_atomB,
                                                             chemical_potential_C,delta_mu_atomC,
                                                             chemical_potential_D,delta_mu_atomD,
                                                             index_of_samples)
                
                if len(var) > 3:
                    new_testset_index = var[-1]
                    min_dft = var[1]
                    min_ml = var[2]
                    #print (new_testset_index)
                    elements = [test_set_variables[k] for k in new_testset_index]
                    for elem in elements:
                        indx = test_set_variables.index(elem)
                        
                        Add_str.append('_'.join(test_set_AdsSlabSite[indx][0:4]))
                        
                        ads1 = test_set_AdsSlabSite[indx][0]
                        site1 = test_set_AdsSlabSite[indx][1]
                        ads2 = test_set_AdsSlabSite[indx][2]
                        site2 = test_set_AdsSlabSite[indx][3]                        
                        
                        training_set_variables.append(elem)
                        training_set_observations.append(test_set_observations[indx])
                        test_set_variables.remove(elem)
                        test_set_observations.remove(test_set_observations[indx])
                        
                        training_set_AdsSlabSite.append(test_set_AdsSlabSite[indx])
                        test_set_AdsSlabSite.remove(test_set_AdsSlabSite[indx])
                        
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
                    management = self.grp(training_set_variables,training_set_observations,
                                          test_set_variables,test_set_observations,
                                          training_set_AdsSlabSite,test_set_AdsSlabSite)
                    
                else:
                    Index = var[0]
                    min_E = var[1]
                    #min_based_iteration.plot_deltaE(self,management)
                    #min_based_iteration.plot_DFT_ML(self,management)
                    print ('##################################')
                    stableStr = self.output_structures(management,Index)
                    break
            
            #else:
            #    min_based_iteration.plot_deltaE(self,management)
            #    min_based_iteration.plot_DFT_ML(self,management)
            #    break
        return i, min_E, stableStr, Add_str
    

    def initial_grp(self,training_set_variables,training_set_observations,
                    test_set_variables,test_set_observations,
                    training_set_AdsSlabSite,test_set_AdsSlabSite):
        kernel_rbf = 1 * RBF(length_scale=1,length_scale_bounds=(0.01,500))
        model = GaussianProcessRegressor(kernel=kernel_rbf,alpha=self.alpha,n_restarts_optimizer = self.n_restarts_optimizer)  #   alpha = 0.2
        model.fit(training_set_variables,training_set_observations)
        
        predict_observations_trainingset, std_trainingset = model.predict(training_set_variables,return_std=True)
        predict_observations_testset, std_testset = model.predict(test_set_variables,return_std=True)
        
        management = {'training_set_AdsSlabSite':training_set_AdsSlabSite,
                      'test_set_AdsSlabSite':test_set_AdsSlabSite,
                      'training_set_variables':training_set_variables,
                      'test_set_variables':test_set_variables,
                      'training_set_observations':training_set_observations,
                      'test_set_observations':test_set_observations,
                      'predict_observations_trainingset':predict_observations_trainingset,
                      'predict_observations_testset':predict_observations_testset,
                      'std_trainingset':std_trainingset,
                      'std_testset':std_testset}
        
        # for algorithm E_ml - n*std + factor <= min(E_dft)
        min_dft_trainingset = min(training_set_observations)
        E_ml_testset_nstd = [predict_observations_testset[i] - self.n * std_testset[i] + self.factor for i in range(len(predict_observations_testset))]
        min_E_ml_nstd = min(E_ml_testset_nstd)
        if min_dft_trainingset < min_E_ml_nstd:
            print ('you can find most stable states in the inital traing sets!')
        return management, min_dft_trainingset,min_E_ml_nstd
        
    def grp(self,training_set_variables,training_set_observations,
            test_set_variables,test_set_observations,
            training_set_AdsSlabSite,test_set_AdsSlabSite):
        kernel_rbf = 1 * RBF(length_scale=1,length_scale_bounds=(0.01,500))
        model = GaussianProcessRegressor(kernel=kernel_rbf,alpha=self.alpha,n_restarts_optimizer = self.n_restarts_optimizer)  #   alpha = 0.2
        model.fit(training_set_variables,training_set_observations)
        
        predict_observations_trainingset, std_trainingset = model.predict(training_set_variables,return_std=True)
        predict_observations_testset, std_testset = model.predict(test_set_variables,return_std=True)
        
        deltaE_trainingset = [training_set_observations[i] - predict_observations_trainingset[i] for i in range(len(predict_observations_trainingset))]
        deltaE_testset = [test_set_observations[i] - predict_observations_testset[i] for i in range(len(predict_observations_testset))]  
        
        management = {'training_set_AdsSlabSite':training_set_AdsSlabSite,
                      'test_set_AdsSlabSite':test_set_AdsSlabSite,
                      'training_set_observations':training_set_observations,
                      'test_set_observations':test_set_observations,
                      'training_set_variables':training_set_variables,
                      'test_set_variables':test_set_variables,
                      'predict_observations_trainingset':predict_observations_trainingset,
                      'predict_observations_testset':predict_observations_testset,
                      'std_trainingset':std_trainingset,
                      'std_testset':std_testset,
                      'deltaE_trainingset':deltaE_trainingset,
                      'deltaE_testset':deltaE_testset}
        return management

    def output_structures(self,management,Index):
        training_set_AdsSlabSite = management['training_set_AdsSlabSite']
        stableStr = training_set_AdsSlabSite[Index]
        print ('Final stable structure:',stableStr)
        return stableStr
        
                
    def save_AdsSlabSite(self,col_var,management):
        # This func generates the data that calculating the formation energy at some state
        training_set_observations = management['training_set_observations']
        test_set_observations = management['test_set_observations']
        training_set_AdsSlabSite = management['training_set_AdsSlabSite']
        test_set_AdsSlabSite = management['test_set_AdsSlabSite']
        
        predict_observations_trainingset = management['predict_observations_trainingset']
        predict_observations_testset = management['predict_observations_testset']
        std_trainingset = management['std_trainingset']
        std_testset = management['std_testset']

        AdsSlabSite_trainingset = pd.DataFrame(training_set_AdsSlabSite,columns=col_var)        
        if 'energyDFT' not in col_var:
            AdsSlabSite_trainingset.insert(loc=len(col_var),column='energyDFT',value=training_set_observations)
        else:
            energyDFT = pd.DataFrame({'energyDFT':training_set_observations})
            AdsSlabSite_trainingset.update(energyDFT)
        AdsSlabSite_trainingset.insert(loc=len(col_var),column='Std',value=std_trainingset)
        AdsSlabSite_trainingset.insert(loc=len(col_var),column='energyML',value=predict_observations_trainingset)
        AdsSlabSite_trainingset.to_csv('trainingset_roll.csv',index=False)  # this sholud include the finally most stable states after several iterations
        
        AdsSlabSite_testset = pd.DataFrame(test_set_AdsSlabSite,columns=col_var)
        if 'energyDFT' not in col_var:
            AdsSlabSite_testset.insert(loc=len(col_var),column='energyDFT',value=test_set_observations)  ###
            
        else:
            energyDFT = pd.DataFrame({'energyDFT':test_set_observations})
        AdsSlabSite_testset.insert(loc=len(col_var),column='Std',value=std_testset)
        AdsSlabSite_testset.insert(loc=len(col_var),column='energyML',value=predict_observations_testset)
        AdsSlabSite_testset.to_csv('testset_roll.csv',index=False)
        Number_of_training_set_AdsSlabSite = len(training_set_AdsSlabSite) 
        Number_of_test_set_AdsSlabSite = len(test_set_AdsSlabSite)
        return Number_of_training_set_AdsSlabSite, Number_of_test_set_AdsSlabSite
        
    def plot_deltaE(self,management):
        # Mean Absolute Error
        deltaE_trainingset = management['deltaE_trainingset']
        abs_deltaE = [abs(i) for i in deltaE_trainingset]
        MAE_trainingset = sum(abs_deltaE)/len(deltaE_trainingset)
        print ("The Mean Absolute Error of Training Set is = %s !" % round(MAE_trainingset,3))
        plt.hist(deltaE_trainingset,color='b',edgecolor='black',bins=20,rwidth=5,density=True)
        plt.xlabel('E$_{DFT}$ - E$_{ML}$ (eV)',fontsize=14)
        plt.ylabel('Normalized frequency',fontsize=14)
        plt.xlim(-1.5,1.5)
        plt.text(0.4,1.5,"MAE = %s" % round(MAE_trainingset,3),fontsize=14)
        plt.tight_layout()
        plt.savefig('dE_trainingset.pdf')
        plt.show()
        deltaE_testset = management['deltaE_testset']
        abs_deltaE = [abs(i) for i in deltaE_testset]
        MAE_testset = sum(abs_deltaE)/len(deltaE_testset)
        print ("The Mean Absolute Error of Test Set is = %s !" % round(MAE_testset,3))
        plt.hist(deltaE_testset,color='r',edgecolor='black',bins=20,rwidth=5,density=True)
        plt.xlabel('E$_{DFT}$ - E$_{ML}$ (eV)',fontsize=14)
        plt.ylabel('Normalized frequency',fontsize=14)
        plt.xlim(-1.5,1.5)
        plt.text(0.4,0.6,"MAE = %s" % round(MAE_testset,3),fontsize=14)
        plt.tight_layout()
        plt.savefig('dE_testset.pdf')
        plt.show()
        return MAE_trainingset, MAE_testset
        
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
        #print (len(training_set_observations),len(predict_observations_trainingset),len(test_set_observations),len(predict_observations_testset),len(std_trainingset),len(std_testset))
        plt.errorbar(training_set_observations,predict_observations_trainingset,std_trainingset,fmt='bo')
        plt.errorbar(test_set_observations,predict_observations_testset,std_testset,fmt='ro')
        plt.legend(['train','test'],loc='best',fontsize=14)
        plt.xlabel('E$_{DFT}$ (eV)',fontsize=14)
        plt.ylabel('E$_{ML}$ (eV)',fontsize=14)
        t = np.linspace(-8,8,30)
        y = t
        plt.plot(t,y,'--',)
        plt.xlim(-2.5,7.5)
        plt.ylim(-2.5,7.5)
        plt.tight_layout()
        plt.savefig('gpr.pdf')
        plt.show()
        Number_of_trainingset = len(training_set_observations)
        Number_of_testset = len(test_set_observations)
        return Number_of_trainingset, Number_of_testset
        
#class Formation_energy:
    # this class is to generate the formation energy at specific chemical_state
    def calc_formation_energy(self,chemical_potential_A,delta_mu_atomA,
                              chemical_potential_B,delta_mu_atomB,chemical_potential_C,delta_mu_atomC,
                              chemical_potential_D,delta_mu_atomD,index_of_samples):
        """
        this class gets the value:
        1). Ead_ml(A@B) + Ead_df(B) - n * sigma(A@B) -n_i(A,B)*deltaMu]
        2). min([Ead_dft(A@B) + Ead_dft(B) - n_i(A,B)*deltaMu]
        This two values will be used to run the iterations according to:
        Ead_ml(A@B) + Ead_dft(B) - n * sigma(A@B) -n_i(A,B)*deltaMu] <= min([Ead_dft(A@B) + Ead_dft(B) - n_i(A,B)*deltaMu]
        """
        
        #data = 'training_roll.csv' generated by min_based_iteration.save_AdsSlabSite() function
        df_tr = pd.read_csv('trainingset_roll.csv')    # df_tr  is df_trainingset
        E_Ads1_Ads2_DFT = df_tr['energyDFT'].tolist()
        Ads_Nums = self.NumOfAtoms(df=df_tr)
        # E_Ads1_Ads2 = E(Ads1|Ads2) is the adsorption energy after the Ads2 adsorbed on surface.
        Ef_DFT = []
        
        """ from this step, we should get min(Ead_dft(A@B)+Ead(B)) at various environment"""
        String,E_x_ml,E_x_dft = self.get_string(df_tr)
        
        E_corr_ml,E_corr_dft = self.get_corrections(df_tr,String,E_x_ml,E_x_dft)
        for k in range(len(E_Ads1_Ads2_DFT)):
            E_f = E_Ads1_Ads2_DFT[k] + E_corr_dft[k] # normal formation enrgy    # E_corr_dft
            formation_energy = E_f


            if self.ref_atomType_chemPot is not None:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = Ads_Nums[k][chemical_potential_C]
                    num_atomD = Ads_Nums[k][chemical_potential_D]
            else:
                if len(self.atomType_chemPot) == 4:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = Ads_Nums[k][chemical_potential_C]
                    num_atomD = Ads_Nums[k][chemical_potential_D]

                if len(self.atomType_chemPot) == 3:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = Ads_Nums[k][chemical_potential_C]
                    num_atomD = 0                   

                if len(self.atomType_chemPot) == 2:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = 0
                    num_atomD = 0
                    
                if len(self.atomType_chemPot) == 1:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = 0
                    num_atomC = 0
                    num_atomD = 0
                
            E_f_c = self.calc_nd_phase_diagram_formation_energy(formation_energy,num_atomA,num_atomB,num_atomC,num_atomD,
                                                                delta_mu_atomA,delta_mu_atomB,delta_mu_atomC,delta_mu_atomD)

            Ef_DFT.append(E_f_c)
            
        min_dft = min(Ef_DFT)   # min(Ead_dft(A@B)+Ead(B))
        Index = Ef_DFT.index(min_dft)   # This Index will be used to output the most stable structure
        
        #--------------------------------------------------------------------------
        """  from this step, we should get some data (Ead_dft(A@B)+Ead(B)-n*sigam(A@B)) at various environment(chemical_potential correction)""" 
        df_te = pd.read_csv('testset_roll.csv')  # df_te is df_testset
        Ef_ML = []
        E_Ads1_Ads2_ML = df_te['energyML'].tolist()
        Std_Ads1_Ads2_ML = df_te['Std'].tolist()
        Ads_Nums = self.NumOfAtoms(df=df_te)
        E_corr_ml,E_corr_dft = self.get_corrections(df_te,String,E_x_ml,E_x_dft)
        for k in range(len(E_Ads1_Ads2_ML)):
            E_f = E_Ads1_Ads2_ML[k] + E_corr_dft[k] - self.n * Std_Ads1_Ads2_ML[k] + self.factor ###  E_corr_dft  or E_corr_ml
            formation_energy = E_f
            

            if self.ref_atomType_chemPot is not None:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = Ads_Nums[k][chemical_potential_C]
                    num_atomD = Ads_Nums[k][chemical_potential_D]
            else:
                if len(self.atomType_chemPot) == 4:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = Ads_Nums[k][chemical_potential_C]
                    num_atomD = Ads_Nums[k][chemical_potential_D]

                if len(self.atomType_chemPot) == 3:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = Ads_Nums[k][chemical_potential_C]
                    num_atomD = 0  # just a number                  

                if len(self.atomType_chemPot) == 2:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = Ads_Nums[k][chemical_potential_B]
                    num_atomC = 0
                    num_atomD = 0
                    
                if len(self.atomType_chemPot) == 1:
                    num_atomA = Ads_Nums[k][chemical_potential_A]
                    num_atomB = 0
                    num_atomC = 0
                    num_atomD = 0
                
            E_f_c = self.calc_nd_phase_diagram_formation_energy(formation_energy,num_atomA,num_atomB,num_atomC,num_atomD,
                                                                delta_mu_atomA,delta_mu_atomB,delta_mu_atomC,delta_mu_atomD)  
            Ef_ML.append(E_f_c)
        min_ml = min(Ef_ML)
        
        #----------------------------------------------------------------------------
        if self.random_state is True:
            sizeOf_dft = len(Ef_DFT)
            deltaE = [value for value in list(set(Ef_ML)) if value <= min_dft]
            if len(deltaE) > 10:
                k = self.kk
                list_dE = random.sample(population=deltaE,k=k)
                new_testset_index = [Ef_ML.index(de) for de in list_dE]
                var3 = [Index, min_dft, min_ml, new_testset_index]
                return var3, sizeOf_dft
                
            elif 0 < len(deltaE) <= 10:
                k = 1
                list_dE = random.sample(population=deltaE,k=k)
                new_testset_index = [Ef_ML.index(de) for de in list_dE]
                var3 = [Index, min_dft, min_ml, new_testset_index]
                return var3, sizeOf_dft
            else:
                print (" # # # # # # # # Iterations Have Convereged # # # # # # # # #")
                print ("########### Summary ##########:")
                print ("The number of samples = %s" % index_of_samples)
                print ("The number of converged samples = %s" % sizeOf_dft)
                var2 = [Index, min_dft, min_ml]
                return var2, sizeOf_dft           
                
        else:
            deltaE = sorted([value for value in list(set(Ef_ML)) if value <= min_dft])
            sizeOf_dft = len(Ef_DFT)
            if len(deltaE) > 10:
                k = self.kk
                list_dE = deltaE[:k]
                new_testset_index = [Ef_ML.index(de) for de in list_dE]
                var3 = [Index, min_dft, min_ml, new_testset_index]
                return var3, sizeOf_dft
                
            elif 0 < len(deltaE) <= 10:
                k = 1
                list_dE = deltaE[:k]
                new_testset_index = [Ef_ML.index(de) for de in list_dE]
                var3 = [Index, min_dft, min_ml, new_testset_index]
                return var3, sizeOf_dft
                
            else:
                print (" # # # # # # # # Iterations Have Convereged # # # # # # # # #")
                print ("########### Summary ##########:")
                print ("The number of samples = %s" % index_of_samples)
                print ("The number of converged samples = %s" % sizeOf_dft)
                
                print ('min_dft:',min_dft)
                print ('min_ml:',min_ml)
                #print (df_tr.values.tolist()[Index][0:6]) ######################
                var2 = [Index, min_dft, min_ml]
                return var2, sizeOf_dft

        
    def get_string(self,df):
        """ df: 'trainingset_roll.csv' """
        Ads1 = df['Ads1'].tolist()
        Site1 = df['Site1'].tolist()
        Slab = df['Slab'].tolist()
        E_ML_corr = df['energyML'].tolist()
        E_DFT_corr = df['energyDFT'].tolist()
    
        String = []
        E_x_ml = []
        E_x_dft = []

        for i in range(self.initial_training_number):
            """ Eads(B) """
            string = Ads1[i] + '_' + Site1[i] + '_' + Slab[i]
            e_ml = E_ML_corr[i]
            e_dft = E_DFT_corr[i]
            String.append(string)
            E_x_ml.append(e_ml)
            E_x_dft.append(e_dft)
        return String,E_x_ml,E_x_dft
    
    def get_corrections(self,df,String,E_x_ml,E_x_dft):
        """ df:'testset_roll.csv'  """
        ads_strings = ['H','O','OH','H2','O2','H2O','C','CO','CH','CH2','CH3','CH4','N','N2','NO','NH','NH2','NH3']
        Ads2 = df['Ads2'].tolist()
        Site2 = df['Site2'].tolist()
        Slab = df['Slab'].tolist()
        E_corr_ml = []
        E_corr_dft = []
        for k in range(len(Ads2)):
            ads2 = Ads2[k]
            site2 = Site2[k]
            slab = Slab[k]
            string = ads2 + '_' + site2 + '_' + slab
            #print (string)            
            if ads2 not in ads_strings:
                e_corr = 0
                E_corr_ml.append(e_corr)
                E_corr_dft.append(e_corr)
            else:
                if string in String:
                    idx = String.index(string)
                    e_corr_ml = E_x_ml[idx]
                    E_corr_ml.append(e_corr_ml)
                    
                    e_corr_dft = E_x_dft[idx]
                    E_corr_dft.append(e_corr_dft)
                else:
                    e_corr = 100
                    E_corr_ml.append(e_corr)
                    E_corr_dft.append(e_corr)
        #print len(E_corr_ml)       
        return E_corr_ml,E_corr_dft
        
        
#class Formation_energy_correction:
    """
    this calss implements the correction of formation energy at specific 'chemical potential' and 'delta_mu_atom'
    """
    def calc_nd_phase_diagram_formation_energy(self,formation_energy,num_atomA,num_atomB,num_atomC,num_atomD,delta_mu_atomA,delta_mu_atomB,delta_mu_atomC,delta_mu_atomD):
        if len(self.atomType_chemPot) == 4:
            func = formation_energy - num_atomA * delta_mu_atomA - num_atomB * delta_mu_atomB - num_atomC * delta_mu_atomC - num_atomD * delta_mu_atomD
        
        if len(self.atomType_chemPot) == 3:
            if self.ref_atomType_chemPot is None:
                func = formation_energy - num_atomA * delta_mu_atomA - num_atomB * delta_mu_atomB - num_atomC * delta_mu_atomC
            else:
                func = formation_energy - num_atomA * delta_mu_atomA - num_atomB * delta_mu_atomB - num_atomC * delta_mu_atomC - num_atomD * delta_mu_atomD
        
        if len(self.atomType_chemPot) == 2:
            if self.ref_atomType_chemPot is None:
                func = formation_energy - num_atomA * delta_mu_atomA - num_atomB * delta_mu_atomB
            else:
                func = formation_energy - num_atomA * delta_mu_atomA - num_atomB * delta_mu_atomB - num_atomC * delta_mu_atomC - num_atomD * delta_mu_atomD
        
        if len(self.atomType_chemPot) == 1:
            if self.ref_atomType_chemPot is None:
                func = formation_energy - num_atomA * delta_mu_atomA
            else:
                func = formation_energy - num_atomA * delta_mu_atomA - num_atomB * delta_mu_atomB - num_atomC * delta_mu_atomC - num_atomD * delta_mu_atomD
        return func
        
    def NumOfAtoms(self,df):
        ''' count the number of each atom in coadsorbed species '''
        Ads_Nums = []
        ads1 = df["Ads1"].tolist()
        ads2 = df["Ads2"].tolist()
        Slab = df["Slab"].tolist()
        Atom_num = self.NumOf_basis()

        for k in range(len(Slab)):
                numOf_H = Atom_num[ads1[k]]['H'] + Atom_num[ads2[k]]['H']
                numOf_C = Atom_num[ads1[k]]['C'] + Atom_num[ads2[k]]['C']
                numOf_N = Atom_num[ads1[k]]['N'] + Atom_num[ads2[k]]['N']
                numOf_O = Atom_num[ads1[k]]['O'] + Atom_num[ads2[k]]['O']
                Ads_Nums.append({'H':numOf_H,'C':numOf_C,'N':numOf_N,'O':numOf_O})
        return Ads_Nums
        
    def NumOf_basis(self):
        Atom_num = {'None':{'H':0, 'C':0,'N':0, 'O':0},
                    'H' : {'H':1, 'C':0, 'N':0, 'O':0},
                    'O' : {'H':0, 'C':0, 'N':0, 'O':1},
                    'OH' : {'H':1, 'C':0, 'N':0, 'O':1},
                    'H2' : {'H':2, 'C':0, 'N':0, 'O':0},
                    'O2' : {'H':0, 'C':0, 'N':0, 'O':2},
                    'H2O' : {'H':2, 'C':0, 'N':0, 'O':1},
                    'C' : {'H':0, 'C':1, 'N':0, 'O':0},
                    'CO' : {'H':0, 'C':1, 'N':0, 'O':1},
                    'CH' : {'H':1, 'C':1, 'N':0, 'O':0},
                    'CH2' : {'H':2, 'C':1, 'N':0, 'O':0},
                    'CH3' : {'H':3, 'C':1, 'N':0, 'O':0},
                    'CH4' : {'H':4, 'C':1, 'N':0, 'O':0},
                    'N' : {'H':0, 'C':0, 'N':1, 'O':0},
                    'N2' : {'H':0, 'C':0, 'N':2, 'O':0},
                    'NO' : {'H':0, 'C':0, 'N':1, 'O':1},
                    'NH' : {'H':1, 'C':0, 'N':1, 'O':0},
                    'NH2' : {'H':2, 'C':0, 'N':1, 'O':0},
                    'NH3' : {'H':3, 'C':0, 'N':1, 'O':0},
                    'CHO' : {'H':1,'C':1,'N':0,'O':1},
                    'CH2O': {'H':2,'C':1,'N':0,'O':1},
                    'CH3O': {'H':3,'C':1,'N':0,'O':1},
                    'NHO': {'H':1,'C':0,'N':1,'O':1},
                    'NH2O': {'H':2,'C':0,'N':1,'O':1},
                    'NH3O': {'H':3,'C':0,'N':1,'O':1},
                    'HONO': {'H':1,'C':0,'N':1,'O':2},
                    'HCNO':{'H':1,'C':1,'N':1,'O':1},}
                    
        return Atom_num
