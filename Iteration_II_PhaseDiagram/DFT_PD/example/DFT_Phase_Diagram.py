# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:29:29 2019

@author: fuzhu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:09:42 2019

@author: fuzhu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    def __init__(self,data_ASS,atomType_chemPot,N_int,ref_atomType_chemPot=None,initial_training_number=43):
        """
        atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2]} # 3d
        ref_atomType_chemPot = {'O':0}  # 3d
        
        atomType_chemPot = {'H':[-2,2],'C':[0,4]} # 2d
        ref_atomType_chemPot = {'O':0,'N':0}  # 2d
        
        if ref_atomType_chemPot = None, the ref is to 0
        """

        self.data_ASS = data_ASS
        self.atomType_chemPot = atomType_chemPot
        self.ref_atomType_chemPot = ref_atomType_chemPot
        self.N_int = N_int
        self.initial_training_number = initial_training_number

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
        
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures
            
        for stateA in muA:
            delta_mu_atomA = stateA
            minE, stableStr = self.launch_finder(chemical_potential_A,delta_mu_atomA,
                                                    chemical_potential_B,delta_mu_atomB,
                                                    chemical_potential_C,delta_mu_atomC,
                                                    chemical_potential_D,delta_mu_atomD)

            Matrix_energy.append(minE)
            Matrix_structure.append(stableStr)


        new_matrix = []
        for mat in Matrix_structure:
            del mat[-3:]
            new_elem = '-'.join(mat)
            new_matrix.append(new_elem)
                
        name = self.str_name()
        #print (new_matrix)
        with open('matrix_structures_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(Matrix_energy, data)


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
        
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures
        for i in range(self.N_int):
            Matrix_energy.append([])
            Matrix_structure.append([])
        for k, stateA in enumerate(muA):
            delta_mu_atomA = stateA
            for stateB in muB:
                delta_mu_atomB = stateB
                minE, stableStr = self.launch_finder(chemical_potential_A,delta_mu_atomA,
                                                     chemical_potential_B,delta_mu_atomB,
                                                     chemical_potential_C,delta_mu_atomC,
                                                     chemical_potential_D,delta_mu_atomD)
                Matrix_energy[k].append(minE)
                Matrix_structure[k].append(stableStr)


        new_matrix = []
        for i in range(self.N_int):
            new_matrix.append([])
            
        for k, mat in enumerate(Matrix_structure):
            for elem in mat:
                del elem[-3:]
                new_elem = '-'.join(elem)
                new_matrix[k].append(new_elem)
                
        name = self.str_name()
        #print (new_matrix)
        with open('matrix_structures_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(Matrix_energy, data)
            
            
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
        
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures
        for i in range(self.N_int):
            Matrix_energy.append([])
            Matrix_structure.append([])
            for j in range(self.N_int):
                Matrix_energy[i].append([])
                Matrix_structure[i].append([])
                
        for k, stateA in enumerate(muA):
            delta_mu_atomA=stateA
            for z, stateB in enumerate(muB):
                delta_mu_atomB=stateB
                for stateC in muC:
                    delta_mu_atomC=stateC
                    minE, stableStr = self.launch_finder(chemical_potential_A,delta_mu_atomA,
                                                         chemical_potential_B,delta_mu_atomB,
                                                         chemical_potential_C,delta_mu_atomC,
                                                         chemical_potential_D,delta_mu_atomD)
                    Matrix_energy[k][z].append(minE)
                    Matrix_structure[k][z].append(stableStr)


        new_matrix = []
        for i in range(self.N_int):
            new_matrix.append([])
            for j in range(self.N_int):
                new_matrix[i].append([])
        
        for k, mat in enumerate(Matrix_structure):
            for z, elems in enumerate(mat):
                for elem in elems:
                    del elem[-3:]
                    new_elem = '-'.join(elem)
                    new_matrix[k][z].append(new_elem)
        
        name = self.str_name()            
        #print (new_matrix)
        with open('matrix_structures_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(Matrix_energy, data)
          

    def run_4d(self):
        atomA, muA, atomB, muB, atomC, muC, atomD, muD = self.mu_X()
        chemical_potential_A = atomA
        chemical_potential_B = atomB
        chemical_potential_C = atomC
        chemical_potential_D = atomD
        
        Matrix_energy = []  # storage energies
        Matrix_structure = []  # storage structures
        for k in range(self.N_int):
            Matrix_energy.append([])
            Matrix_structure.append([])
            for z in range(self.N_int):
                Matrix_energy[k].append([])
                Matrix_structure[k].append([])
                for w in range(self.N_int):
                    Matrix_energy[k][z].append([])
                    Matrix_structure[k][z].append([])
                
        for k, stateA in enumerate(muA):
            delta_mu_atomA=stateA
            for z, stateB in enumerate(muB):
                delta_mu_atomB=stateB
                for w, stateC in enumerate(muC):
                    delta_mu_atomC=stateC
                    for stateD in muD:
                        delta_mu_atomD = stateD
                        minE, stableStr = self.launch_finder(chemical_potential_A,delta_mu_atomA,
                                                             chemical_potential_B,delta_mu_atomB,
                                                             chemical_potential_C,delta_mu_atomC,
                                                             chemical_potential_D,delta_mu_atomD)
                        Matrix_energy[k][z][w].append(minE)
                        Matrix_structure[k][z][w].append(stableStr)


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
                        del D[-3:]
                        new_ele = '-'.join(D)
                        new_matrix[k][z][w].append(new_ele)
        
        name = self.str_name()            
        #print (new_matrix)
        with open('matrix_structures_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(new_matrix,data)

        with open('matrix_energies_%s_%s.pickle' % (name,self.N_int),'wb') as data:
            pickle.dump(Matrix_energy, data)
        

    
    """
    [Ead_ml(A@B) + Ead_df(B) - n * sigma(A@B) - n_i(A,B)*deltaMu] <= min([Ead_dft(A@B) + Ead_dft(B) - n_i(A,B)*deltaMu])
    
    The passing data is:
    data_ml = 'ml_data_f.csv'
    data_ASS = 'for_formation_AdsSlabSite.csv'
    
    """
    def launch_finder(self,chemical_potential_A, delta_mu_atomA,
                         chemical_potential_B, delta_mu_atomB,
                         chemical_potential_C, delta_mu_atomC,
                         chemical_potential_D, delta_mu_atomD):
        #-----------------------------------------------------------------------------------------
        #initialize the analysized data
        # read the columns names of ads_slab_site
        data_ASS= self.data_ASS
        df_origin = pd.read_csv(data_ASS)
        df_origin = df_origin.dropna(axis=0,how='any')
        df = df_origin
        #*****************************************************************************************

        minE, stableStr = self.calc_formation_energy(df,chemical_potential_A,delta_mu_atomA,
                                                     chemical_potential_B,delta_mu_atomB,
                                                     chemical_potential_C,delta_mu_atomC,
                                                     chemical_potential_D,delta_mu_atomD,)
        
        return minE, stableStr
    
        
                
#class Formation_energy:
    # this class is to generate the formation energy at specific chemical_state
    def calc_formation_energy(self,df,chemical_potential_A,delta_mu_atomA,
                              chemical_potential_B,delta_mu_atomB,chemical_potential_C,delta_mu_atomC,
                              chemical_potential_D,delta_mu_atomD):
        """
        this class gets the value:
        1). Ead_ml(A@B) + Ead_df(B) - n * sigma(A@B) -n_i(A,B)*deltaMu]
        2). min([Ead_dft(A@B) + Ead_dft(B) - n_i(A,B)*deltaMu]
        This two values will be used to run the iterations according to:
        Ead_ml(A@B) + Ead_dft(B)] <= min([Ead_dft(A@B) + Ead_dft(B)]
        """
        
        #data = 'training_roll.csv' generated by min_based_iteration.save_AdsSlabSite() function
        initial_training_number = self.initial_training_number
        df_tr = df.iloc[:initial_training_number,:]
        E_Ads1_Ads2_DFT = df_tr['Gf'].tolist()
        Ads_Nums = self.NumOfAtoms(df=df_tr)
        # E_Ads1_Ads2 = E(Ads1|Ads2) is the adsorption energy after the Ads2 adsorbed on surface.
        Ef_DFT_lc = []
        
        """ from this step, we should get min(Ead_dft(A@B)+Ead(B)) at various environment"""
        String,E_x_dft = self.get_string(df_tr)
        
        E_corr_dft = self.get_corrections(df_tr,String,E_x_dft)
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

            Ef_DFT_lc.append(E_f_c)
        
        
        #--------------------------------------------------------------------------
        """  from this step, we should get some data (Ead_dft(A@B)+Ead(B)) at various environment(chemical_potential correction)""" 
        df_te = df.iloc[initial_training_number:,:]  ##
        Ef_DFT_hc = []
        E_Ads1_Ads2_DFT_hc = df_te['Gf'].tolist()
        Ads_Nums = self.NumOfAtoms(df=df_te)
        E_corr_dft = self.get_corrections(df_te,String,E_x_dft)
        for k in range(len(E_Ads1_Ads2_DFT_hc)):
            E_f = E_Ads1_Ads2_DFT_hc[k] + E_corr_dft[k]
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
            Ef_DFT_hc.append(E_f_c)
        
        Ef_DFT = Ef_DFT_lc + Ef_DFT_hc
        minE = min(Ef_DFT)
        Index = Ef_DFT.index(minE)
        df['mod_Gf'] = Ef_DFT
        stableStr = df.values.tolist()[Index]
        print ('Final stable structure:',stableStr)
        return minE, stableStr

        
    def get_string(self,df):
        """ df: 'trainingset_roll.csv' """
        Ads1 = df['Ads1'].tolist()
        Site1 = df['Site1'].tolist()
        Slab = df['Slab'].tolist()
        E_DFT_corr = df['Gf'].tolist()###
    
        String = []
        E_x_dft = []

        for i in range(self.initial_training_number):
            """ Eads(B) """
            string = Ads1[i] + '_' + Site1[i] + '_' + Slab[i]
            e_dft = E_DFT_corr[i]
            String.append(string)
            E_x_dft.append(e_dft)
        return String,E_x_dft
    
    def get_corrections(self,df,String,E_x_dft):
        """ df:'testset_roll.csv'  """
        ads_strings = ['H','O','OH','H2','O2','H2O','C','CO','CH','CH2','CH3','CH4','N','N2','NO','NH','NH2','NH3']
        Ads2 = df['Ads2'].tolist()
        Site2 = df['Site2'].tolist()
        Slab = df['Slab'].tolist()
        E_corr_dft = []
        for k in range(len(Ads2)):
            ads2 = Ads2[k]
            site2 = Site2[k]
            slab = Slab[k]
            string = ads2 + '_' + site2 + '_' + slab
            #print (string)            
            if ads2 not in ads_strings:
                e_corr = 0
                E_corr_dft.append(e_corr)
            else:
                if string in String:
                    idx = String.index(string)
                    
                    e_corr_dft = E_x_dft[idx]
                    E_corr_dft.append(e_corr_dft)
                else:
                    e_corr = 100
                    E_corr_dft.append(e_corr)
       
        return E_corr_dft
        
        
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

    def plot_algorithm(self, ALL_Ef_DFT, ALL_Ef_ML, ALL_min_DFT, ALL_min_ML):
        N_dft = len(ALL_Ef_DFT)
        N_ml = len(ALL_Ef_ML)
        
        fig, ax = plt.subplots()
        ax.tick_params(axis='both',direction='in',labelsize=14)
        for n in range(N_dft):
            N = len(ALL_Ef_DFT[n])
            x = N * [n]
            ax.scatter(x,ALL_Ef_DFT[n],color='g',label='DFT')
            ax.scatter(n,ALL_min_DFT[n],color='b',label='min_DFT')  # one point
        for n in range(N_ml):
            N = len(ALL_Ef_ML[n])
            x = N * [n]
            #ax.scatter(x,ALL_Ef_ML[n],color='g')
            ax.scatter(n,ALL_min_ML[n],color='r',label='min_ML')  # one point
        ax.set_xlabel('Iteration #',fontsize=16)
        ax.set_ylabel('Energy (eV)', fontsize=16)
        fig.tight_layout()
        plt.savefig('ML_algorithm_2.pdf')
        plt.savefig('ML_algorithm_2.png')        
        