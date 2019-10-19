# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:40:35 2019

@author: fuzhu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import pickle
from itertools import chain
from sklearn.preprocessing import LabelEncoder

"""
# Usage:
from Label_Encode import Run_Label_Encode
#atomType_chemPot = {'H':[-2,2],'C':[0,4]}   # 2d
#atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2],'O':[-2,2]}  # 4d
atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2]}     # 3d
N_int = 10
rLE = Run_Label_Encode(atomType_chemPot=atomType_chemPot, N_int=N_int)
XY, label_matrix = rLE.run_data_encode()
#rLE.plot_3d_slice(label_matrix)
rLE.plot_3d_pd_mlab(label_matrix)
"""
class Run_Label_Encode:
    
    def __init__(self,atomType_chemPot,N_int,ref_atomType_chemPot=None,contour3d=True,n=2,
                 plane_orientation='x_axes',slice_index=0,mu=None,axis='z',s=20,rounds=2,
                 labels='Some Units',DFT_States=None):
        """
        atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2]}
        ref_atomType_chemPot = ['O':2.4]
        mu = [] # a list. For example, mu = [0.2]; mu = [0.2,0.55]
        """
        self.atomType_chemPot = atomType_chemPot
        self.N_int = N_int
        self.n = n
        self.ref_atomType_chemPot = ref_atomType_chemPot
        self.contour3d = contour3d
        self.plane_orientation = plane_orientation
        self.slice_index = slice_index
        self.mu = mu
        self.axis = axis
        self.s = s
        self.rounds = rounds
        self.labels = labels
        self.DFT_States = DFT_States
        
        
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
    
    def run_data_encode(self):
        LE = LabelEncoder()
        name = self.str_name()
        N = self.N_int
        n = self.n
        data = pickle.load(open('matrix_structures_%s_%s_%s.pickle' % (name,N,n),'rb'))
        X, XXX, IV = self.unique_encoder(data)
        A, B = self.drop_duplicate(X, XXX, IV)
        
        LE.fit(X)
        Y = LE.transform(X)
        XY = dict(zip(X,Y))
        print ("=====================================")
        print (XY)
        print ("=====================================")
      
        #---------------------------------
        if len(self.atomType_chemPot) == 1:
            for k, dat in enumerate(data):
                if dat in X:
                    pass
                else:
                    index = B.index(dat)
                    data[k] = A[index]
            
            label_matrix = []
            coder = LE.transform(data)
            label_matrix.append(coder)
            
            with open ('label_matrix_%s_%s_%s.pickle' % (name,N,self.n),'wb') as data_label:
                pickle.dump(label_matrix,data_label)
            
            energy_matrix = pickle.load(open('matrix_energies_%s_%s_%s.pickle' % (name,N,self.n),'rb'))
            self.plot_1d_pd(energy_matrix) #### 1D
            All_Str = pickle.load(open('All_Str_%s_%s_%s.pickle' % (name,N,self.n),'rb'))
            self.plot_nums_Str(All_Str)
        # ----------------------------------
        
        if len(self.atomType_chemPot) == 2:
            for k, dat in enumerate(data):
                for z, ele in enumerate(dat):
                    if ele in X:
                        pass
                    else:
                        index = B.index(ele)
                        data[k][z] = A[index]
        
            label_matrix = []
            if self.DFT_States is None:
                for k, mat in enumerate(data):
                    coder = LE.transform(mat)
                    label_matrix.append(list(coder))
            else:
                Keys = list(self.DFT_States.keys())
                
                Keys_switch = [] ######### important #######
                for k, key in enumerate(Keys):
                    ss = key.split('-')
                    Keys_switch.append('-'.join([ss[2],ss[3],ss[0],ss[1],ss[4]]))
                
                for k, Mat in enumerate(data):
                    Coder = []
                    for mat in Mat:
                        
                        for k, m in enumerate(Keys_switch):
                            if mat == m:
                                mat = Keys[k]
                        
                        if len(X) <= len(self.DFT_States):
                            if mat in Keys:
                                coder = self.DFT_States[mat]
                                Coder.append(coder)
                            else:
                                dN = list(set(X)-set(Keys))
                                ind = range(len(Keys),len(Keys)+len(dN))
                                dN_ind = dict(zip(dN,ind))
                                coder = len(self.DFT_States) + dN_ind[mat]
                                Coder.append(coder)
                        else:
                            if mat in Keys:
                                coder = self.DFT_States[mat]
                                Coder.append(coder)
                            else:
                                dN = list(set(X)-set(Keys))
                                ind = range(len(dN))
                                dN_ind = dict(zip(dN,ind))
                                coder = len(Keys) + dN_ind[mat]
                                Coder.append(coder)
                    label_matrix.append(Coder)

            with open ('label_matrix_%s_%s_%s.pickle' % (name,N,self.n),'wb') as data_label:
                pickle.dump(label_matrix,data_label)
            
            self.plot_2d_pd(label_matrix) #### 2D            
            #All_Str = pickle.load(open('All_Str_%s_%s_%s.pickle' % (name,N,self.n),'rb'))  ##################################?????
            #self.unique_structures_2d(All_Str)
        #-------------------------------------
                
        if len(self.atomType_chemPot) == 3:
            for k, dat in enumerate(data):
                for z, elems in enumerate(dat):
                    for w, elem in enumerate(elems):
                        if elem in X:
                            pass
                        else:
                            index = B.index(elem)
                            data[k][z][w] = A[index]
            
            label_matrix = []
            for k, mat in enumerate(data):
                label_matrix.append([])
                for elems in mat:
                    coder = LE.transform(elems)
                    label_matrix[k].append(list(coder))
                    
            with open ('label_matrix_%s_%s_%s.pickle' % (name,N,self.n),'wb') as data_label:
                pickle.dump(label_matrix,data_label)
                            
            self.plot_3d_pd(label_matrix)  #### 3D
        #--------------------------------------
           
        if len(self.atomType_chemPot) == 4:
            for k, dat in enumerate(data):
                for z, elems in enumerate(dat):
                    for w, elem in enumerate(elems):
                        for p, ele in enumerate(elem):
                            if elem in X:
                                pass
                            else:
                                index = B.index(ele)
                                data[k][z][w][p] = A[index]
        
            label_matrix = []
            for k, mat in enumerate(data):
                label_matrix.append([])
                for z, elems in enumerate(mat):
                    label_matrix[k].append([])
                    for elem in elems:
                        coder = LE.transform(elem)
                        label_matrix[k][z].append(list(coder))
                    
            with open ('label_matrix_%s_%s_%s.pickle' % (name,N,self.n),'wb') as data_label:
                pickle.dump(label_matrix,data_label)
        
        return XY, label_matrix
                            
        
    def unique_encoder(self,data):
        if len(self.atomType_chemPot) == 1:
            X = list(set(data))
            
            XX = [x.split('-') for x in X]
            XXX = [sorted(x) for x in XX]
            
        if len(self.atomType_chemPot) == 2:
            X = list(set((chain.from_iterable(data))))
            
            XX = [x.split('-') for x in X]
            XXX = [sorted(x) for x in XX]
            
        if len(self.atomType_chemPot) == 3:
            x_data = list((chain.from_iterable(data)))
            X = list(set((chain.from_iterable(x_data))))
            
            XX = [x.split('-') for x in X]
            XXX = [sorted(x) for x in XX]

        if len(self.atomType_chemPot) == 4:
            x_data = list((chain.from_iterable(data)))
            xx_data = list((chain.from_iterable(x_data)))
            X = list(set((chain.from_iterable(xx_data))))
            
            XX = [x.split('-') for x in X]
            XXX = [sorted(x) for x in XX]
            
        IV = []
        for x in XXX:
            if x not in IV:
                IV.append(x)
        return X, XXX, IV
    
    def drop_duplicate(self,X, XXX, IV):
        A = []
        B = []
        for x in IV:
            indx = XXX.index(x)
            strut = X[indx]
            F = strut.split('-')
            rep_elem = '-'.join([F[2],F[3],F[0],F[1],F[4]])
            
            for k, ele in enumerate(X):
                if strut != rep_elem:
                    if ele == rep_elem:
                        X[k] = strut
                        A.append(strut)
                        B.append(ele)
        return A, B


    def plot_1d_pd(self,energy_matrix):
        atomA, muA = self.mu_X()
        name = self.str_name()
        N = self.N_int
        fig, ax = plt.subplots()
        ax.plot(muA,energy_matrix,linewidth=1.2)  # Dark2,   # cmap=plt.cm.get_cmap('RdBu_r')
        ax.set_xlabel('$\mu_%s$ (eV)' % atomA,fontsize=16)
        ax.set_ylabel('Formation energy (eV)', fontsize=16)
        ax.tick_params(axis='both',direction='in',labelsize=14)
        fig.tight_layout()
        plt.savefig('1d_pd_%s_%s.pdf' % (name,N))
        plt.savefig('1d_pd_%s_%s.png' % (name,N))
        

    def plot_nums_Str(self,All_Str):
        atomA, muA = self.mu_X()
        name = self.str_name()
        N = self.N_int
        # Num. of cumulative\nstructures
        nums = []
        add_str = All_Str[0]
        nums.append(len(add_str))
        for k in range(1,N):
            add_str_k = All_Str[k]
            for stru in add_str_k:
                if stru not in add_str:
                    add_str.append(stru)
            num = len(add_str)
            nums.append(num)        
        
        fig, ax = plt.subplots()
        ax.plot(muA,nums,linewidth=1.2)  # Dark2,   # cmap=plt.cm.get_cmap('RdBu_r')
        ax.set_xlabel('$\mu_%s$ (eV)' % atomA,fontsize=16)
        ax.set_ylabel('# Num. of cumulative\nstructures', fontsize=16)
        ax.tick_params(axis='both',direction='in',labelsize=14)
        fig.tight_layout()
        plt.savefig('1d_nums_%s_%s.pdf' % (name,N))
        plt.savefig('1d_nums_%s_%s.png' % (name,N))
        
        
    def plot_2d_pd(self,label_matrix):
        atomA, muA, atomB, muB = self.mu_X()
        name = self.str_name()
        N = self.N_int
        fig, ax = plt.subplots()
        p = ax.pcolor(muB,muA,label_matrix,cmap=plt.cm.get_cmap('tab20c'),vmin=0,vmax=20)  # Dark2,   # cmap=plt.cm.get_cmap('RdBu_r')
        fig.colorbar(p)
        ax.set_xlabel('$\mu_%s$ (eV)' % atomB,fontsize=16)
        ax.set_ylabel('$\mu_%s$ (eV)' % atomA,fontsize=16)
        ax.tick_params(axis='both',direction='in',labelsize=14)
        fig.tight_layout()
        plt.savefig('2d_pd_%s_%s.pdf' % (name,N))
        plt.savefig('2d_pd_%s_%s.png' % (name,N))
        
    def unique_structures_2d(self,All_Str):
        List = []
        for k in range(len(All_Str)):
            List += All_Str[k]
        all_List = []
        for k in range(len(List)):
            all_List += List[k]
        unique_structures = list(set(all_List))
        nums_2d = len(unique_structures)
        df = open('nums_2d.txt','a')
        df.write("The numer of unique structures are:")
        df.write(str(nums_2d))
        df.write('\nNew information:')
        df.close()
        
    def plot_3d_pd(self,label_matrix):
        atomA, muA, atomB, muB, atomC, muC = self.mu_X()
        x = muA #  ==  np.linsacpe(-2,2,N)
        y = muB #  ==  np.linspace(-2,2,N)
        z = muC #  ==  np.linspace(0,4,N)
        nx, ny, nz = np.meshgrid(x,y,z)
        x_data = list((chain.from_iterable(label_matrix)))
        color = list((chain.from_iterable(x_data)))  # a list
        #print (len(color))
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        plot = ax.scatter(nx,ny,nz,zdir='z',c=color,marker='o',cmap=plt.cm.rainbow)  #
        ax.set_xlabel('$\mu_%s$ (eV)' % atomA,fontsize=14)
        ax.set_ylabel('$\mu_%s$ (eV)' % atomB,fontsize=14)
        ax.set_zlabel('$\mu_%s$ (eV)' % atomC,fontsize=14)
        fig.colorbar(plot)
        name = self.str_name()
        N_int = self.N_int
        plt.savefig('3d_%s_%s.pdf' % (name,N_int))
        plt.savefig('3d_%s_%s.png' % (name,N_int))
        plt.show()
    
    def plot_3d_slice(self,label_matrix): 
        atomA, muA, atomB, muB, atomC, muC = self.mu_X()
        X = muA #  ==  np.linspace(-2,2,N)
        Y = muB #  ==  np.linspace(-2,2,N)
        Z = muC #  ==  np.linspace(0,4,N)
        nx, ny, nz = np.meshgrid(X,Y,Z)
        mu = self.mu

        print ('Reminder:\n[Please Check Whether mu[k] In np.linsapce(a,b,N))]')       
        
        CC = []  # color
        SS = []  #  size
        for mu_i in mu:
            cc = []  # color
            ss = []  #  size
            if self.axis == 'z':
                for m in Z:
                    if mu_i == round(m,self.rounds):
                        indx = list(Z).index(m)
                            
                for i, x in enumerate(label_matrix):
                    for j, y in enumerate(x):
                        for k, z in enumerate(y):  
                            cc.append(np.array(label_matrix[i][j][k]))
                            if k == indx:
                                ss.append(self.s)
                            else:
                                ss.append(0)

            if self.axis == 'y':
                for m in Y:
                    if mu_i == round(m,self.rounds):
                        indx = list(Y).index(m)
                for i, x in enumerate(label_matrix):
                    for j, y in enumerate(x):
                        for k, z in enumerate(y):
                            cc.append(np.array(label_matrix[i][j][k]))
                            if j == indx:
                                ss.append(self.s)
                            else:
                                ss.append(0)

            if self.axis == 'x':
                for m in X:
                    if mu_i == round(m,self.rounds):
                        indx = list(X).index(m)
                for i, x in enumerate(label_matrix):
                    for j, y in enumerate(x):
                        for k, z in enumerate(y):
                            cc.append(np.array(label_matrix[i][j][k]))
                            if i == indx:
                                ss.append(self.s)
                            else:
                                ss.append(0)
            CC.append(cc)
            SS.append(ss)
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k in range(len(mu)):
            ax.scatter(nx, ny, nz, zdir=self.axis, c=CC[k], s=SS[k], cmap=plt.cm.rainbow)
            
        ax.set_xlabel('$\mu_%s$ (eV)' % atomA,fontsize=14)
        ax.set_ylabel('$\mu_%s$ (eV)' % atomB,fontsize=14)
        ax.set_zlabel('$\mu_%s$ (eV)' % atomC,fontsize=14)
        name = self.str_name()
        N_int = self.N_int
        axis = self.axis
        plt.savefig('3d_%s_%s_%s_%s.pdf' % (name,N_int,axis,self.n))
        plt.savefig('3d_%s_%s_%s_%s.png' % (name,N_int,axis,self.n))
        plt.show()
        
    def plot_3d_pd_mlab(self,label_matrix):
        #http://docs.enthought.com/mayavi/mayavi/auto/mlab_pipeline_reference.html 
        from mayavi import mlab
        atomA, muA, atomB, muB, atomC, muC = self.mu_X()
        x = muA # np.linsacpe(-2,2,N)
        y = muB # np.linspace(-2,2,N)
        z = muC # np.linspace(0,4,N)
        nx, ny, nz = np.meshgrid(x,y,z)
        if self.contour3d is True:
            mlab.contour3d(label_matrix)
        else:    # cut plane
            new_label_matrix = self.reshape_3d_data(label_matrix)
            mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(new_label_matrix),
                                             plane_orientation=self.plane_orientation,
                                             slice_index=self.slice_index)
                                         
        mlab.outline()
        mlab.show()
        
    def reshape_3d_data(self,label_matrix):
        N_int = self.N_int
        x_data = list((chain.from_iterable(label_matrix)))
        LM = list((chain.from_iterable(x_data)))
        r_LM = np.array(LM)
        return r_LM.reshape(N_int,N_int,N_int)
        
    def only_colorbar(self,XY,label_materix):
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cmap = mpl.cm.rainbow   # mpl.cm.hot, mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=len(XY))
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
                                        
        labels = self.labels
        cb.set_label(labels)
        
