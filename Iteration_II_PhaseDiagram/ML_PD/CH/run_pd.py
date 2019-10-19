# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 12:48:54 2019

@author: fuzhu
"""


import pandas as pd
from Phase_Diagram import Run_phase_diagram

data_ASS = 'formation_ASS_X.csv'
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
#atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2],'O':[-2,2]}  # 4d
#atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2]}     # 3d
#ref_atomType_chemPot = {'O':2}  # 3d
atomType_chemPot = {'H':[-1,0],'C':[3,4]}   # 2d
#atomType_chemPot = {'H':[-2,2]}   # 1d
n = 1
N_int = 50
rPD = Run_phase_diagram(data_ml=data_ml,data_ASS=data_ASS,atomType_chemPot=atomType_chemPot,N_int=N_int,n=n,initial_training_number=43,kk=1)
if len(atomType_chemPot) == 1:
    rPD.run_1d()
if len(atomType_chemPot) == 2:
    rPD.run_2d()
if len(atomType_chemPot) == 3:
    rPD.run_3d()
if len(atomType_chemPot) == 4:
    rPD.run_4d()

from Label_Encode import Run_Label_Encode
#atomType_chemPot = {'H':[-2,2],'C':[0,4]}   # 2d
#atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2],'O':[-2,2]}  # 4d
#atomType_chemPot = {'H':[-2,2],'C':[0,4],'N':[-2,2]}     # 3d
#N_int = 10
rLE = Run_Label_Encode(atomType_chemPot=atomType_chemPot,N_int=N_int,n=n)
XY, label_matrix = rLE.run_data_encode()
rLE.plot_2d_pd(label_matrix)
#rLE.plot_3d_pd_mlab(label_matrix)

"""
from mayavi import mlab
#mlab.contour3d(label_matrix)
#mlab.pipeline.volume(mlab.pipeline.scalar_field(label_matrix))
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(label_matrix),plane_orientation='x_axes',slice_index=0,)
#mlab.volume_slice(label_matrix,plane_orientation='x_axes',slice_index=0)
mlab.outline()
mlab.show()
"""
