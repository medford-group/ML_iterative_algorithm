#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:16:06 2018

@author: fuzhu
"""

import pandas as pd
from copy import deepcopy
import numpy as np
from edge_pairs import generate_edges
GE = generate_edges()

class CM_descriptors:
    
    def CoulombMatrix(self,data):
        Nodes_Ads, Edges_Ads, ordered_node, ordered_edge, Position_edges, cell = CM_descriptors.odered_nodes_edges(self,data)
        CM_fpt = []
        for i in range(len(Edges_Ads)):
            Edges_ads = deepcopy(Edges_Ads)
            edges = Edges_ads[i]
            ordered_edge_copy = deepcopy(ordered_edge)
            Position_edges_copy = deepcopy(Position_edges)
            Position_edge = deepcopy(Position_edges_copy)
            fpt = []
            position_edges = Position_edge[i]
            for edge in ordered_edge_copy:
                if edge in edges:
                    k = edges.index(edge)
                    position_edge = position_edges[k]
                    r_ij = CM_descriptors.Rij_distance(self,position_edge)
                    atomic_number = CM_descriptors.Atomic_number(self)
                    fp = CM_descriptors.CM_nondiagnal(self,edge,r_ij,atomic_number)
                    fpt.append(fp)
                    del edges[k]
                    del position_edges[k]
                else:
                    fpt.append(0)
            
            Nodes_ads = deepcopy(Nodes_Ads)
            ordered_node_copy = deepcopy(ordered_node)
            nodes = Nodes_ads[i]
            for node in ordered_node_copy:
                if node in nodes:
                    k = nodes.index(node)
                    atomic_number = CM_descriptors.Atomic_number(self)
                    fp = CM_descriptors.CM_diagnal(self,node,atomic_number)
                    fpt.append(fp)
                    nodes.remove(node)
                else:
                    fpt.append(0)
            CM_fpt.append(fpt)
        columns = [(edge[0] + '-' + edge[1]) for edge in ordered_edge] + ordered_node #### sorted(xx)
        df = pd.DataFrame(CM_fpt,columns=columns)
        df.to_csv('custom_CM_fpts.csv', index=False)
        
        data = CM_descriptors.get_atom_connectivity_CM(self,df)
        return CM_fpt, columns
                        
    def SinCoulombMatrix(self, data):
        Nodes_Ads, Edges_Ads, ordered_node, ordered_edge, Position_edges, cell = CM_descriptors.odered_nodes_edges(self,data)
        SinCM_fpt = []
        inverse_cell = CM_descriptors.cell_inverse(self, cell)
        ek = CM_descriptors.e_k(self)
        for i in range(len(Edges_Ads)):
            Edges_ads = deepcopy(Edges_Ads)
            edges = Edges_ads[i]
            ordered_edge_copy = deepcopy(ordered_edge)
            Position_edges_copy = deepcopy(Position_edges)
            Position_edge = deepcopy(Position_edges_copy)
            fpt = []
            position_edges = Position_edge[i]
            for edge in ordered_edge_copy:
                if edge in edges:
                    k = edges.index(edge)
                    position_edge = position_edges[k]
                    v_ij = CM_descriptors.Ri_Rj_vec(self,position_edge)
                    atomic_number = CM_descriptors.Atomic_number(self)
                    fp = CM_descriptors.SinCM_nondiagnal(self,edge,cell,inverse_cell,ek,v_ij,atomic_number)
                    fpt.append(fp)
                    del edges[k]
                    del position_edges[k]
                else:
                    fpt.append(0)
            
            Nodes_ads = deepcopy(Nodes_Ads)
            ordered_node_copy = deepcopy(ordered_node)
            nodes = Nodes_ads[i]
            for node in ordered_node_copy:
                if node in nodes:
                    k = nodes.index(node)
                    atomic_number = CM_descriptors.Atomic_number(self)
                    fp = CM_descriptors.CM_diagnal(self,node,atomic_number)
                    fpt.append(fp)
                    nodes.remove(node)
                else:
                    fpt.append(0)
            SinCM_fpt.append(fpt)
        columns = [(edge[0] + '-' + edge[1]) for edge in ordered_edge] + ordered_node
        df = pd.DataFrame(SinCM_fpt,columns=columns)
        df.to_csv('custom_sinCM_fpts.csv', index=False)
        return SinCM_fpt, columns
           
    def odered_nodes_edges(self,data):
        df = pd.read_csv(data)
        Ads = df['Ads_X']
        Edges_Ads = []
        Position_edges = []
        Nodes_Ads = []
        for string in Ads:
            traj = string + '.traj'
            nodes, nodes_index, edges, edges_index, edges_position, cell = GE.get_graph_input(traj)
            Nodes_Ads.append(nodes)
            Edges_Ads.append(edges)
            Position_edges.append(edges_position)
        
        Nodes_Ads_copy = deepcopy(Nodes_Ads)  ### important
        ordered_node, ordered_edge = CM_descriptors.get_order(self,Edges_Ads,Nodes_Ads_copy)
        return Nodes_Ads, Edges_Ads, ordered_node, ordered_edge, Position_edges, cell
        
    def get_order(self,Edges_Ads,Nodes_Ads):
        ordered_edge = []
        length = [len(kk) for kk in Edges_Ads]
        maxlen_index = length.index(max(length))
        ordered_edge = ordered_edge + Edges_Ads[maxlen_index]
        for edges in Edges_Ads:
            for edge in edges:
                if edge in ordered_edge:
                    n = edges.count(edge)
                    N = ordered_edge.count(edge)
                    if N < n:
                        ordered_edge.append(edge)
                else:
                    ordered_edge.append(edge)
        
        ordered_edge = sorted(ordered_edge)  ###
        print ("Ordered edges:", ordered_edge)
        print ("Length of Ordered edges:", len(ordered_edge))
        
        length = [len(kk) for kk in Nodes_Ads]
        maxlen_index = length.index(max(length))
        ordered_node = Nodes_Ads[maxlen_index]

        for nodes in Nodes_Ads:
            for node in nodes:
                if node in ordered_node:
                    n = nodes.count(node)
                    N = ordered_node.count(node)
                    if N < n:
                        ordered_node.append(node)
                else:
                    ordered_node.append(node)
        
        ordered_node = sorted(ordered_node) ### important
        print ("Ordered nodes:", ordered_node)
        print ("Length of Ordered nodes:", len(ordered_node))
        return ordered_node, ordered_edge

    def CM_nondiagnal(self,edge,r_ij,atomic_number):
        Z_i = atomic_number[edge[0]]
        Z_j = atomic_number[edge[1]]
        fp = Z_i * Z_j / r_ij
        return fp
        
    def CM_diagnal(self,node,atomic_number):
        Z_i = atomic_number[node]
        fp = 0.5 * Z_i ** 2.4
        return fp
        
    def Rij_distance(self,position_edge):
        r_ij = np.linalg.norm(position_edge[0] - position_edge[1])
        return r_ij
        
    def Atomic_number(self):
        atomic_number = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'P':15, 'S':16, 'Cl':17}
        return atomic_number
        
    def e_k(self):
        ek = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
        return ek
    def cell_inverse(self, cell):
        inverse_cell = np.linalg.inv(cell)
        return inverse_cell

    def Ri_Rj_vec(self,position_edge):
        v_ij = position_edge[0] - position_edge[1]
        return v_ij
        
    def SinCM_nondiagnal(self,edge,cell,inverse_cell,ek,v_ij,atomic_number):
        Z_i = atomic_number[edge[0]]
        Z_j = atomic_number[edge[1]]
        x_y_z = []
        for ek_i in ek:
            in_sin = np.pi * np.dot(np.dot(ek_i, inverse_cell), v_ij)
            x_y_z.append(ek_i * np.sin(in_sin)**2)
        #print (x_y_z)
        new_xyz = np.array(x_y_z[0]) + np.array(x_y_z[1]) + np.array(x_y_z[2])
        F = np.linalg.norm(cell*new_xyz)
        fp = Z_i * Z_j / F
        return fp     
        
    def get_atom_connectivity_CM(self,df):
        """ 
        custom_CM_fpts = 'custom_CM_fpts.csv'
        df = pd.read_csv(custom_CM_fpts) 
        """
        columns = df.columns.get_values()
        unique_columns = list(set(columns))
        Cols = []
        for col in unique_columns:
            lists = []
            for k in range(len(columns)):
                if col == columns[k]:
                    lists.append(k)
            Cols.append(lists)
        
        #new_columns = list(range(len(columns)))
        #dff = df.rename(columns=dict(zip(columns,new_columns)))
        
        DF = {}
        for k, col in enumerate(unique_columns):
            lists = Cols[k]
            DF[col] = df.iloc[:,lists].sum(axis=1)  ###
            
        data = pd.DataFrame(DF)
        data.to_csv('AC_CM.csv',index=False)
        return data