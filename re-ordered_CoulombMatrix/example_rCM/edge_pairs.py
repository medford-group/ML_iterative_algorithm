#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:45:36 2018

@author: fuzhu
"""

#df[(df['A']=='H') & (df['B']=='O') & (df['S1']=='t')].index.tolist()[0]


from ase.io import read
from itertools import permutations

class generate_edges:
    def get_graph_input(self,traj):            
        atoms = read(traj)
        Positions = generate_edges.Attr_positions(self,atoms)
        nodes, nodes_index = generate_edges.Nodes(self,atoms)
        edges, edges_index, edges_position = generate_edges.Edges(self,nodes,Positions)
        
        cell = generate_edges.get_cell(self,atoms)
        # nodes: [element0,element1,element2 ...]
        # nodes_index: [0, 1, 2, 3, ...]
        # edges: [(C,O),(C,H), ...]
        # edges_index: [(0,1), (0,2)]
        return nodes, nodes_index, edges, edges_index, edges_position, cell
    
    def Edges(self,nodes,Positions):
        edges = []
        edges_index = []
        edges_position = []
        positions = tuple(Positions)
        N = len(positions)
        for position_pair in permutations(range(N), r=2):
            if sorted(position_pair) == list(position_pair):
                pos_pair = tuple([nodes[i] for i in position_pair]) # convert numer index to element
                edges.append(pos_pair)
                edges_index.append(position_pair)
                edge_position = tuple([positions[i] for i in position_pair])  # covert number index to xyz position
                edges_position.append(edge_position)
        return edges,edges_index, edges_position
                                 
    def Attr_positions(self,atoms):
        Positions = atoms.positions   # position of each atom
        return Positions
    
    def Nodes(self,atoms):
        nodes = atoms.get_chemical_symbols()  # elemetns
        nodes_index = range(len(nodes))   # index each element
        return nodes, nodes_index
    
    def get_cell(self, atoms):
        cell = atoms.get_cell()
        return cell