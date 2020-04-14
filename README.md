#### ML_iterative_algorithm

This repository includes the python scripts to generate the surface/molecule descriptors and to run the developed iterative ML algorithms. Each folder has the main codes and an example.

##### re-ordered Coulomb Matrix

1. Prepare the molecule structures. Here we recommend to use the [ASE](https://wiki.fysik.dtu.dk/ase/) to build the structure with the *.traj* format, other formats also work, such as *.cif*, *.xyz*.
2. See the example to get the reordered Coulomb Matrix.  ($python run_test.py ). The descriptors will be saved in a *.csv* file.

##### Extract the DOS of active site

The *extractDos.py* file is in the path of */data/scripts_py.zip*, this file is used to extract the DOS of active site. 

1. DFT calculated DOS was saved in the *.pickle* file.
2. The site name and atom index on clean surface are required. ($python run_get_data.py) The descriptors are also saved in a *.csv* file.

##### Iterative algorithm for looking for the valuable training set

The *Iteration\*.py*  files are in the folder of *Iteration_I_AdsorptionE* , all these scripts work for training process. We also show an example in this folder. ($python run_test.py).

1. Max predicted error: a parameter to control the accuracy and number of DFT calculations. 

   <div align=center>![ML vs. DFT] (Iteration_I_AdsorptionE/example/ML_algorithm_1.png)

##### Iterative algorithm for finding the low-energy structure

The example is constructing the phase diagrams including ML phase diagrams and DFT phase diagrams. Please see the folder of *Iteration_II_PhaseDiagram*. Each folder have the main scripts and an example. The algorithm can construct the N dimensional phase diagrams (1<= N <=4).

1. *atomType_chemPot* : a dictionary to specify the atomic type and the range of chemical potential.
2. *N_int* : Resolution of phase diagram
3. *$python run_pd.py*



###### Note

The *data* folder includes the structure file (*.cif*) and python script files.

###### Email

More questions: [lfzxsx@126.com](lfzxsx@126.com)









