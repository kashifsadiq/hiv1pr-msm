MOLECULAR DYNAMICS MARKOV STATE MODEL AND CONFORMATIONAL GATING FACTOR ANALYSIS FOR HIV-1 PROTEASE

Author: Dr. S. Kashif Sadiq
Affiliation: 1. Heidelberg Institute for Theoretical Studies, HITS gGmbH 2. European Moelcular Biology Laboratory
Correspondence: kashif.sadiq@embl.de 

Last modified: 18.06.2021

Manuscript:  S. Kashif Sadiq, Abraham Muñiz Chicharro, Patrick Friedrich, Rebecca Wade, A multiscale approach for computing gated ligand binding from molecular dynamics and Brownian dynamics simulations. (2021)

####################################################################################################################################

features  figures  MSM_obj  notebook  README.txt  reference  scripts  traj  volmap

The current directory contains a number of analysis pipelines for reproducing the MD conformational analysis, MSM and gating factor calculation:

notebook - 
hivpr-conf-gating.ipynb: Python Jupyter notebook for performing analysis. Load this to work through the analysis pipeline.
hivprgaing.py: Python module of analysis functions for use with the Jupyter notebook

traj -
location of all-atom MD simulation trajectories in xtc format (stripped of water and ions)
	- macro_bestmicro_lamsort1000_xtcs sub-directory contains trajectories of representative conformations of each macrostate. 

reference - contains pdb structure and topology file from initial build. 
	- p2nc subdirectory contains pdb, topology and representative trajectory of HIV-protease in closed conformation with bound substrate. See:
	S.K. Sadiq‡ (2020) Catalysts, Fine-tuning of sequence-specificity by near attack conformations in enzyme-catalyzed peptide hydrolysis, 10 (6) 684.
	Sadiq, S.K. ‡ and Coveney P.V. (2015). J Chem Theor Comput. Computing the role of near attack conformations in an enzyme-catalyzed nucleophilic bimolecular reaction. 11 (1), pp 316–324
	All structures, tooplogies and trajectories are stripped of water and ions. 

features - contains data sets of various collective variables that can be used for conformational analysis of building MSMs
	- lambda: contains the 3D lambda metric data (used to build Markov State model)

MSM_obj - contains pyEMMA objects for kmeans clustering, implied timescales, MSM, and HMM for exact data comparison with that reported in mannuscript. 

figures - conf_figs sub directory contains representative image of each macrostate conformation from superimposed snapshots (png format)

scripts - contains a variety of bash scripts that execute vmd tcl scripts to:
	a) reproduce feature data, including performing minimum fluctuation alignment
	b) perform volumetric mapping (volmap)
	Executing ./run_analyses.sh from within this directory will run both sets of analyses

volmap - contains output grid data for the volmap analysis

####################################################################################################################################
