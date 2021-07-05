# HIV-1 protease Markov State Model Conformational Gating Analysis

#Author: S. Kashif Sadiq

#Correspondence: kashif.sadiq@embl.de, Affiliation: 1. Heidelberg Institute for Theoretical Studies, HITS gGmbH 2. European Moelcular Biology Laboratory

#This module contains core functions for molecular dynamics (MD) simulation and Markov state model analyses of apo HIV-1 protease conformational gating for the manuscript:

#S.Kashif. Sadiq‡, Abraham Muñiz Chicharro, Patrick Friedrich, Rebecca Wade (2021)   A multiscale approach for computing gated ligand binding from molecular dynamics and Brownian dynamics simulations

########################################################################################################################################

from __future__ import print_function
import warnings
import pyemma
import os
#%pylab inline
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt


from pyemma import config
config.show_progress_bars = False
#print(config.show_progress_bars)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.image as mpimg

from collections import OrderedDict
import math
import numpy as np
import sys
import os.path
import random
import errno 
from shutil import copyfile
import operator
import re
from glob import glob
#from kmodes.kmodes import KModes
import random
import MDAnalysis
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import align, rms, distances, contacts
from MDAnalysis.analysis.base import AnalysisFromFunction 
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import density
#import MDAnalysis.analysis.hbonds
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA


mpl.rcParams.update({'font.size': 12})

print('pyEMMA version: '+ pyemma.__version__)
print('MDAnalysis version: ' + MDAnalysis.version.__version__)


from sklearn.neighbors import KernelDensity
from matplotlib import gridspec
from scipy.stats import norm




########################################################################################################################################

##########################################################################################################
#
# FUNCTIONS
#
##########################################################################################################


##########################################################################################################

#################
#pyEMMA standard Functions
#################

#################
def save_figure(name):
    # change these if wanted
    do_save = True
    fig_dir = './figs/'
    if do_save:
        savefig(fig_dir + name, bbox_inches='tight')

#################
def plot_sampled_function(ax_num, xall, yall, zall,  dim, msm_dims, ticks_set, labels, ax=None, nbins=100, nlevels=20, cmap=cm.bwr, cbar=True, cbar_label=None):
    # histogram data
    xmin = np.min(xall)
    xmax = np.max(xall)
    dx = (xmax - xmin) / float(nbins)
    ymin = np.min(yall)
    ymax = np.max(yall)
    dy = (ymax - ymin) / float(nbins)
    # bin data
    #eps = x
    xbins = np.linspace(xmin - 0.5*dx, xmax + 0.5*dx, num=nbins)
    ybins = np.linspace(ymin - 0.5*dy, ymax + 0.5*dy, num=nbins)
    xI = np.digitize(xall, xbins)
    yI = np.digitize(yall, ybins)
    # result
    z = np.zeros((nbins, nbins))
    N = np.zeros((nbins, nbins))
    # average over bins
    for t in range(len(xall)):
        z[xI[t], yI[t]] += zall[t]
        N[xI[t], yI[t]] += 1.0
    with warnings.catch_warnings() as cm:
        warnings.simplefilter('ignore')
        z /= N
    # do a contour plot
    extent = [xmin, xmax, ymin, ymax]
    if ax is None:
        ax = gca()
    s = ax.contourf(z.T, 100, extent=extent, cmap=cmap)
    if cbar:
        cbar = fig.colorbar(s)
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label)
    
    
    ax.set_xlim(xbins.min()-5,xbins.max()+5)
    ax.set_xticks(ticks_set[np.where(msm_dims==dim[0])[0][0]])
    ax.set_xlabel(labels[dim[0]],fontsize=10)
    
    ax.set_ylim(ybins.min()-5,ybins.max()+5)
    ax.set_yticks(ticks_set[np.where(msm_dims==dim[1])[0][0]])
    
    if ax_num==0:
        ax.set_ylabel(labels[dim[1]],fontsize=10)

    
    return ax

#################
def plot_sampled_density(ax_num, xall, yall, zall, dim, msm_dims, ticks_set, labels, ax=None, nbins=100, cmap=cm.Blues, cbar=True, cbar_label=None):
    return plot_sampled_function(ax_num, xall, yall, zall, dim, msm_dims, ticks_set, labels, ax=ax, nbins=nbins, cmap=cmap, cbar=cbar, cbar_label=cbar_label)

##########################################################################################################

#################
#pyEMMA MSM functions
#################

#################
def eval_transformer(trans_obj):
    # Effective dimension (Really? If we just underestimate the Eigenvalues this value also shrinks...))
    print('Evaluating transformer: ', str(trans_obj.__class__))
    print('effective dimension', np.sum(1.0 - trans_obj.cumvar))
    print('eigenvalues', trans_obj.eigenvalues[:5])
    print('partial eigensum', np.sum(trans_obj.eigenvalues[:10]))
    print('total variance', np.sum(trans_obj.eigenvalues ** 2))
    print()

#################    
def project_and_cluster(trajfiles, featurizer, sparsify=False, tica=True, lag=100, scale=True, var_cutoff=0.95, ncluster=100):
    """
    Returns
    -------
    trans_obj, Y, clustering

    """
    X = coor.load(trajfiles, featurizer)
    if sparsify:
        X = remove_constant(X)
    if tica:
        trans_obj = coor.tica(X, lag=lag, var_cutoff=var_cutoff)
    else:
        trans_obj = coor.pca(X, dim=-1, var_cutoff=var_cutoff)
    Y = trans_obj.get_output()
    if scale:
        for y in Y:
            y *= trans_obj.eigenvalues[:trans_obj.dimension()]
    cl_obj = coor.cluster_kmeans(Y, k=ncluster, max_iter=3, fixed_seed=True)
    return trans_obj, Y, cl_obj    


##########################################################################################################

#################
#File reading functions
#################
def read_int_matrix(fname):
    """
    reads a file containing a matrix of integer numbers
    """
    a = []
    with open(fname) as f:
        for line in f:
            row = line.rstrip().split()
            a.append(row)
    foo = np.array(a)
    bar = foo.astype(np.int)
    return bar

#################
#Read in matrix of floats from file
def read_float_matrix(fname):
    """
    reads a file containing a matrix of floating point numbers
    """
    a = []
    with open(fname) as f:
        for line in f:
            row = line.rstrip().split()
            a.append(row)
    foo = np.array(a)
    bar = foo.astype(np.float)
    return bar


def READ_INITIAL_FILE ( filename ):
    # read in group data into lists of lists
    file = open(filename,'r')
    coords=[]
    for line in file:
        vals=line.split()
        vals2 = [float(numeric_string) for numeric_string in vals[3:6]]        
        coords.append(vals2)
        
    return coords;

##########################################################################################################

#################
#Trajectory Processing Functions
#################
# This sorts the list of trajectories in double numerical order e.g. 1-1.dcd
def sorted_traj_list(traj_list):
    s=[]
    for i in range(len(traj_list)):
        string = traj_list[i]
        s.append([int(n) for n in re.findall(r'\d+\d*', string)])
    s = sorted(s, key = operator.itemgetter(0, 1))
    
    sorted_traj_list = []
    for i in range(len(s)):
        sorted_traj_list.append(indir+'/'+str(s[i][0])+'-'+str(s[i][1])+'.dcd')

    return(sorted_traj_list)

#################
#Creates a trajectory list from an array that contains the format: batch sims frames
def traj_list_from_sims_array(sims_array, indir):
    traj_list = []
    for i in range(len(sims_array)):
        traj_list.append(indir+'/'+str(sims_array[i][0])+'-'+str(sims_array[i][1])+'.dcd')
    return traj_list

#################
#Creates a trajectory list from an array that contains the format: batch sims frames
def traj_list_from_sims_array_xtc(sims_array, indir):
    traj_list = []
    for i in range(len(sims_array)):
        traj_list.append(indir+'/'+str(sims_array[i][1])+'-filtered.xtc')
    return traj_list


#################
#Select only those trajectories from an trajlist/array that have >= than a certain threshold of frames
def thresh_subset(sims_array,thresh):
    frames_thresh=np.empty((0,3))
    for i in range(len(sims_array)):
        if sims_array[i][2]>=thresh:
            frames_thresh=np.vstack((frames_thresh,sims_array[i]))
    f=frames_thresh.astype(np.int)
    return f


def predefined_simsarray(full_sims_array):
    """
    #creates a subarray from a predefined sim list of batch and sim numbers and a complete sims array 
    # this is for testing a limited number of sims e.g. if copied to local resources
    """        
    simlist=[[1,1],[2,1],[3,1],[4,1],[5,1],[6,1]]
    sublist = []
    for i in range(len(simlist)):
        sublist = sublist + [x for x in full_sims_array.tolist() if x[0]==simlist[i][0] and x[1]==simlist[i][1]]
    subarray=np.array(sublist)
    return subarray

##########################################################################################################

#################
#Functions for calculating continuous minimum nearest neighbour contact
#################

#################
#Minimum Mean Continuous minimum distance across sliding window tau
def cmindist(data, tau):
    """
    computes continuous minimum distance of data array as the minimum of the mean sliding window of length tau
    """
    tscan=np.shape(data)[0]-tau+1
    num_feat=np.shape(data)[1]
    cmd=np.empty((0,num_feat))
    for i in range(tscan):
        cmd=np.vstack((cmd,np.mean(data[i:i+tau,:],axis=0)))
    return np.min(cmd,axis=0)

#################
#Mean Continuous minimum distance across sliding window tau
def taumean_mindist(data, tau):
    """
    computes continuous minimum distance of data array as the mean sliding window of length tau
    """
    tscan=np.shape(data)[0]-tau+1
    num_feat=np.shape(data)[1]
    cmd=np.empty((0,num_feat))
    for i in range(tscan):
        cmd=np.vstack((cmd,np.mean(data[i:i+tau,:],axis=0)))
    return cmd

#################
#Longest continuous time of minimum distance
def long_mindist(data, thresh):
    """
    computes the longest time the minimum distance stays within a threshhold of thresh
    """
    tscan=np.shape(data)[0]
    num_feat=np.shape(data)[1]
    count=np.empty(num_feat)
    lmd=np.empty(num_feat)
    for i in range(tscan):
        for j in range(num_feat):
            if data[i,j] < thresh:
                count[j] += 1
            else:
                if count[j] > lmd[j]:
                    lmd[j] = count[j]
                count[j] = 0
                
    return lmd.astype(np.int)

#################
#Determine res-res pairs included for which to calculate minimum distance features
def res_pairs(num_res, nearn):
    """
    computes res-res pairs included for which to calculate minimum distance features
    state num of residues, and nearest neighbour skip e.g. i+3 is nearn=3
    """
    res=[]
    for i in range(num_res-nearn):
        for j in range(i+nearn,num_res):
            res.append([i+1,j+1])
    return res

#################
#Calculate longest duration of minimum distance below a threshold of each res-res pair across traj ensemble
def ensemble_maxdur(traj_arr, col_exc, res, tau, thresh):
    """
    computes longest duration of minimum distance below a threshold of each res-res pair across traj ensemble
    using: list of traj nums -traj_array, res-pair list - res, sliding mean smoothing - tau, mindist threshold - thresh 
    col_exc is the number of colums in data file specified by traj_arr to exclude - normally col_exc=3
    """
    lmd=np.empty((0,len(res)))
    for i in range(len(traj_arr)):
        fname = './analysis/resres_mindist/'+str(traj_arr[i,0])+'-'+str(traj_arr[i,1])+'.dat'
        mindist = read_float_matrix(fname)
        mindist=mindist[:,col_exc:]
        #cmd=cmindist(mindist,tau)
        if tau>1:
            taumd=taumean_mindist(mindist,tau)
        else:
            taumd=mindist
        lmd=np.vstack((lmd,long_mindist(taumd, thresh)))
        print("Batch: "+str(traj_arr[i,0])+", Sim: "+str(traj_arr[i,1]))
    #return np.max(lmd.astype(np.int),axis=0)
    return lmd.astype(np.int)

#################
#Continuous minimum nearest neighbour contact calculation
def mindist_contacts(res_start, res_end, tau_c):
    #Number of residues
    num_res=23
    #Next nearest neighbour - e.g. i+3
    nearn=3
    #List of i!=j res-res number pairs with i:i+3
    res=res_pairs(num_res,nearn)
    #Maximum duration each above res-res contact is formed in each traj
    #In reality this is done once on the server and saved as a file as time consuming
    #Number of columns to exclude in data files
    #col_exc=3
    #window length for calculating sliding mean minimum distance 
    #tau=10
    #Threshold distance in Angstrom
    #thresh=4.0
    #ens_max_dur=ensemble_maxdur(sims_array, col_exc, res, tau, thresh)
    #np.savetxt('ens_max_dur.dat', ens_max_dur, fmt='%1d',delimiter=' ')
    fname = './ens_max_dur.dat'
    ens_max_dur = read_int_matrix(fname)#Collapse all trajectories into 1 row showing maximum of each res-res pair
    max_dur=np.max(ens_max_dur,axis=0)
    
    #List of res-res contacts that fulfil tau_c - res labelling starting from 1
    contacts_list=[res[x] for x in range(len(res)) if max_dur[x]>=tau_c]
    contacts=np.array(contacts_list)
    
    contacts=contacts[contacts[:,0]>=res_start]
    contacts=contacts[contacts[:,1]<=res_end]
    
    #Con0 is relabeling residue pairs starting from 0
    con0_list=[[x[0]-1, x[1]-1] for x in contacts.tolist()]
    con0=np.array(con0_list)
    
    #Theoretical maximum size of res list for given residue range
    num_res_select = res_end - res_start + 1
    res_select=res_pairs(num_res_select,nearn)
    max_res_select = len(res_select)
    
    return con0, res, max_res_select, max_dur



##########################################################################################################

#################
#Feature Data Loading Functions
#################

#################
#Lambda coordinate space
def lambda_obj(lamdir,sims_array,num_frames=None):
    """
    # loads values from lambda space for HIV-1 PR into lambda_obj
    """ 
    coords=[]
    for i in range(len(sims_array)):
        filename=lamdir + '/' + str(sims_array[i][0])+'-'+str(sims_array[i][1]) + '.dat'
        if os.path.isfile(filename):
            tmpcoords=read_float_matrix(filename)
            if num_frames==None:
                coords.append(tmpcoords[:,3:6])
            else:
                coords.append(tmpcoords[0:num_frames,3:6])
    return coords

#################
#Multidimenstional metric files coordinate space
def multidir_obj(dir_array,sims_array,num_frames=None):
    """
    # loads values from lambda space and other metrics for HIV-1 PR into lambda_obj
    """ 
    coords=[]
    for i in range(len(sims_array)):
        #Make a list for each correspoinding file in different directories
        filename=[dir_array[x] + '/' + str(sims_array[i][0])+'-'+str(sims_array[i][1]) + '.dat' for x in range(len(dir_array))]

        #Check that same files exist across all deisgnated directories
        if np.sum([os.path.isfile(filename[x]) for x in range(len(dir_array))])==len(dir_array):
            
            tmpcoords=read_float_matrix(filename[0])
            tmpcoords=tmpcoords[:,3:]
            for i in range(1,len(dir_array)):
                tmpcoords_i=read_float_matrix(filename[i])
                tmpcoords_i=tmpcoords_i[:,3:]
                tmpcoords=np.hstack((tmpcoords,tmpcoords_i))
            
            if num_frames==None:
                coords.append(tmpcoords)
            else:
                coords.append(tmpcoords[0:num_frames,:])
                
    return coords

##########################################################################################################

#################
#Coordinate Transformation Functions
#################

#################
def xyz_to_cyl_coords(data,th_offset=0):
    x,y = data[:,0], data[:,1]
    rho = np.sqrt(x**2+y**2)
    theta = 180*np.arctan2(y,x)/np.pi - th_offset
    theta %= 360
    z=data[:,2]
    
    return np.transpose(np.vstack((rho,theta,z)))


##########################################################################################################

#################
#Plotting functions
#################

#################
def plot_frames(plt,num_frames):
    """
    # Plot number of frames for each sim
    """
    fig, axs = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    ax=plt.axes()
    plt.xticks(np.arange(0, len(num_frames), 100))
    plt.yticks(np.arange(0, 2000, 100))
    ax.set_xlim(0,len(num_frames))
    ax.set_ylim(0,2000)
    x=np.array(range(len(num_frames)))
    y=num_frames
    p1 = ax.plot(x, y,'k-o')
    plt.show()
    
    return

#################
def plot_dropoff(plt,sorted_frames):
    """
    #Plot Drop off of trajectories with increasing number of frames
    """
    fig, axs = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    ax=plt.axes()
    plt.xticks(np.arange(0, 9000, 1000))
    plt.yticks(np.arange(0, 700, 100))
    ax.set_xlim(0,9000)
    ax.set_ylim(0,600)
    plt.xlabel('Number of frames')
    plt.ylabel('Number of trajectories')
    
    x = sorted_frames
    y = np.arange(sorted_frames.size)
    #p1 = ax.step(x, y,'k-')
    p2 = ax.step(x[::-1], y,'r-')
    plt.show()    



#################
def plot_minmax_coverage(plt,min_req_frames,sorted_frames,min_coverage,max_coverage):
    """
    #Plot minimum and maximum coverage based on a minimum required number of frames/traj
    """    
    fig, axs = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    ax=plt.axes()
    plt.xticks(np.arange(0, 9000, 1000))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(0,9000)
    ax.set_ylim(0,1)
    plt.xlabel('Used traj length (frames)')
    plt.ylabel('Number of frames used')
    
    x = min_req_frames
    y = min_coverage/sum(sorted_frames)
    y2 = max_coverage/sum(sorted_frames)
    
    p1 = ax.step(x, y,'k-')
    p2 = ax.step(x, y2,'b-')
    plt.show()    
    
    return

#################
def trajplot_format(plt,tlim,ylims,ylab="Distance (\AA)"):
    plt.rc('text', usetex=True)
    plt.xlim(0,(tlim+1)/10)
    if ylims[0]>=0:
        plt.ylim(0,ylims[1]+1)
    else:
        plt.ylim(ylims[0],ylims[1]+1)    
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    
    plt.xlabel(r"Time (ns)", fontsize=30)   
    plt.ylabel(ylab, fontsize=30)   
    
    return

#################
def plot_traj(Y,sims_array,traj_no,dims,colors=['b','k','r','y'],md_numbering=True):

    tlim=np.shape(Y)[1]
    if md_numbering is True: 
        traj_id=np.where(sims_array[:,1]==traj_no)[0][0]
    else:
        traj_id=traj_no
        
    ydat=np.array([j for m in [Y[traj_id][:tlim,dims[i]] for i in range(len(dims))] for j in m])
    ylims=[ydat.min(),ydat.max()]

    for i in range(len(dims)):
        plt.plot([x/10 for x in range(1,tlim+1)], Y[traj_id][:tlim,dims[i]], '-', color=colors[i])

    trajplot_format(plt,tlim,ylims)

    return

#################
def plot_traj_from_Z(Z,nsnaps,sims_array,traj_no,dims,colors=['b','k','r','y'],md_numbering=True):

    tlim=nsnaps
    if md_numbering is True: 
        traj_id=np.where(sims_array[:,1]==traj_no)[0][0]
    else:
        traj_id=traj_no
    
    traj_start_ind=traj_id*nsnaps
    ydat=np.array([j for m in [Z[traj_start_ind:traj_start_ind+tlim,dims[i]] for i in range(len(dims))] for j in m])
    ylims=[ydat.min(),ydat.max()]

    for i in range(len(dims)):
        plt.plot([x/10 for x in range(1,tlim+1)], Z[traj_start_ind:traj_start_ind+tlim,dims[i]], '-', color=colors[i])

    trajplot_format(plt,tlim,ylims)

    return

#################
def plot_traj_from_MD(data,nsnaps,dims,colors=['b','k','r','y']):

    tlim=nsnaps
    ydat=np.array([j for m in [data[:tlim,dims[i]] for i in range(len(dims))] for j in m])
    ylims=[ydat.min(),ydat.max()]

    for i in range(len(dims)):
        plt.plot([x/10 for x in range(1,tlim+1)], data[:tlim,dims[i]], '-', color=colors[i])

    trajplot_format(plt,tlim,ylims)

    return

#################
def plot_free_energy_landscape(Z,plt,xdim,ydim,labels,cmap="jet",fill=True, contour_label=True,contour_color='k',wg=None):
    
    #x=np.vstack(Y)[:,0]
    #y=np.vstack(Y)[:,2]
    x=Z[:,xdim]
    y=Z[:,ydim]
    rho,xbins,ybins = np.histogram2d(x,y,bins=[100,100],weights=wg)
    kBT=0.596
    G=-kBT*np.log(rho+0.1)
    Gzero=G-np.min(G)
    fig, ax = plt.subplots(figsize=(12,9))
    ex=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    lev=[x /10.0 for x in range(int(5*round(np.min(G)*2))-10,0,5)]
    contours=plt.contour(np.transpose(G), extent=ex, levels = lev, colors=contour_color,linestyles= '-' )
    
    if fill is True:
        plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    
    if contour_label is True:
        plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    
    
    #plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    #plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    cbar = plt.colorbar()
    #plt.clim(np.min(G)-0.5,np.max(G)+0.5)
    plt.clim(-10,0)
    cbar.set_label(r"G (kcal/mol)", rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=30)
    plt.rc('text', usetex=True)
    plt.xlim(xbins.min()-5,xbins.max())
    plt.ylim(ybins.min()-5,ybins.max())
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    
         
    plt.xlabel(labels[xdim], fontsize=30)
    plt.ylabel(labels[ydim], fontsize=30)
    
    return plt

#################
def plot_free_energy_landscape_nocbar(Z,plt,xdim,ydim,labels,cmap="jet",fill=True, contour_label=True,contour_color='k',wg=None):
    
    #x=np.vstack(Y)[:,0]
    #y=np.vstack(Y)[:,2]
    x=Z[:,xdim]
    y=Z[:,ydim]
    rho,xbins,ybins = np.histogram2d(x,y,bins=[100,100],weights=wg)
    kBT=0.596
    G=-kBT*np.log(rho+0.1)
    Gzero=G-np.min(G)
    #fig, ax = plt.subplots(figsize=(9,9))
    ex=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    lev=[x /10.0 for x in range(int(5*round(np.min(G)*2))-10,0,5)]
    contours=plt.contour(np.transpose(G), extent=ex, levels = lev, colors=contour_color,linestyles= '-' )
    
    
    if fill is True:
        plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    
    if contour_label is True:
        plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    
    
    #plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    #plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    #cbar = plt.colorbar()
    #plt.clim(np.min(G)-0.5,np.max(G)+0.5)
    plt.clim(-10,30)
    #cbar.set_label(r"G (kcal/mol)", rotation=90, fontsize=30)
    #cbar.ax.tick_params(labelsize=30)
    plt.rc('text', usetex=True)
    plt.xlim(xbins.min()-5,xbins.max())
    plt.ylim(ybins.min()-5,ybins.max())
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    
    plt.xlabel(labels[xdim], fontsize=30)
    plt.ylabel(labels[ydim], fontsize=30)
    
    return plt


#################
def plot_free_energy_landscape_nocbar_array(Z,plt,xdim,ydim,labels,cmap="jet",
                                            fill=False, contour_label=False,contour_color='k',
                                            wg=None,show_ticks=False,show_labels=False):
    
    #x=np.vstack(Y)[:,0]
    #y=np.vstack(Y)[:,2]
    x=Z[:,xdim]
    y=Z[:,ydim]
    rho,xbins,ybins = np.histogram2d(x,y,bins=[100,100],weights=wg)
    kBT=0.596
    G=-kBT*np.log(rho+0.1)
    Gzero=G-np.min(G)
    #fig, ax = plt.subplots(figsize=(9,9))
    ex=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    lev=[x /10.0 for x in range(int(5*round(np.min(G)*2))-10,0,5)]
    contours=plt.contour(np.transpose(G), extent=ex, levels = lev, colors=contour_color,linestyles= '-' )
    
    
    if fill is True:
        plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    
    if contour_label is True:
        plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    
    
    #plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    #plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    #cbar = plt.colorbar()
    #plt.clim(np.min(G)-0.5,np.max(G)+0.5)
    plt.clim(-10,30)
    #cbar.set_label(r"G (kcal/mol)", rotation=90, fontsize=30)
    #cbar.ax.tick_params(labelsize=30)
    plt.rc('text', usetex=True)
    plt.xlim(xbins.min()-5,xbins.max())
    plt.ylim(ybins.min()-5,ybins.max())
    
    if show_ticks:
        plt.xticks(fontsize=30, rotation=0)
        plt.yticks(fontsize=30, rotation=0)
    else:
        plt.xticks([])
        plt.yticks([])
    
    if show_labels:
        plt.xlabel(labels[xdim], fontsize=30)
        plt.ylabel(labels[ydim], fontsize=30)
    
    return plt


#################
def plot_weighted_free_energy_landscape(Z,plt,xdim,ydim,labels, cmap="jet", fill=True, contour_label=True, contour_color='k', clim=[-10,0],cbar=False, cbar_label="G (kcal/mol)",lev_max=-1,shallow=False,wg=None,fsize_cbar=(12,9),fsize=(9,9),standalone=True):
 
    if standalone:
        if cbar:
            fig, ax = plt.subplots(figsize=fsize_cbar)
        else:
            fig, ax = plt.subplots(figsize=fsize)
    

    #x=np.vstack(Y)[:,0]
    #y=np.vstack(Y)[:,2]
    x=Z[:,xdim]
    y=Z[:,ydim]
    rho,xbins,ybins = np.histogram2d(x,y,bins=[100,100],weights=wg)
    rho += 0.1
    kBT=0.596
    G=-kBT*np.log(rho/np.sum(rho))
    G=G-np.max(G)
    ex=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    lev=[x /10.0 for x in range(int(5*round(np.min(G)*2))-10,int(lev_max*10),5)]
    if shallow is True:
        lev_shallow=[-0.4,-0.3,-0.2,-0.1]
        lev+=lev_shallow    
    contours=plt.contour(np.transpose(G), extent=ex, levels = lev, colors=contour_color, linestyles= '-' )
    
    if fill is True:
        plt.contourf(np.transpose(G), extent=ex,cmap = cmap, levels = lev)
    
    if contour_label is True:
        plt.clabel(contours, inline=True, fmt='%1.1f', fontsize=20)
    
    plt.clim(clim[0],clim[1])
    
    plt.rc('text', usetex=True)
        
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=30)
            cbar.ax.tick_params(labelsize=30)    
    
    plt.xlim(xbins.min()-5,xbins.max()+5)
    plt.ylim(ybins.min()-5,ybins.max()+5)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)         
    plt.xlabel(labels[xdim], fontsize=30)
    plt.ylabel(labels[ydim], fontsize=30)
    
    return plt


#################
def annotate_microstates(plt,sets,cl_x,cl_y,tsize=12):
    i=0
    for xy in zip(cl_x,cl_y): 
        plt.annotate('  %s' % sets[i], xy=xy, textcoords='data',
                     size=tsize,weight='bold',color='black', fontname='Courier'
                     )
        #arrowprops=dict(edgecolor='red',facecolor='red', shrink=0.02,width=1,headwidth=5)
        #,edgecolor='red',facecolor='red', shrink=0.05,width=2
        #arrowstyle="->",edgecolor='white',facecolor='white'
        i+=1
    return plt



#################
def plot_metastable_sets(plt,cl_obj,meta_sets,MSM_dims,dim,mstate_color,msize=10,annotate=False,textsize=18):
    for k in range(len(meta_sets)):
        cl_x=cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[0])[0][0]]
        cl_y=cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[1])[0][0]]
        plt.plot(cl_x,cl_y, linewidth=0, marker='o', markersize=msize, markeredgecolor=mstate_color[k],markerfacecolor=mstate_color[k], markeredgewidth=2)
        #plt.plot(cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[0])[0][0]],cl_obj.clustercenters[meta_sets[k],np.where(MSM_dims==dim[1])[0][0]], linewidth=0, marker='o', markersize=msize, markeredgecolor=mstate_color[k],markerfacecolor=mstate_color[k], markeredgewidth=2)

        if annotate is True:
            plt=annotate_microstates(plt,meta_sets[k],cl_x,cl_y,tsize=textsize)
        
    return 


#################
def plot_projected_density(Z, zall, plt, xdim, ydim, labels, nbins=100, nlevels=20, cmap=cm.bwr, cbar=False, cbar_label=None):
    
    if cbar:
        fig, ax = plt.subplots(figsize=(12,9))
    else:
        fig, ax = plt.subplots(figsize=(9,9))
    
    xall=Z[:,xdim]
    yall=Z[:,ydim]
    
    # histogram data
    xmin = np.min(xall)
    xmax = np.max(xall)
    dx = (xmax - xmin) / float(nbins)
    ymin = np.min(yall)
    ymax = np.max(yall)
    dy = (ymax - ymin) / float(nbins)
    # bin data
    #eps = x
    xbins = np.linspace(xmin - 0.5*dx, xmax + 0.5*dx, num=nbins)
    ybins = np.linspace(ymin - 0.5*dy, ymax + 0.5*dy, num=nbins)
    xI = np.digitize(xall, xbins)
    yI = np.digitize(yall, ybins)
    # result
    z = np.zeros((nbins, nbins))
    N = np.zeros((nbins, nbins))
    # average over bins
    for t in range(len(xall)):
        z[xI[t], yI[t]] += zall[t]
        N[xI[t], yI[t]] += 1.0
    #with warnings.catch_warnings() as cm:
        #warnings.simplefilter('ignore')
        #z /= N
    # do a contour plot
    extent = [xmin, xmax, ymin, ymax]

    lev_step=0.0001
    lev=[x*lev_step for x in range(400)]
    plt.contourf(z.T, 100, extent=extent, cmap=cmap, levels = lev)
    plt.clim(0,0.05)
    
    plt.rc('text', usetex=True)
    
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=30)
            cbar.ax.tick_params(labelsize=30)

    plt.xlim(xbins.min()-5,xbins.max()+5)
    plt.ylim(ybins.min()-5,ybins.max()+5)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    plt.xlabel(labels[xdim], fontsize=30)
    plt.ylabel(labels[ydim], fontsize=30)
    
    return plt

#################
#Plot Timescale curves
def plot_its(mplt,its_dim_type,x_lim,y_lim):
    #Plot relaxation timescales
    mpl.rcParams.update({'font.size': 20})
    mplt.plot_implied_timescales(its_dim_type, ylog=True, dt=0.1, units='ns', linewidth=2)
    plt.xlim(0, x_lim); plt.ylim(0, y_lim);
    #save_figure('its.png')
    
    return

#################    
def plot_timescale_ratios(its,ntims=5,ylim=4):

    tim=np.transpose(its.timescales)
    lags=its.lags
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(ntims):
        plt.plot(lags/10,tim[i]/tim[i+1],'-o',label="$t_{"+str(i+1)+"}$/$t_{"+str(i+2)+"}$")

    plt.rc('text', usetex=True)    
    
    plt.xlim(0,30+np.max(lags)/10)
    plt.ylim(0,ylim)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    plt.xlabel("Time (ns)", fontsize=30)
    plt.ylabel(r"$t_{i}/t_{i+1}$", fontsize=30)

    legend = plt.legend(loc='upper right', shadow=False, fontsize='small')

    return    

#################
def plot_kinetic_variance(its,ylim=20):

    lags=its.lags
    fig, ax = plt.subplots(figsize=(6,4))
    kinvar=[(M.eigenvalues()**2).sum() for M in its.models]
    plt.plot(0.1*lags, kinvar, linewidth=2)
    

    plt.rc('text', usetex=True)    
    
    plt.xlim(0,np.max(lags)/10)
    plt.ylim(0,ylim)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)
    plt.xlabel("Time (ns)", fontsize=30)
    plt.ylabel(r"$\sigma^{2}$", fontsize=30)

    return

##########################################################################################################

#################
#File Writing Functions
#################

#################
def write_list_to_file(fname,lname):
    """
    #Writes a list to a filename: fname is filename, lname is list name e.g. traj_list
    """        
    with open(fname,'w') as f:
        for item in lname:
            f.write("%s\n" % item)

    return

#################
def save_current_fig(plt, figname):

    fig = plt.gcf()
    fig.set_size_inches(12, 9)
    plt.savefig(figname, dpi=600, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    
    return

#################
def save_current_fig2(plt, figname,pad=0.1):

    fig = plt.gcf()
    #fig.set_size_inches(12, 9)
    plt.savefig(figname, dpi=600, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches='tight', pad_inches=pad,
            metadata=None)
    
    return



##########################################################################################################

#################
#Pre-Optimized Coarse-Grained Metastable State Kinetic Rate Calculation and Transition Path Theory Functions
#################

#################
def tpt_rate_matrix(M,pcca_sets,tfac):
    n_sets=len(pcca_sets)
    rate=np.zeros((n_sets,n_sets))
    for i in range(n_sets):
        for j in range(n_sets):
            if i != j:
                rate[i,j]=tfac*(msm.tpt(M,pcca_sets[i],pcca_sets[j]).rate)
    return rate

#################
def gamma_factor(kon,init,fin):
    
    #gam=np.sum(kon[init,fin])/(np.sum(kon[init,fin]) + np.sum(kon[fin,init]))
    gam=np.sum(kon[np.ix_(init,fin)])/(np.sum(kon[np.ix_(init,fin)]) + np.sum(kon[np.ix_(fin,init)]))
    return gam

def tau_c(kon,init,fin):
    
    #gam=np.sum(kon[init,fin])/(np.sum(kon[init,fin]) + np.sum(kon[fin,init]))
    tau_c=1/(np.sum(kon[np.ix_(init,fin)]) + np.sum(kon[np.ix_(fin,init)]))
    return tau_c

#################
def metastable_kinetics_calc(M,tau,n_sets,init,fin):
    
    #Create the PCCA sets, distributions, memberships
    M.pcca(n_sets)
    pccaX = M.metastable_distributions
    pccaM = M.metastable_memberships  # get PCCA memberships
    pcca_sets = M.metastable_sets
    pcca_assign = M.metastable_assignments
    
    #Calculate Free energy based on the raw prob of discretized snapshots of microstates belonging to metastable state
    all_disc_snaps=np.hstack([dtraj for dtraj in M.discrete_trajectories_full])
    mstate_rho=np.array([len(np.where(all_disc_snaps==i)[0]) for i in range(n_clusters)])/len(all_disc_snaps)
    meta_rho=np.array([np.sum(mstate_rho[pcca_sets[i]]) for i in range(len(pcca_sets))])

    # Calculate Free Energy based on sum of stationary distribution (as calculated by transition matrix) of microstates belonging to each metastable state
    P_msm=M.transition_matrix
    meta_pi=np.array([np.sum(M.stationary_distribution[pcca_sets[i]]) for i in range(len(pcca_sets))])
    
    #Manually Calculate the HMM free energy from the X and M matrices
    NORM_M=np.linalg.inv(np.dot(np.transpose(pccaM),pccaM))
    NORM_X=np.linalg.inv(np.dot(pccaX,np.transpose(pccaX)))
    I=np.identity(len(pccaM))
    PI=np.transpose(M.pi)*I
    cg_pi=np.dot(NORM_X,np.dot(pccaX,np.dot(PI,pccaM)))
    cg_pi=np.sum(np.identity(len(pccaX))*cg_pi,axis=0)

    #Calculate CG transition matrix from manually constructed HMM (prior to Baum-Welch optimisation)
    P_tilda=np.dot(NORM_M,np.dot(np.transpose(pccaM), np.dot(P_msm,pccaM)))
        
    #Calculate k_on matrix from CG transition matrix
    #1000 factor is to convert to microseconds^-1
    kon_tilda=1000.*P_tilda/tau
    
    #Non-diagonal rate matrix with from/to state labelling
    kon_tilda_nd=nondiag_rates(kon_tilda)
    
    #Calculate k_on from TPT rate matrix
    tfac=10000
    kon_tpt=tpt_rate_matrix(M,pcca_sets,tfac)
    kon_tpt_nd=nondiag_rates(kon_tpt)
    
    #Calculate gating factor for various kon_matrices
    gam_kon_tilda= gamma_factor(kon_tilda,init,fin)
    gam_kon_tpt= gamma_factor(kon_tpt,init,fin)
    
    
    
    return meta_rho, meta_pi, cg_pi, kon_tilda_nd, kon_tpt_nd, gam_kon_tilda, gam_kon_tpt

#################
def nondiag_rates(kon):
    nd_rates=np.zeros((0,4))
    for i in range(len(kon)):
        for j in range(i+1,len(kon)):
            nd_rates=np.vstack((nd_rates, [int(i), int(j), kon[i,j], kon[j,i]]))
            
    return nd_rates

#################
def tau_c(kon,init,fin):
    
    #gam=np.sum(kon[init,fin])/(np.sum(kon[init,fin]) + np.sum(kon[fin,init]))
    tau_c=1/(np.sum(kon[np.ix_(init,fin)]) + np.sum(kon[np.ix_(fin,init)]))
    return tau_c



##########################################################################################################

#################
#Functions to Identify and Extract Representative Conformations of Metastable States
# Using both MSM approach and also from subselection of PMF landscapes
#################

#################

def micro_order(Xsets, Xdist,macro_state):
    
    kBT=0.596
    a=np.transpose(np.vstack((Xsets[macro_state],Xdist[macro_state,Xsets[macro_state]])))
    b = a[a[:,1].argsort()[::-1]]
    c = (-kBT*np.log(b[:,1])-np.min(-kBT*np.log(b[:,1]))).reshape(-1,1)

    micro_state_order=np.hstack((b,c))

    return micro_state_order


# Microstate and snapshot extractor
def top_microstates(macro_state, Xdist, Xsets, energy_factor):
    """
    a: Creates a Boltz-weighted list of fuzzy microstates of a given macrostate
    b: Creates a Boltz-weighted probability-sorted list of fuzzy microstates of a given macrostate
    from the pccaX (Xdist) distribution. Most prob state is set to 0, other states are ranked relative to the most prob state
    energy factor - define a cut off Boltz-weighted energy difference to select only most probable states = 0.5*kBT
    chi_conf_sets: list of chosen microstates
    lam - normalized lambda metric for clustercenters correponding to microstates
    """    
    kBT=0.596
    energy_thresh=energy_factor*kBT
    mic_states = np.array([x for x in range(np.shape(Xdist)[1])])
    a=np.transpose(np.vstack((mic_states,-kBT*np.log(Xdist[macro_state]))))
    ind=np.argsort(a[:,-1])
    b=a[ind]
    #Calculate B-weighted prob relative to most popular state
    b=b-np.transpose(np.vstack((np.zeros(np.shape(b)[0]),b[0][1]*np.ones(np.shape(b)[0]))))
    b[:,0]=b[:,0].astype(int)
    np.set_printoptions(precision=3)
    #print(b)
    
    #Select only those microstates that are within the corresponding macrostate set
    b_set=b[ [np.where(b[:,0]==x)[0][0] for x in Xsets[macro_state]], : ]
    
    #Choose only microstates with Boltz-weighted Chi probabilities within energy threshold from most favoured microstate
    b_set=b_set[ np.where(b_set[:,1]<=energy_thresh)[0], :  ]
    
    b_set= b_set[b_set[:,1].argsort()]
    b_set[:,0]=b_set[:,0].astype(int)
    
    chi_conf_sets=b_set[:,0].astype(int)
    
    #chi_conf_sets=np.take(b_set[:,0],np.where(b_set[:,1]<=energy_thresh)).astype(np.int)[0]
    #lam=np.array([np.linalg.norm(lambda_cl_obj.clustercenters[x]) for x in chi_conf_sets])

    
    return chi_conf_sets, b_set

#################
#Find trajectory indices and snapshot indices that belong to a given microstate
#All snaps contains list of: microstate id, traj id, snap id - all starting enummeration from 0
def micro_snap_extractor(macro_state, Xdist, chi_conf_sets, lambda_dtrajs):
    """
    micro_confs: makes a list of the microstate confs (id), kBT-weighted Chi prob, normalized lambda distance of cluster center, number of snapshots
    all_snaps: makes a list of all snapshots (micorstate id, traj id and frame id) belonging to selected microstates - 
    pulls this out of the discretized trajectories after the clustering step - lamda_dtrajs
    Input - list of microstates (chi_conf_sets), lambda_dtrajs, macro_state, Xdist
    """
    #kBT=0.596
    #mic_states = np.array([x for x in range(np.shape(Xdist)[1])])
    #a=np.transpose(np.vstack((mic_states,-kBT*np.log(Xdist[macro_state]))))
    all_snaps=np.empty((0,3))
    num_snaps=np.empty((0,1))
    for j in chi_conf_sets:
        mstate_snaps=np.empty((0,3))
        mstate=j
        for i in range(np.shape(lambda_dtrajs)[0]):
            tr=i
            snap=np.where(lambda_dtrajs[tr]==mstate)
            if not snap[0].size==0:
                inds=np.transpose(np.vstack( (mstate*np.ones(np.shape(snap[0])),tr*np.ones(np.shape(snap[0])),snap[0])  ) )
                #print(inds)
                mstate_snaps=np.vstack( (mstate_snaps,inds ) ).astype(int)
        num_snaps=np.vstack( (num_snaps,np.shape(mstate_snaps)[0] ) ).astype(int)
        all_snaps=np.vstack( (all_snaps,mstate_snaps ) ).astype(int)

    #print(all_snaps)
    #micro_confs=np.transpose(np.vstack( (np.transpose(a[chi_conf_sets]),np.transpose(lam),np.transpose(num_snaps)) ) )
    #micro_confs=np.transpose(np.vstack( (np.transpose(a[chi_conf_sets]),np.transpose(num_snaps)) ) )

    micro_confs=np.transpose(np.vstack( (np.transpose(chi_conf_sets),np.transpose(num_snaps)) ) )

    
    return micro_confs, all_snaps


def lamsort_all_snaps(Yred_cyl, mic_centroid,all_snaps):

    all_lam=np.empty((0,4))
    for x in all_snaps:
        lam=Yred_cyl[x[1]][x[2]]
        dist=np.linalg.norm(lam-mic_centroid)
        lam=np.hstack((lam,dist))    
        all_lam=np.vstack(( all_lam, lam))

    all_lam = np.hstack((all_snaps, all_lam))

    sort_all_snaps = all_lam[all_lam[:,6].argsort()[::1]]

    return sort_all_snaps[:,:3].astype(int), sort_all_snaps


def lamsort_cartesian_all_snaps(Yred, mic_centroid,all_snaps,th_off=50):

    mic_centroid_xyz=np.array([ mic_centroid[0]*np.cos(np.pi*(mic_centroid[1]+th_off)/180), 
                               mic_centroid[0]*np.sin(np.pi*(mic_centroid[1]+th_off)/180), mic_centroid[2] ])
    
    all_lam=np.empty((0,4))
    for x in all_snaps:
        lam=Yred[x[1]][x[2]]
        dist=np.linalg.norm(lam-mic_centroid_xyz)
        lam=np.hstack((lam,dist))    
        all_lam=np.vstack(( all_lam, lam))

    all_lam = np.hstack((all_snaps, all_lam))

    sort_all_snaps = all_lam[all_lam[:,6].argsort()[::1]]

    return sort_all_snaps[:,:3].astype(int), sort_all_snaps




#################
#Obsolete
def create_snap_sample(lambda_Y, all_snaps, snap_fname, num_samples):
    #Create and save a snapshot sample list of traj and snapshot indices information
    arr=[]
    for x in range(num_samples):
        arr.append(random.randint(0,np.shape(all_snaps)[0]))    

    #print(all_snaps[arr])
    #sample contains: extracted Microstate, traj index and snapshot indices for a given sample of snapshots
    sample =np.transpose(np.vstack( ( np.transpose(all_snaps[arr]),np.array([np.linalg.norm(lambda_Y[x[0]][x[1]]) for x in all_snaps[arr][:,1:3] ]) ) ) )
    #print(sample)
    sample[:,0:3]=sample[:,0:3].astype(int)
    #Pull out only a list of traj and snap indices
    numpy.set_printoptions(precision=3)
    snap_sample=sample[:,1:3].astype(int)
    #Save traj and snap indices into a sample snapshots file
    np.savetxt(snap_fname, snap_sample, fmt='%1d',delimiter=' ')
    return snap_sample, sample

#################
#Obsolete
def create_back_sample(lambda_Y, all_snaps, snap_sample):
    #extract sample list from snap_sample file
    arr = np.array([numpy.where((all_snaps[:,1]==x[0]) & (all_snaps[:,2]==x[1]) )[0][0] for x in snap_sample])
    sample =np.transpose(np.vstack( ( np.transpose(all_snaps[arr]),np.array([np.linalg.norm(lambda_Y[x[0]][x[1]]) for x in all_snaps[arr][:,1:3] ]) ) ) )
    sample[:,0:3]=sample[:,0:3].astype(int)
    return sample


#################
def create_snaps(all_snaps,sims_array,sample=False,num_samples=100):
    #Create snapshot sample list of traj (MD traj no) and snapshot indices (frame no) information

    #Create a random sample snapshots of size num_samples from all_snaps     
    if sample:
        arr=[]
        lim=np.min([num_samples,len(all_snaps)])
        for x in range(lim):
            arr.append(random.randint(0,len(all_snaps)-1))    
    else:
        arr=[x for x in range(len(all_snaps))]
    
    # We need all_snaps last 2 cols are already what we need for snap_list_xtc
    snap_list_xtc=all_snaps[arr,1:3].astype(int)
    
    snap_list_mlabel=all_snaps[arr,:].astype(int)
    
    # Then change to MD traj no and frame id format for snap_list - using info from sims_array
    traj_id=snap_list_xtc[:,0]
    snap=snap_list_xtc[:,1]    
    snap_list=np.transpose(np.vstack((sims_array[traj_id,1], snap)))
    snap_list_m=np.transpose(np.vstack(( snap_list_mlabel[:,0], sims_array[traj_id,1], snap)))
    
    #snap_list can later be saved as a datafile, snap_list_xtc used to save an xtc file
    return snap_list, snap_list_xtc, snap_list_m


#################
#Determine snapshots/indices corresponding to sub-regions of the plot
#Works for any number of dimensions
def conf_indices(Z,dims,bounds):
    
    condition=[True for x in range(len(Z))] 
    for i in range(len(dims)):           
        condition = condition & (Z[:,dims[i]]>=bounds[2*i]) & (Z[:,dims[i]]<=bounds[2*i + 1])
    
    Cind=np.where(condition)[0]    
    
    return Cind


#################
def snaps_to_indices(nsnaps,snaps,sims_array,sample=False,num_samples=100):
    
    #traj_no=sims_array[traj_id,1]
    #traj_id=np.where(sims_array[:,1]==traj_no)
    
    #Create a random sample snapshots of size num_samples from snap_list
    if sample:
        arr=[]
        lim=np.min([num_samples,len(snaps)])
        for x in range(lim):
            arr.append(random.randint(0,len(snaps)-1))    
    else:
        arr=[x for x in range(len(snaps))]    
    
    snaps=snaps[arr]
    
    #Array of traj indices as stored in inp/coor info and used for Y, Z from an array of traj_no as given in snaps
    traj_id=np.array([np.where(sims_array[:,1]==traj_no)[0][0] for traj_no in snaps[:,0]])

    #Array of indices in Z of corresponding snaps array
    ind = traj_id*nsnaps + snaps[:,1]

    return ind, snaps

#################
#Convert concatenated list of indices into [traj (starting from 1), snapshot (frame=starting from 0)] format
def indices_to_snaps(nsnaps,ind,sims_array,sample=False,num_samples=100):

    #Create a random sample snapshots of size num_samples from ind array    
    if sample:
        arr=[]
        lim=np.min([num_samples,len(ind)])
        for x in range(lim):
            arr.append(random.randint(0,len(ind)-1))    
    else:
        arr=[x for x in range(len(ind))]    
    
    ind=ind[arr]
    
    #Array of which traj id of uploaded Y data (numbering starting from 0) each snapshot is in
    traj_id=np.floor(ind/nsnaps).astype(int)
        
    
    #Array of snapshot number in that corresponding trajectory starting from 0
    snap=ind - (traj_id)*nsnaps

    #sims_array matches Y, so the traj index_th row of sims_array column index 1 (2nd columnn) contains the MD traj number 
    
    #batch=np.ones( (1,len(ind)) ).astype(int)
    snap_list = np.transpose(np.vstack((sims_array[traj_id,1], snap)))
    snap_list_xtc = np.transpose(np.vstack((traj_id, snap)))   
    
    return ind, snap_list, snap_list_xtc

## remember sims_array has traj and snaps starting from 1 not 0, whereas snap_list ouput here starts from traj_number 1 and frame 0
## Snap_list therefore lists the trajectory numbers of the MD sims
## Snap_list_xtc lists which sim array index of loaded trajs in the inp/sims_array to use in saving xtcs

#################
def save_snapshots_as_xtc(inp, snap_list_xtc, xtc_fname):
    
    #Create tuple array for processing by save_trajs
    tuple_sample=[np.array([tuple(snap_list_xtc[x]) for x in range(np.shape(snap_list_xtc)[0])])]
    
    #Save coordinates corresponding to traj and snapshot indices in xtc file
    coor.save_trajs(inp, tuple_sample, outfiles=[xtc_fname])
    
    return

#################
#Selection of conformation snapshots based on subsets of PMF space and corresponding indices
def conformation_selection(inp,sims_array,nsnaps,Z,dims,bounds,sample=False,num_samples=100,datfile=None,xtcfile=None):

    inds=conf_indices(Z, dims, bounds)
    inds, snaps, snaps_xtc=indices_to_snaps(nsnaps,inds,sims_array,sample,num_samples)
    traj, counts = np.unique(snaps[:,0], return_counts=True)
    trajcounts=np.transpose(np.vstack((traj, counts)))
    trajcounts=trajsort(trajcounts)
    if datfile is not None:
        np.savetxt(datfile, snaps, fmt='%d', delimiter=' ')
    if xtcfile is not None:
        save_snapshots_as_xtc(inp, snaps_xtc, xtcfile)
        
    return inds, snaps, trajcounts


#################
def msm_conformation_selection(inp,sims_array,all_snaps,sample=False,num_samples=100,datfile=None,xtcfile=None,mlabelfile=None):

    snaps, snaps_xtc, snaps_mlabel=create_snaps(all_snaps,sims_array,sample,num_samples)
    traj, counts = np.unique(snaps[:,0], return_counts=True)
    trajcounts=np.transpose(np.vstack((traj, counts)))
    trajcounts=trajsort(trajcounts)
    if datfile is not None:
        np.savetxt(datfile, snaps, fmt='%d', delimiter=' ')
    if xtcfile is not None:
        save_snapshots_as_xtc(inp, snaps_xtc, xtcfile)
    if mlabelfile is not None:
        np.savetxt(mlabelfile, snaps_mlabel, fmt='%d', delimiter=' ')
        
        
    return snaps, trajcounts, snaps_mlabel



##########################################################################################################

#################
#Array Sorting functions

#################
def trajsort(trajcounts):
    #Sort trajcounts in descending order 
    return trajcounts[trajcounts[:,1].argsort()[::-1],:]


##########################################################################################################

#################
#Minimum Fluctuation Alignment functions - for use with MD Analysis

#################
def MFA_matrix(ax_points):

    v1=ax_points['ax1_end'].centroid() - ax_points['ax1_origin'].centroid()
    v2=ax_points['ax2_end'].centroid() - ax_points['ax2_origin'].centroid()

    mfa_z=np.cross(v2,v1)/np.linalg.norm(np.cross(v2,v1))
    mfa_x=-1*v2/np.linalg.norm(v2)
    mfa_y=np.cross(mfa_z,mfa_x)
    mfa_origin=ax_points['ax2_origin'].centroid() + 0.5*v2

    # for u'_j new coords to be transformed from u_i coords in old coord space
    #u_i=Q_ij.u'_j, where Q_ij=cos(x_i,x'_j) 
    #and x_i and x'_j are the Cartesian axes directions in the old and new space respectively
    I=np.identity(3)
    Q=np.transpose(np.vstack((mfa_x,mfa_y,mfa_z)))

    #Here we calculate the inverse=transpose of Q to find the new coords
    #print(np.linalg.inv(np.dot(I,MFA)))
    QT=np.transpose(np.dot(I,Q))

    return QT, mfa_origin 

#################
def MFA(u,ax_points,sel):
    
    QT, mfa_origin = MFA_matrix(ax_points)
    atom_positions=u.select_atoms(sel).positions
    atom_positions -= mfa_origin
    mfa_atom_positions=np.transpose(np.dot(QT,np.transpose(atom_positions)))

    return mfa_atom_positions


#################
def MFA_CENTROID(u,ax_points,sel):
    
    QT, mfa_origin = MFA_matrix(ax_points)
    atom_positions=u.select_atoms(sel).centroid()
    atom_positions -= mfa_origin
    mfa_atom_positions=np.transpose(np.dot(QT,np.transpose(atom_positions)))

    return mfa_atom_positions


#################
def MFA_projected_vector(u,ax_points, sel_1, sel_2,frames=None):
    
    if frames is None:
        frames=[x for x in range(len(u.trajectory))]
        
    data=np.empty((0,3))
    for ts in u.trajectory[frames]:
        r_1=MFA_CENTROID(u,ax_points,sel_1)
        r_2=MFA_CENTROID(u,ax_points,sel_2) 
        data=np.vstack((data,r_2-r_1))
    
    return data


def rmsd_analysis(sys_id,systems, macro_dir, ref_prmtop, ref_pdb, traj_prmtop):

    ALIGN_SELECTION="protein and name CA and resid 1:42 59:141 158:198"
    PROTEIN="protein and name CA and resid 1:198"
    FLAPS="protein and name CA and resid 43:58 142:157"

    trajfile=macro_dir+'/hmm_'+str(systems[sys_id])+'.xtc'
    u_ref = MDAnalysis.Universe(ref_prmtop,ref_pdb)
    u = MDAnalysis.Universe(traj_prmtop,trajfile)

    #RMSD
    R = rms.RMSD(u, u_ref, select=ALIGN_SELECTION, groupselections=[PROTEIN, FLAPS],ref_frame=0)
    R.run()

    return R.rmsd



##########################################################################################################



def kde_function(Y,Y_plot,h):
    
    Y_range=Y_plot.reshape((-1,1))    
    kde = KernelDensity(kernel='epanechnikov', bandwidth=h).fit(Y)
    log_dens = kde.score_samples(Y_range)
    
    return Y_range[:,0], np.exp(log_dens)


def timeseries_axes(ax,xlim,ylim,x_ticks,y_ticks,xlabel,ylabel):
    
    plt.rc('text', usetex=True)
    
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=20)

    ax.set_ylim(ylim[0],ylim[1])
    ax.set_ylabel(ylabel,fontsize=20)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=20)

    return ax


def distribution_plot(gs_id,data,metric,conf,D_space,h,color,alph=1):

    
    ax = plt.subplot(gs_id)
    D=data[:,metric,conf].reshape(-1,1)
    yfunc, epan = kde_function(D,D_space,h)    
    ax.fill(yfunc,epan, fc=color,alpha=alph)

    return ax

##########################################################################################################

def conformation_data(Z, sys_id, systems, macro_dir, sims_array):

    snaps=read_int_matrix(macro_dir+'/hmm_'+str(systems[sys_id])+'.dat')
    snap_inds,snaps=snaps_to_indices(1300,snaps,sims_array)
    Zdata=Z[snap_inds]
    
    return Zdata, snap_inds, snaps



##########################################################################################################
# Volmap functions

def make_ticks(D,edge_id,increment=20,orient=1):

    dim=D.edges[edge_id][::orient]
    zeropoint=np.where(dim==0)[0][0]
    half_range=np.floor(np.array([len(dim)-zeropoint, zeropoint]).min()/increment).astype(int)*increment
    tic=[x for x in range(0,len(dim))][zeropoint-half_range:zeropoint+half_range:increment]
    lab=dim[zeropoint-half_range:zeropoint+half_range:increment].astype(int)

    return tic,lab

def Zpoint(D, Zheight):
    
    Zdim=D.edges[2]
    Z_zeropoint=np.where(Zdim==0)[0][0]
    Zdelta=D.delta[2]
    Zgridpoint=Z_zeropoint+(Zheight/Zdelta).astype(int)
    
    return Zgridpoint


def XYZpoint(D, dim, XYZheight):
    
    XYZdim=D.edges[dim]
    XYZ_zeropoint=np.where(XYZdim==0)[0][0]
    XYZdelta=D.delta[dim]
    XYZgridpoint=XYZ_zeropoint+(XYZheight/XYZdelta).astype(int)
    
    return XYZgridpoint

def D_cross_section(D,dim,XYZheight):
    if dim==0: 
        D_csec=D.grid[XYZpoint(D,dim,XYZheight),:,:]
    elif dim==1:
        D_csec=D.grid[:,XYZpoint(D,dim,XYZheight),:]
    elif dim==2:
        D_csec=D.grid[:,:,XYZpoint(D,dim,XYZheight)]
    
    return D_csec


def Dmatrix_cross_section(D,Do,dim,XYZheight):
    if dim==0: 
        D_csec=Do[XYZpoint(D,dim,XYZheight),:,:]
    elif dim==1:
        D_csec=Do[:,XYZpoint(D,dim,XYZheight),:]
    elif dim==2:
        D_csec=Do[:,:,XYZpoint(D,dim,XYZheight)]
    
    return D_csec


def plot_cross_section(plt, D, dim, XYZheight,cmp='Greys'):

    D_csec=D_cross_section(D,dim,XYZheight)  
    plt.imshow(np.transpose(np.flipud(D_csec)),cmap=cmp)
    
    
    Xtic,Xlab=make_ticks(D,0,20)
    Ytic,Ylab=make_ticks(D,1,20,-1)

    plt.xticks(Xtic,Xlab,fontsize=10, rotation=0)
    plt.yticks(Ytic,Ylab,fontsize=10, rotation=0)
    
    
    return plt
    
    

def plot_cross_section_overlay(plt, D1, D2, dim, XYZheight,cmp1='Reds',cmp2='Blues',overlay=False,flip=True,fsize=10):
    
    D1_csec=D_cross_section(D1,dim,XYZheight)
        
    if flip:
        plt.imshow(np.flipud(np.transpose(D1_csec)),cmap=cmp1)
    else:
        plt.imshow(np.transpose(D1_csec),cmap=cmp1)
        
    if overlay:
        D2_csec=D_cross_section(D2,dim,XYZheight)
        if flip:
            plt.imshow(np.flipud(np.transpose(D2_csec)),cmap=cmp2,alpha=0.3)
        else:
            plt.imshow(np.transpose(D2_csec),cmap=cmp2,alpha=0.3)

            
    Xtic,Xlab=make_ticks(D1,0,increment=40,orient=1)
    Ytic,Ylab=make_ticks(D1,1,increment=40,orient=-1)

    plt.rc('text', usetex=True)

    plt.xticks(Xtic,Xlab,fontsize=fsize, rotation=0)
    plt.yticks(Ytic,Ylab,fontsize=fsize, rotation=0)
    
    lab=['x (\AA)','y (\AA)','z (\AA)']
    lab.pop(dim)
    
    plt.xlabel(lab[0],fontsize=fsize)
    plt.ylabel(lab[1],fontsize=fsize)
    
    
    return plt


def plot_Doverlap_cross_section(plt, D1, Do, dim, XYZheight, cmp1='Reds',flip=True):
    
    D1_csec=Dmatrix_cross_section(D1,Do,dim,XYZheight)
    
    if flip:
        plt.imshow(np.flipud(np.transpose(D1_csec)),cmap=cmp1)
    else:
        plt.imshow(np.transpose(D1_csec),cmap=cmp1)
                    
    Xtic,Xlab=make_ticks(D1,0,increment=20,orient=1)
    Ytic,Ylab=make_ticks(D1,1,increment=20,orient=-1)

    plt.xticks(Xtic,Xlab,fontsize=10, rotation=0)
    plt.yticks(Ytic,Ylab,fontsize=10, rotation=0)
    
    
    return plt


def plot_cross_section_overlay_cbar(plt, D1, D2, dim, XYZheight,cmp1='Reds',cmp2='Blues',cbar=False, cbar_label=None, overlay=False,flip=True,fsize=10):
    
    D1_csec=D_cross_section(D1,dim,XYZheight)
    
    if cbar:
        cbar = plt.colorbar()
    if cbar_label is not None:
        cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=30)
        cbar.ax.tick_params(labelsize=30)
    
    
    if flip:
        plt.imshow(np.flipud(np.transpose(D1_csec)),cmap=cmp1)
    else:
        plt.imshow(np.transpose(D1_csec),cmap=cmp1)
        
    if overlay:
        D2_csec=D_cross_section(D2,dim,XYZheight)
        if flip:
            plt.imshow(np.flipud(np.transpose(D2_csec)),cmap=cmp2,alpha=0.3)
        else:
            plt.imshow(np.transpose(D2_csec),cmap=cmp2,alpha=0.3)
            
    Xtic,Xlab=make_ticks(D1,0,increment=40,orient=1)
    Ytic,Ylab=make_ticks(D1,1,increment=40,orient=-1)

    plt.rc('text', usetex=True)

    plt.xticks(Xtic,Xlab,fontsize=fsize, rotation=0)
    plt.yticks(Ytic,Ylab,fontsize=fsize, rotation=0)
    
    lab=['x (\AA)','y (\AA)','z (\AA)']
    lab.pop(dim)
    
    plt.xlabel(lab[0],fontsize=fsize)
    plt.ylabel(lab[1],fontsize=fsize)
    
    
    return plt

def plot_cross_section_systems(plt, systems, gs, gs_id, Dlist, D2, dim, extent,fs):

    for i in range(len(systems)):
        ax = plt.subplot(gs[gs_id])        
        D1=Dlist[i]
        plt=plot_cross_section_overlay(plt, D1, D2, dim, extent,cmp1='binary',cmp2='Reds',overlay=True,fsize=fs)

        gs_id+=1
    
    
    return plt, gs_id

def plot_structures_systems(plt, systems, gs, gs_id,orient):
    
    for i in range(len(systems)):
        ax=plt.subplot(gs[gs_id])
        ax.axis("off")
        img = mpimg.imread('../figures/conf_figs/png/'+str(systems[i])+'_'+orient+'.png')
        imgplot = plt.imshow(img)
    
        plt.xticks([],fontsize=10, rotation=0)
        plt.yticks([],fontsize=10, rotation=0)
    
        gs_id+=1
    
    return plt, gs_id


##########################################################################################################
# MD Analysis for HIV-1 protease system

def load_universe(sys_id,systems,macro_dir,traj_prmtop):
    
    trajfile=macro_dir+'/hmm_'+str(systems[sys_id])+'.xtc'
    u =  MDAnalysis.Universe(traj_prmtop,trajfile)
    
    return u


def MFA_axis_points(u):
    #Define axis points from which to define MFA vectors
    ax_points = {
        'ax1_origin': u.select_atoms("protein and resid 89-92 and backbone"),
        'ax1_end': u.select_atoms("protein and resid 188-191 and backbone"),
        'ax2_origin': u.select_atoms("protein and resid 23 24 85 and backbone"),
        'ax2_end': u.select_atoms("protein and resid 122 123 184 and backbone"),
        }
    
    return ax_points

def macrostate_structural_data(u,cl_obj,top_sets,macro_state,ranked_microstate):

    ax_points=MFA_axis_points(u)
    #lam_xyz for a given MSM macrostate conformation selection
    sel_1="protein and backbone and resid 50"
    sel_2="protein and backbone and resid 149"
    lam_xyz=MFA_projected_vector(u,ax_points, sel_1, sel_2)
    #convert to cylindrical polar and shift by 50 degrees to match recentering for landscape
    lam_cyl=xyz_to_cyl_coords(lam_xyz,50)

    lam = np.linalg.norm(lam_xyz,axis=1).reshape(-1,1)
    
    #Proximity of cylindrical coords to centroid of a defined microstate
    mic_centroid=cl_obj.clustercenters[top_sets[macro_state][ranked_microstate]]
    dist=np.linalg.norm(lam_cyl-mic_centroid,axis=1).reshape(-1,1)
    
    
        
    data=np.hstack((lam_xyz,lam_cyl,lam,dist))

    return data
#data=macrostate_structural_data(u_c1a,cl_obj,top_sets,0,1)

##########################

def HIVPR_ax_points(u):

    #Define axis points from which to define MFA vectors
    ax_points = {
        'ax1_origin': u.select_atoms("protein and resid 89-92 and backbone"),
        'ax1_end': u.select_atoms("protein and resid 188-191 and backbone"),
        'ax2_origin': u.select_atoms("protein and resid 23 24 85 and backbone"),
        'ax2_end': u.select_atoms("protein and resid 122 123 184 and backbone"),
        }

    return ax_points

def MFA_pos(ax_points,selection):
    
    QT, mfa_origin = MFA_matrix(ax_points)
    atom_positions=selection.positions
    atom_positions -= mfa_origin
    mfa_atom_positions=np.transpose(np.dot(QT,np.transpose(atom_positions)))

    return mfa_atom_positions

##########################

### Distance distance map


def atom_selections(u):

    prot=u.select_atoms("protein and name CA and resid 1:198")
    monA=u.select_atoms("protein and name CA and resid 1:99")
    monB=u.select_atoms("protein and name CA and resid 100:198")
    flapA=u.select_atoms("protein and name CA and resid 43:58")
    flapB=u.select_atoms("protein and name CA and resid 142:157")
    pep=u.select_atoms("protein and name CA and resid 199:206")

    sel_list=[prot,monA,monB,flapA,flapB,pep]
    
    return sel_list



def ave_mfa_positions(u,ax_points,selection):
    
    mfa_pos_traj=np.empty((len(selection),3,0 ) )
    for ts in u.trajectory:
        
        mfa_positions= MFA_pos(ax_points,selection)
        mfa_pos_traj=np.dstack((mfa_pos_traj,mfa_positions))
        
    ave_mfa_pos=np.mean(mfa_pos_traj,axis=2)
    
    return ave_mfa_pos


def dmap_trajectory_old(u,pos1,pos2):

    dmap_traj=np.empty((len(pos1),len(pos2),0 ) )
    for ts in u.trajectory:

        dmap_frame=distances.distance_array(pos1,pos2)
        dmap_traj=np.dstack((dmap_traj,dmap_frame))

    ave_dmap=np.mean(dmap_traj,axis=2)
        
    return ave_dmap,dmap_traj


def dmap_trajectory(u,sel1,sel2,ax_points,mfa=True):

        
    
    dmap_traj=np.empty((len(sel1),len(sel2),0 ) )
    for ts in u.trajectory:

        if mfa:
            pos1=MFA_pos(ax_points,sel1)
            pos2=MFA_pos(ax_points,sel2)
        else:    
            pos1=sel1.positions
            pos2=sel2.positions
        
        dmap_frame=distances.distance_array(pos1,pos2)
        dmap_traj=np.dstack((dmap_traj,dmap_frame))

    ave_dmap=np.mean(dmap_traj,axis=2)
        
    return ave_dmap,dmap_traj



def within_cutoff(sys_list,cutoff):
    
    wcc_all=np.empty((1,0))
    for i in sys_list:
        wcc=np.unique(np.where(i<cutoff)[1]).reshape(1,-1)
        wcc_all=np.hstack((wcc_all,wcc))

    wcc_all=np.sort(np.unique(wcc_all)).astype(int)
    
    return wcc_all


def dmap_figure (plt, sys1, sys2, wcc, V1, V2, diff_plot=True, cmp1="hot", cmp2="jet"):

    if diff_plot:
        npanels=3        
    else:
        npanels=2
    
    npanels=int(npanels)
    
    fig = plt.figure(figsize=(npanels*3, 6))     
    fig.subplots_adjust(hspace=0.1, wspace=0.1)    

    gs = gridspec.GridSpec(1, npanels)

    ax=plt.subplot(gs[0])
    plt.imshow(np.transpose(sys1[:,wcc]),vmin=V1[0],vmax=V1[1],cmap=cmp1)
    ax=plt.subplot(gs[1])
    plt.imshow(np.transpose(sys2[:,wcc]),vmin=V1[0],vmax=V1[1],cmap=cmp1)
    
    cbar=plt.colorbar()    

    if diff_plot:
        ax=plt.subplot(gs[2])
        plt.imshow(np.transpose(sys1[:,wcc]-sys2[:,wcc]),vmin=V2[0],vmax=V2[1],cmap=cmp2)

        cbar2=plt.colorbar()

    plt.show()
        
    return plt


##########################

# Minimum Instantaneous Side-Chain Distance Functions

def minimum_distance_between_two_selections(sel_a,sel_b):
        
    dlist=[]
    for a in sel_a.positions:
        for b in sel_b.positions:    
            d=np.linalg.norm(a-b)
            dlist.append(d)
    dlist=np.asarray(dlist)

    return np.min(dlist)

def broadcast_minimum_distance_between_two_selections_of_positions(A,B):

    x=np.min(np.linalg.norm((A[:, np.newaxis]-B).reshape(-1,A.shape[1]),axis=1))

    return x


def broadcast_minimum_distance_between_two_selections(sel1,sel2,ax_points1,ax_points2,mfa=True):
    

    if mfa:
        A=MFA_pos(ax_points1,sel1)
        B=MFA_pos(ax_points2,sel2)
    else:    
        A=sel1.positions
        B=sel2.positions

    x=np.min(np.linalg.norm((A[:, np.newaxis]-B).reshape(-1,A.shape[1]),axis=1))

    return x


def sidechain_min_dist_matrix(u,reslim_a,reslim_b,mfa=False,ax_points={}):

    sidechain="not name H* and not name N CA O OX*"
    reslist_a=[a for a in range(reslim_a[0],reslim_a[1]+1)]
    reslist_b=[b for b in range(reslim_b[0],reslim_b[1]+1)]
    x_all=np.empty((0,1))
    for i in reslist_a:
        for j in reslist_b:
            res_a=u.select_atoms("protein and resid "+ str(i) +" and " + sidechain)
            res_b=u.select_atoms("protein and resid "+ str(j) +" and " + sidechain)

            if mfa:
                x=broadcast_minimum_distance_between_two_selections(res_a,res_b,ax_points)
            else:
                A=res_a.positions
                B=res_b.positions
                x=np.min(np.linalg.norm((A[:, np.newaxis]-B).reshape(-1,A.shape[1]),axis=1))
            
            x_all=np.vstack((x_all,x))
        
    x_all=x_all.reshape((len(reslist_a), len(reslist_b)  ) )        

    return x_all


def general_min_dist_matrix(u_a,u_b,reslim_a,reslim_b,selection_component,mfa=False,ax_points1={},ax_points2={}):

    reslist_a=[a for a in range(reslim_a[0],reslim_a[1]+1)]
    reslist_b=[b for b in range(reslim_b[0],reslim_b[1]+1)]
    x_all=np.empty((0,1))
    for i in reslist_a:
        for j in reslist_b:
            res_a=u_a.select_atoms("protein and resid "+ str(i) +" and " + selection_component)
            res_b=u_b.select_atoms("protein and resid "+ str(j) +" and " + selection_component)

            if mfa:
                x=broadcast_minimum_distance_between_two_selections(res_a,res_b,ax_points1,ax_points2)
            else:
                A=res_a.positions
                B=res_b.positions
                x=np.min(np.linalg.norm((A[:, np.newaxis]-B).reshape(-1,A.shape[1]),axis=1))
            
            x_all=np.vstack((x_all,x))
        
    x_all=x_all.reshape((len(reslist_a), len(reslist_b)  ) )        

    return x_all

##########################

#### Filtering Key Interactions

#### Close interactions

def close_interactions(sysmap,reslim_a,reslim_b,prox_thresh):
    
    a=np.where( (sysmap<prox_thresh) )[0]+reslim_a[0]
    b=np.where( (sysmap<prox_thresh) )[1]+reslim_b[0]
    
    resid_array=np.vstack((a,b))

    return resid_array
    
    

#prox_thresh=7
#diff_thresh=0

#interacting_flap_resids=np.where( (sys2<prox_thresh) & (sys1-sys2>diff_thresh) )[0]+43
#interacting_mon_resids=np.where( (sys2<prox_thresh) & (sys1-sys2>diff_thresh) )[1]+1

#print(np.vstack((interacting_flap_resids,interacting_mon_resids)))


#Trajectory

#def minimum_instantaneous_distance(u_a, resid_a,uresid_b)

def min_distance_trajectory(u,sel1,sel2,ax_points1,ax_points2):
    
    
    min_dist=np.empty((0,1)) 
    for ts in u.trajectory:
                
        x=broadcast_minimum_distance_between_two_selections(sel1,sel2,ax_points1,ax_points2)

        min_dist=np.vstack((min_dist,x))
        
    return min_dist


def res_res_array_min_sidechain_distance_trajectory(u_a,u_b, resid_array,selection_component):

    mind_all=np.empty((len(u_b.trajectory),0))
    for i in range(np.shape(resid_array)[1]):

        resid_a=resid_array[0,i]
        resid_b=resid_array[1,i]
        
        res_a=u_a.select_atoms("protein and resid " + str(resid_a) +" and " + selection_component)
        res_b=u_b.select_atoms("protein and resid " + str(resid_b) +" and " + selection_component)
        mind=min_distance_trajectory(u_b,res_a,res_b,HIVPR_ax_points(u_a),HIVPR_ax_points(u_b))
        
        mind_all=np.hstack((mind_all,mind))
        
    mean=np.mean(mind_all,axis=0)
    std=np.std(mind_all,axis=0)
        
        
    stats=np.transpose(np.vstack((resid_array,
                                  mean.reshape(1,-1),std.reshape(1,-1))))
        
    return mind_all,stats

def convert_hivpr_monB_to_amber_resids(residlist):
    
    residlist=residlist+99
    
    return residlist


##########################



def hbond_atom_definitions(u,hbdat):

    hbres=hbdat

    atmlist=np.empty((np.shape(hbres)[0],0))
    for i in range(1,4):
        atm=np.transpose(np.vstack(( np.array(u.atoms[hbres[:,i].astype(int)].names) , 
                np.array(u.atoms[hbres[:,i].astype(int)].resnames) ,
                np.array(u.atoms[hbres[:,i].astype(int)].resids) )))
        atmlist=np.hstack((atmlist,atm))

        
    return atmlist


def join_atom_defs_to_hbond_data(hbdat,atmlist):

    hbdat_atms=np.hstack((hbdat,atmlist))
    
    return hbdat_atms


def select_hbond_subset(hbdat_atms,donor_array,acceptor_array):
    
    
    hbdat_atms=hbdat_atms[np.isin(hbdat_atms[:,8],donor_array),:]
    hbdat_atms=hbdat_atms[np.isin(hbdat_atms[:,14],acceptor_array),:]
    
    
    return hbdat_atms


def unique_subset_hbond_data(hbdat_atms,reslim1,reslim2):

    res_arr1=[x for x in range(reslim1[0],reslim1[1]+1)]
    res_arr2=[x for x in range(reslim2[0],reslim2[1]+1)]

    hb_da_forward=select_hbond_subset(hbdat_atms,res_arr1,res_arr2) 
    hb_da_reverse=select_hbond_subset(hbdat_atms,res_arr2,res_arr1) 

    hb_combo=np.vstack((hb_da_forward,hb_da_reverse))

    unique_bonds=np.unique(hb_combo[:,1:4].astype(int), axis=0)

    prec=3
    unique_hb_list=np.empty((0,14))
    for x in unique_bonds:    
        hb_combo_x=hb_combo[np.all(hb_combo[:,1:4].astype(int)==x,axis=1),:]
    
        mean_hb_dist=np.around(np.mean(hb_combo_x[:,4]),prec)
        std_hb_dist=np.around(np.std(hb_combo_x[:,4]),prec)
        mean_hb_ang=np.around(np.mean(hb_combo_x[:,5]),prec)
        std_hb_ang=np.around(np.std(hb_combo_x[:,5]),prec)
    
        hb_combo_unique=np.hstack((hb_combo_x[0,6:],len(hb_combo_x),mean_hb_dist,std_hb_dist,mean_hb_ang,std_hb_ang))

        unique_hb_list=np.vstack((unique_hb_list,hb_combo_unique))
    

    return unique_hb_list,hb_combo



def convert_mixed_array_to_float(mix_arr):
    'converts mixed array of text and values to float where possbile '
    mixed=[]
    for a in list(mix_arr):
        try:
            mixed.append(float(a))
        except:
            mixed.append(a)
    mixed=np.array(mixed,dtype=object)
    return mixed


def individual_hbond_calculation(u,don_sel,hyd_sel,acc_sel):

    hb_dist=np.empty((0,1))
    hb_ang=np.empty((0,1))
    
    for ts in u.trajectory:
    
        don=u.select_atoms(don_sel).positions
        hyd=u.select_atoms(hyd_sel).positions
        acc=u.select_atoms(acc_sel).positions

        hb_dist=np.vstack((hb_dist,np.linalg.norm(don-acc)))
        hb_ang=np.vstack((hb_ang,(180/np.pi)*np.dot((don-hyd)[0],(acc-hyd)[0])
                          /(np.linalg.norm((don-hyd)[0])*np.linalg.norm((acc-hyd)[0])) % 180))

    data=np.hstack((hb_dist,hb_ang))
        
    return data

