# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt


import sys

#for regular expression search
import re

##############################################################
## first miscellaneous utilities
##############################################################


#arbitrary aa to int mapping, * stands for stop codon
aa_to_int = {"A":0, "C":1, "D":2, "E":3, "F":4, 
             "G":5,"H":6, "I":7, "K":8, "L":9, 
             "M":10, "N":11, "P":12, "Q":13, "R":14, 
             "S":15, "T":16, "V":17, "W":18, "Y":19, "*": 20}
dna_to_int = {"A":0, "C":1, "G":2, "T":3} 

#genetic code table
codon_to_aa = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                 
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
        'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'
        } 

#from the codon usage data base 
#https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=83333&aa=1&style=GCG
#copied by hand for E.coli K12
#units are codons per 1000 codons in the E.coli genome
#note that the value of TAG is set to a very small 1e-06 to avoid division by
#zero but the database actually reports it as zero
codon_usage_coli = {
        'ATA':3.71,   'ATC':18.16,  'ATT':30.46,  'ATG':24.8,
        'ACA':6.44,   'ACC':22.84,  'ACG':11.52,  'ACT':8.00,
        'AAC':24.4,   'AAT':21.87,  'AAA':33.19,  'AAG':12.1,
        'AGC':16.6,   'AGT':7.22,   'AGA':1.37,   'AGG':1.56,                 
        'CTA':5.27,   'CTC':10.54,  'CTG':46.86,  'CTT':11.91,
        'CCA':6.64,   'CCC':6.44,   'CCG':26.75,  'CCT':8.4,
        'CAC':13.08,  'CAT':15.81,  'CAA':12.10,  'CAG':13.08,
        'CGA':4.30,   'CGC':25.97,  'CGG':4.1,    'CGT':21.09,
        'GTA':11.52,  'GTC':11.71,  'GTG':26.36,  'GTT':16.79,
        'GCA':21.09,  'GCC':31.63,  'GCG':38.46,  'GCT':10.74,
        'GAC':20.50,  'GAT':37.88,  'GAA':43.73,  'GAG':18.35,
        'GGA':9.18,   'GGC':33.39,  'GGG':8.59,   'GGT':21.28,
        'TCA':7.81,   'TCC':5.47,   'TCG':8.00,   'TCT':5.66,
        'TTC':15.03,  'TTT':19.72,  'TTA':15.23,  'TTG':11.91,
        'TAC':14.64,  'TAT':16.79,  'TAA':1.76,   'TAG':1e-06,
        'TGC':8.00,   'TGT':5.86,   'TGA':0.98,   'TGG':10.74
        } 



 
#alphabetical codon to integer encoding
codon_to_int={"AAA": 0,"AAC": 1,"AAG": 2,"AAT": 3,"ACA": 4,
"ACC": 5,"ACG": 6,"ACT": 7,"AGA": 8,"AGC": 9,
"AGG": 10,"AGT": 11,"ATA": 12,"ATC": 13,"ATG": 14,
"ATT": 15,"CAA": 16,"CAC": 17,"CAG": 18,"CAT": 19,
"CCA": 20,"CCC": 21,"CCG": 22,"CCT": 23,"CGA": 24,
"CGC": 25,"CGG": 26,"CGT": 27,"CTA": 28,"CTC": 29,
"CTG": 30,"CTT": 31,"GAA": 32,"GAC": 33,"GAG": 34,
"GAT": 35,"GCA": 36,"GCC": 37,"GCG": 38,"GCT": 39,
"GGA": 40,"GGC": 41,"GGG": 42,"GGT": 43,"GTA": 44,
"GTC": 45,"GTG": 46,"GTT": 47,"TAA": 48,"TAC": 49,
"TAG": 50,"TAT": 51,"TCA": 52,"TCC": 53,"TCG": 54,
"TCT": 55,"TGA": 56,"TGC": 57,"TGG": 58,"TGT": 59,
"TTA": 60,"TTC": 61,"TTG": 62,"TTT": 63}
"""
#produced with this command
i=0
for cod in sorted(codon_to_aa.keys()):
    print("\""+str(cod)+"\":", i, end=',')
    i+=1
"""

#amino acids encoding with georgiev 2009 features, produces from script below
ggencoding={
 'A': [0.57, 3.37, -3.66, 2.34, -1.07, -0.4, 1.23, -2.32, -2.01, 1.31, -1.14, 0.19, 1.66, 4.39, 0.18, -2.6, 1.49, 0.46, -4.22], 
 'C': [2.66, -1.52, -3.29, -3.77, 2.96, -2.23, 0.44, -3.49, 2.22, -3.78, 1.98, -0.43, -1.03, 0.93, 1.43, 1.45, -1.15, -1.64, -1.05], 
 'D': [-2.46, -0.66, -0.57, 0.14, 0.75, 0.24, -5.15, -1.17, 0.73, 1.5, 1.51, 5.61, -3.85, 1.28, -1.98, 0.05, 0.9, 1.38, -0.03], 
 'E': [-3.08, 3.45, 0.05, 0.62, -0.49, 0, -5.66, -0.11, 1.49, -2.26, -1.62, -3.97, 2.3, -0.06, -0.35, 1.51, -2.29, -1.47, 0.15], 
 'F': [3.12, 0.68, 2.4, -0.35, -0.88, 1.62, -0.15, -0.41, 4.2, 0.73, -0.56, 3.54, 5.25, 1.73, 2.14, 1.1, 0.68, 1.46, 2.33], 
 'G': [0.15, -3.49, -2.97, 2.06, 0.7, 7.47, 0.41, 1.62, -0.47, -2.9, -0.98, -0.62, -0.11, 0.15, -0.53, 0.35, 0.3, 0.32, 0.05], 
 'H': [-0.39, 1, -0.63, -3.49, 0.05, 0.41, 1.61, -0.6, 3.55, 1.52, -2.28, -3.12, -1.45, -0.77, -4.18, -2.91, 3.37, 1.87, 2.17], 
 'I': [3.1, 0.37, 0.26, 1.04, -0.05, -1.18, -0.21, 3.45, 0.86, 1.98, 0.89, -1.67, -1.02, -1.21, -1.78, 5.71, 1.54, 2.11, -4.18], 
 'K': [-3.89, 1.47, 1.95, 1.17, 0.53, 0.1, 4.01, -0.01, -0.26, -1.66, 5.86, -0.06, 1.38, 1.78, -2.71, 1.62, 0.96, -1.09, 1.36], 
 'L': [2.72, 1.88, 1.92, 5.33, 0.08, 0.09, 0.27, -4.06, 0.43, -1.2, 0.67, -0.29, -2.47, -4.79, 0.8, -1.43, 0.63, -0.24, 1.01], 
 'M': [1.89, 3.88, -1.57, -3.58, -2.55, 2.07, 0.84, 1.85, -2.05, 0.78, 1.53, 2.44, -0.26, -3.09, -1.39, -1.02, -4.32, -1.34, 0.09], 
 'N': [-2.02, -1.92, 0.04, -0.65, 1.61, 2.08, 0.4, -2.47, -0.07, 7.02, 1.32, -2.44, 0.37, -0.89, 3.13, 0.79, -1.54, -1.71, -0.25], 
 'P': [-0.58, -4.33, -0.02, -0.21, -8.31, -1.82, -0.12, -1.18, 0, -0.66, 0.64, -0.92, -0.37, 0.17, 0.36, 0.08, 0.16, -0.34, 0.04], 
 'Q': [-2.54, 1.82, -0.82, -1.85, 0.09, 0.6, 0.25, 2.11, -1.92, -1.67, 0.7, -0.27, -0.99, -1.56, 6.22, -0.18, 2.72, 4.35, 0.92], 
 'R': [-2.8, 0.31, 2.84, 0.25, 0.2, -0.37, 3.81, 0.98, 2.43, -0.99, -4.9, 2.09, -3.08, 0.82, 1.32, 0.69, -2.62, -1.49, -2.57], 
 'S': [-1.1, -2.05, -2.19, 1.36, 1.78, -3.36, 1.39, -1.21, -2.83, 0.39, -2.92, 1.27, 2.86, -1.88, -2.42, 1.75, -2.77, 3.36, 2.67], 
 'T': [-0.65, -1.6, -1.39, 0.63, 1.35, -2.45, -0.65, 3.43, 0.34, 0.24, -0.53, 1.91, 2.66, -3.07, 0.2, -2.2, 3.73, -5.46, -0.73], 
 'V': [2.64, 0.03, -0.67, 2.34, 0.64, -2.01, -0.33, 3.93, -0.21, 1.27, 0.43, -1.71, -2.93, 4.22, 1.06, -1.31, -1.97, -1.21, 4.77], 
 'W': [1.89, -0.09, 4.21, -2.77, 0.72, 0.86, -1.07, -1.66, -5.87, -0.66, -2.49, -0.3, -0.5, 1.64, -0.72, 1.75, 2.73, -2.2, 0.9], 
 'Y': [0.79, -2.62, 4.11, -0.63, 1.89, -0.53, -1.3, 1.31, -0.56, -0.95, 1.91, -1.26, 1.57, 0.2, -0.76, -5.19, -2.56, 2.87, -3.43]
 }

#translate ntseq into aaseq
#will give an error is ntseq is not a multiple of three
def translate(ntseq):
    aaseq=""
    for i in range(0, len(ntseq), 3): 
        aaseq+=codon_to_aa[ntseq[i:i+3]]
    return aaseq

#takes an array of nucleotide sequences and computes a dictionary 
#with aas as keys and as values the ntseqs that encode the aas
#returns the number of keys in this dict
def total_aas_encoded(ntseqarr):
    aas_to_nts={}
    for nts in ntseqarr:
        aas=translate(nts)
        try:
            aas_to_nts[aas].append(nts)
        except KeyError:
            aas_to_nts[aas]=[nts]

    #aas_arr=sorted(aas_to_nts.keys())
    naas=len(sorted(aas_to_nts.keys()))
    
    #return number of amino acid sequences, from which the number of aas
    #per nucleotide sequences can be computed
    return naas


def euclidean_distance(vx, vy):
    return sum((y-x)**2 for x, y in zip(vx, vy)) ** 0.5
  


####################################################
# utils to calculate percent increase or decrease
####################################################

def perc_inc(pr_new, pr_old):
    return 100*(pr_new-pr_old)/pr_old

def perc_red(pr_new, pr_old):
    return 100*(pr_old-pr_new)/pr_old

    
#########################################################
#### functions for binary classification
#########################################################
#calculates true and false negative predictions on whether a 
#predicted fitness is above or below a viability threshold
#passed arrays real_fit and pred_fit must have the same size
#and contain floating point fitness value
def classify_fitness(real_fit, pred_fit, viab_thresh):
    #four variables for true and false positive (viable) predictions
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(real_fit)):
        if real_fit[i]>=viab_thresh and pred_fit[i]>=viab_thresh:
            TP+=1
        elif real_fit[i]<viab_thresh and pred_fit[i]>=viab_thresh:
            FP+=1
        elif real_fit[i]<viab_thresh and pred_fit[i]<viab_thresh:
            TN+=1
        elif real_fit[i]>=viab_thresh and pred_fit[i]<viab_thresh:
            FN+=1
    #sensitivity and specificity
    sens=TP/(TP+FN)
    spec=TN/(TN+FP)
    return [TP/len(real_fit), FP/len(real_fit), TN/len(real_fit), FN/len(real_fit), sens, spec]

#like classify_fitness, but passed arrays are binary arrays of 
#whether a genotype is viable (1) or not((0))
#calculates true and false negative predictions on whether a 
#predicted fitness is above or below a viability threshold
#passed arrays real_fit and pred_fit must have the same size
#and contain floating point fitness value
def classify_fitness_binary(real_v, pred_v):
    #four variables for true and false positive (viable) predictions
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(real_v)):
        if real_v[i]==1 and pred_v[i]==1:
            TP+=1
        elif real_v[i]==0 and pred_v[i]==1:
            FP+=1
        elif real_v[i]==0 and pred_v[i]==0:
            TN+=1
        elif real_v[i]==1 and pred_v[i]==0:
            FN+=1
    #sensitivity and specificity
    sens=TP/(TP+FN)
    spec=TN/(TN+FP)
    return [TP/len(real_v), FP/len(real_v), TN/len(real_v), FN/len(real_v), sens, spec]





######################################################################################
### functions to prepare the whole dhfr data set data with one hot encoding,flattened or otherwise
### and with integer encoding
######################################################################################

#takes three dictionaries with ntseq as keys (aafit, fit, sefit)
#that contain aas, fitness, and standard error in fitness
#and have ntseqs as keys
#one-hot encodes ntseq and aaseq with (flattenflag==True) 
#or without (flattenflag = False) flattening
#shifts all fitness values by fitshift
#writes four arrays (for ntseqs, aaseqs, fit, and sefit) from the dictionaries and randomly shuffles their 
#entries with the same randsom permutation for all of them
#subdivides a fraction of each into training, validation and test sets, according to f_tr, f_va, f_te
#subdivides each data set further according to whether fithess is above hilothresh or below
def prep_dhfr_data_onehot_ran(aaseq, fit, sefit, flattenflag, fitshift, f_tr, f_va, f_te, hilothresh):
    
    alldat_ntseq=[]
    alldat_aaseq=[]
    alldat_fit=[]
    alldat_sefit=[]
    if flattenflag==True:        
        for ntseq in fit.keys():
            alldat_ntseq.append(onehot_DNA_flat(ntseq))
            #alldat_ntseq.append(ntseq)
            alldat_aaseq.append(onehot_prot_flat(aaseq[ntseq]))
            alldat_fit.append(fit[ntseq])
            alldat_sefit.append(sefit[ntseq])
    elif flattenflag==False: 
        for ntseq in fit.keys():
            alldat_ntseq.append(ntseq)
            #alldat_ntseq.append(ntseq)
            alldat_aaseq.append(aaseq[ntseq])
            alldat_fit.append(fit[ntseq])
            alldat_sefit.append(sefit[ntseq])
        alldat_ntseq=onehot_DNA(alldat_ntseq)
        #also one hot encode aa seq even thougb we currently do not use them
        #andrei's data contains a separate character for stop codons
        alldat_aaseq=onehot_prot_w_stop(alldat_aaseq)  
    else:
        print("error_aw: wrong encoding flag")

    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)


    print("min/max of fitness values", np.min(alldat_fit), np.max(alldat_fit) )

    #shift all fitness values by a constant
    for i in range(len(alldat_fit)):
        alldat_fit[i] += fitshift
    print("min/max of fitness values after transformation", np.min(alldat_fit), np.max(alldat_fit) )


    #since the sequences may have been ordered, randomly permute the 
    #data before using them
    ranper=np.random.permutation(len(alldat_fit))
    alldat_ntseq=[alldat_ntseq[ranper[i]] for i, ntseq in enumerate(alldat_ntseq)]
    alldat_aaseq=[alldat_aaseq[ranper[i]] for i, aaseq in enumerate(alldat_aaseq)]
    alldat_fit=[alldat_fit[ranper[i]] for i, ntseq in enumerate(alldat_fit)]
    alldat_sefit=[alldat_sefit[ranper[i]] for i, ntseq in enumerate(alldat_sefit)]

    #cast to np arrays or the subsetting below won't work
    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)

    alldat_ntseq_hi=alldat_ntseq[alldat_fit>hilothresh]
    alldat_aaseq_hi=alldat_aaseq[alldat_fit>hilothresh]
    alldat_fit_hi=alldat_fit[alldat_fit>hilothresh]
    alldat_sefit_hi=alldat_sefit[alldat_fit>hilothresh]

    alldat_ntseq_lo=alldat_ntseq[alldat_fit<=hilothresh]
    alldat_aaseq_lo=alldat_aaseq[alldat_fit<=hilothresh]
    alldat_fit_lo=alldat_fit[alldat_fit<=hilothresh]
    alldat_sefit_lo=alldat_sefit[alldat_fit<=hilothresh]

    #print("hi fitness subset as percentage of whole ", 100*len(alldat_fit_hi)/len(alldat_fit))
    #print("lo fitness subset as percentage of whole ", 100*len(alldat_fit_lo)/len(alldat_fit))

    #extract  a training, validation and test set from the sequences
    trainfrac=f_tr 
    valfrac=f_va
    testfrac=f_te
    if (trainfrac+valfrac + testfrac) != 1: 
        print("\nerror_aw: invalid data set partition")

    tmpi=int(np.floor((trainfrac)*len(alldat_fit)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit)))

    traindat_ntseq=alldat_ntseq[:tmpi]
    traindat_aaseq=alldat_aaseq[:tmpi]
    traindat_fit=alldat_fit[:tmpi]
    traindat_sefit=alldat_sefit[:tmpi]

    valdat_ntseq=alldat_ntseq[tmpi+1:tmpj]
    valdat_aaseq=alldat_aaseq[tmpi+1:tmpj]
    valdat_fit=alldat_fit[tmpi+1:tmpj]
    valdat_sefit=alldat_sefit[tmpi+1:tmpj]

    testdat_ntseq=alldat_ntseq[tmpj+1:]
    testdat_aaseq=alldat_aaseq[tmpj+1:]
    testdat_fit=alldat_fit[tmpj+1:]
    testdat_sefit=alldat_sefit[tmpj+1:]

    #now do the same for the high and low data subsets
    tmpi=int(np.floor((trainfrac)*len(alldat_fit_hi)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit_hi)))

    traindat_ntseq_hi=alldat_ntseq_hi[:tmpi]
    traindat_aaseq_hi=alldat_aaseq_hi[:tmpi]
    traindat_fit_hi=alldat_fit_hi[:tmpi]
    traindat_sefit_hi=alldat_sefit_hi[:tmpi]

    valdat_ntseq_hi=alldat_ntseq_hi[tmpi+1:tmpj]
    valdat_aaseq_hi=alldat_aaseq_hi[tmpi+1:tmpj]
    valdat_fit_hi=alldat_fit_hi[tmpi+1:tmpj]
    valdat_sefit_hi=alldat_sefit_hi[tmpi+1:tmpj]

    testdat_ntseq_hi=alldat_ntseq_hi[tmpj+1:]
    testdat_aaseq_hi=alldat_aaseq_hi[tmpj+1:]
    testdat_fit_hi=alldat_fit_hi[tmpj+1:]
    testdat_sefit_hi=alldat_sefit_hi[tmpj+1:]

    
    #now do the same for the high and low data subsets
    tmpi=int(np.floor((trainfrac)*len(alldat_fit_lo)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit_lo)))

    traindat_ntseq_lo=alldat_ntseq_lo[:tmpi]
    traindat_aaseq_lo=alldat_aaseq_lo[:tmpi]
    traindat_fit_lo=alldat_fit_lo[:tmpi]
    traindat_sefit_lo=alldat_sefit_lo[:tmpi]

    valdat_ntseq_lo=alldat_ntseq_lo[tmpi+1:tmpj]
    valdat_aaseq_lo=alldat_aaseq_lo[tmpi+1:tmpj]
    valdat_fit_lo=alldat_fit_lo[tmpi+1:tmpj]
    valdat_sefit_lo=alldat_sefit_lo[tmpi+1:tmpj]

    testdat_ntseq_lo=alldat_ntseq_lo[tmpj+1:]
    testdat_aaseq_lo=alldat_aaseq_lo[tmpj+1:]
    testdat_fit_lo=alldat_fit_lo[tmpj+1:]
    testdat_sefit_lo=alldat_sefit_lo[tmpj+1:]

    
    
    return [alldat_ntseq, alldat_aaseq,alldat_fit,alldat_sefit,
            traindat_ntseq,traindat_aaseq,traindat_fit,traindat_sefit,
            valdat_ntseq,valdat_aaseq,valdat_fit,valdat_sefit,
            testdat_ntseq,testdat_aaseq,testdat_fit,testdat_sefit,
            traindat_ntseq_lo,traindat_aaseq_lo,traindat_fit_lo,traindat_sefit_lo,
            valdat_ntseq_lo,valdat_aaseq_lo,valdat_fit_lo,valdat_sefit_lo,
            testdat_ntseq_lo,testdat_aaseq_lo,testdat_fit_lo,testdat_sefit_lo,
            traindat_ntseq_hi,traindat_aaseq_hi,traindat_fit_hi,traindat_sefit_hi,
            valdat_ntseq_hi,valdat_aaseq_hi,valdat_fit_hi,valdat_sefit_hi,
            testdat_ntseq_hi,testdat_aaseq_hi,testdat_fit_hi,testdat_sefit_hi]


#takes three dictionaries with ntseq as keys (aafit, fit, sefit)
#that contain aas, fitness, and standard error in fitness
#and have ntseqs as keys
#encodes ntseq and aaseq with an arbitrary integer encoding in preparation
#for an embedding layer
#shifts all fitness values by fitshift
#writes three arrays from each dictionary and randomly shuffles their 
#entries with the same randsom permutation for all of them
#subdivides into training, validation and test set according to f_tr, f_va, f_te
#subdivides each data set further according to whether fithess is above hilothresh or below
def prep_dhfr_data_int_ran(aaseq, fit, sefit, fitshift, f_tr, f_va, f_te, hilothresh):

    
    # a sequence of length L corresponds to an 1D array of length L

    aa_to_int = {"A":0, "C":1, "D":2, "E":3, "F":4, 
                 "G":5,"H":6, "I":7, "K":8, "L":9, 
                 "M":10, "N":11, "P":12, "Q":13, "R":14, 
                 "S":15, "T":16, "V":17, "W":18, "Y":19, "*": 20}
    dna_to_int = {"A":0, "C":1, "G":2, "T":3} 

    alldat_ntseq=[]
    alldat_aaseq=[]
    alldat_fit=[]
    alldat_sefit=[]
       
    for ntseq in fit.keys():
        alldat_ntseq.append([dna_to_int[x] for x in ntseq])
        alldat_aaseq.append([aa_to_int[x] for x in aaseq[ntseq]])
        alldat_fit.append(fit[ntseq])
        alldat_sefit.append(sefit[ntseq])


    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)


    print("min/max of fitness values", np.min(alldat_fit), np.max(alldat_fit) )
    #problem here: fitness values very close to zero may yield very large mape errors,
    #so shift all fitness values by a constant
    for i in range(len(alldat_fit)):
        alldat_fit[i] += fitshift
    print("min/max of fitness values after transformation", np.min(alldat_fit), np.max(alldat_fit) )


    #since the sequences may have been ordered, randomly permute the 
    #data before using them
    ranper=np.random.permutation(len(alldat_fit))
    alldat_ntseq=[alldat_ntseq[ranper[i]] for i, ntseq in enumerate(alldat_ntseq)]
    alldat_aaseq=[alldat_aaseq[ranper[i]] for i, aaseq in enumerate(alldat_aaseq)]
    alldat_fit=[alldat_fit[ranper[i]] for i, ntseq in enumerate(alldat_fit)]
    alldat_sefit=[alldat_sefit[ranper[i]] for i, ntseq in enumerate(alldat_sefit)]

    #cast to np arrays or the subsetting below won't work
    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)

    alldat_ntseq_hi=alldat_ntseq[alldat_fit>hilothresh]
    alldat_aaseq_hi=alldat_aaseq[alldat_fit>hilothresh]
    alldat_fit_hi=alldat_fit[alldat_fit>hilothresh]
    alldat_sefit_hi=alldat_sefit[alldat_fit>hilothresh]

    alldat_ntseq_lo=alldat_ntseq[alldat_fit<=hilothresh]
    alldat_aaseq_lo=alldat_aaseq[alldat_fit<=hilothresh]
    alldat_fit_lo=alldat_fit[alldat_fit<=hilothresh]
    alldat_sefit_lo=alldat_sefit[alldat_fit<=hilothresh]

    print("hi fitness subset as percentage of whole ", 100*len(alldat_fit_hi)/len(alldat_fit))
    print("lo fitness subset as percentage of whole ", 100*len(alldat_fit_lo)/len(alldat_fit))

    #extract  a training, validation and test set from the sequences
    trainfrac=f_tr 
    valfrac=f_va
    testfrac=f_te
    if (trainfrac+valfrac + testfrac) != 1: 
        print("\nerror_aw: invalid data set partition")

    tmpi=int(np.floor((trainfrac)*len(alldat_fit)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit)))

    traindat_ntseq=alldat_ntseq[:tmpi]
    traindat_aaseq=alldat_aaseq[:tmpi]
    traindat_fit=alldat_fit[:tmpi]
    traindat_sefit=alldat_sefit[:tmpi]

    valdat_ntseq=alldat_ntseq[tmpi+1:tmpj]
    valdat_aaseq=alldat_aaseq[tmpi+1:tmpj]
    valdat_fit=alldat_fit[tmpi+1:tmpj]
    valdat_sefit=alldat_sefit[tmpi+1:tmpj]

    testdat_ntseq=alldat_ntseq[tmpj+1:]
    testdat_aaseq=alldat_aaseq[tmpj+1:]
    testdat_fit=alldat_fit[tmpj+1:]
    testdat_sefit=alldat_sefit[tmpj+1:]

    #now do the same for the high and low data subsets
    tmpi=int(np.floor((trainfrac)*len(alldat_fit_hi)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit_hi)))

    traindat_ntseq_hi=alldat_ntseq_hi[:tmpi]
    traindat_aaseq_hi=alldat_aaseq_hi[:tmpi]
    traindat_fit_hi=alldat_fit_hi[:tmpi]
    traindat_sefit_hi=alldat_sefit_hi[:tmpi]

    valdat_ntseq_hi=alldat_ntseq_hi[tmpi+1:tmpj]
    valdat_aaseq_hi=alldat_aaseq_hi[tmpi+1:tmpj]
    valdat_fit_hi=alldat_fit_hi[tmpi+1:tmpj]
    valdat_sefit_hi=alldat_sefit_hi[tmpi+1:tmpj]

    testdat_ntseq_hi=alldat_ntseq_hi[tmpj+1:]
    testdat_aaseq_hi=alldat_aaseq_hi[tmpj+1:]
    testdat_fit_hi=alldat_fit_hi[tmpj+1:]
    testdat_sefit_hi=alldat_sefit_hi[tmpj+1:]

    
    #now do the same for the high and low data subsets
    tmpi=int(np.floor((trainfrac)*len(alldat_fit_lo)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit_lo)))

    traindat_ntseq_lo=alldat_ntseq_lo[:tmpi]
    traindat_aaseq_lo=alldat_aaseq_lo[:tmpi]
    traindat_fit_lo=alldat_fit_lo[:tmpi]
    traindat_sefit_lo=alldat_sefit_lo[:tmpi]

    valdat_ntseq_lo=alldat_ntseq_lo[tmpi+1:tmpj]
    valdat_aaseq_lo=alldat_aaseq_lo[tmpi+1:tmpj]
    valdat_fit_lo=alldat_fit_lo[tmpi+1:tmpj]
    valdat_sefit_lo=alldat_sefit_lo[tmpi+1:tmpj]

    testdat_ntseq_lo=alldat_ntseq_lo[tmpj+1:]
    testdat_aaseq_lo=alldat_aaseq_lo[tmpj+1:]
    testdat_fit_lo=alldat_fit_lo[tmpj+1:]
    testdat_sefit_lo=alldat_sefit_lo[tmpj+1:]

    
    
    return [alldat_ntseq, alldat_aaseq,alldat_fit,alldat_sefit,
            traindat_ntseq,traindat_aaseq,traindat_fit,traindat_sefit,
            valdat_ntseq,valdat_aaseq,valdat_fit,valdat_sefit,
            testdat_ntseq,testdat_aaseq,testdat_fit,testdat_sefit,
            traindat_ntseq_lo,traindat_aaseq_lo,traindat_fit_lo,traindat_sefit_lo,
            valdat_ntseq_lo,valdat_aaseq_lo,valdat_fit_lo,valdat_sefit_lo,
            testdat_ntseq_lo,testdat_aaseq_lo,testdat_fit_lo,testdat_sefit_lo,
            traindat_ntseq_hi,traindat_aaseq_hi,traindat_fit_hi,traindat_sefit_hi,
            valdat_ntseq_hi,valdat_aaseq_hi,valdat_fit_hi,valdat_sefit_hi,
            testdat_ntseq_hi,testdat_aaseq_hi,testdat_fit_hi,testdat_sefit_hi]




#takes three dictionaries with ntseq as keys (aafit, fit, sefit)
#that contain aas, fitness, and standard error in fitness
#and have ntseqs as keys
#encodes ntseq and aaseq with an arbitrary integer encoding in preparation
#for an embedding layer, in addition, also tokenizes codons and encodes
#them with an integer from 1 to 64
#shifts all fitness values by fitshift
#writes four arrays (for ntseqs, aaseqs, fit, and sefit) from the dictionaries,
#and randomly shuffles their entries with the same random permutation for all of them
#subdivides into training, validation and test set according to f_tr, f_va, f_te
#subdivides each data set further according to whether fithess is above hilothresh or below
def prep_dhfr_data_int_codon_ran(aaseq, fit, sefit, fitshift, f_tr, f_va, f_te, hilothresh):
   
    # a sequence of length L corresponds to a 1D array of length L 
    
    
    alldat_ntseq=[]
    alldat_aaseq=[]
    alldat_codseq=[]
    alldat_fit=[]
    alldat_sefit=[]
       
    for ntseq in fit.keys():
        alldat_ntseq.append([dna_to_int[x] for x in ntseq])
        alldat_aaseq.append([aa_to_int[x] for x in aaseq[ntseq]])
        alldat_codseq.append([codon_to_int[ntseq[i:i+3]] for i in range(0, 7, 3)])   
        alldat_fit.append(fit[ntseq])
        alldat_sefit.append(sefit[ntseq])
        
    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_codseq=np.array(alldat_codseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)


    print("min/max of fitness values", np.min(alldat_fit), np.max(alldat_fit) )
    #shift all fitness values by a constant
    for i in range(len(alldat_fit)):
        alldat_fit[i] += fitshift
    print("min/max of fitness values after transformation", np.min(alldat_fit), np.max(alldat_fit) )


    #since the sequences may have been ordered, randomly permute the 
    #data before using them
    ranper=np.random.permutation(len(alldat_fit))
    alldat_ntseq=[alldat_ntseq[ranper[i]] for i, ntseq in enumerate(alldat_ntseq)]
    alldat_aaseq=[alldat_aaseq[ranper[i]] for i, aaseq in enumerate(alldat_aaseq)]
    alldat_codseq=[alldat_codseq[ranper[i]] for i, codseq in enumerate(alldat_codseq)]
    alldat_fit=[alldat_fit[ranper[i]] for i, ntseq in enumerate(alldat_fit)]
    alldat_sefit=[alldat_sefit[ranper[i]] for i, ntseq in enumerate(alldat_sefit)]

    #cast to np arrays or the subsetting below won't work
    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_codseq=np.array(alldat_codseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)

    alldat_ntseq_hi=alldat_ntseq[alldat_fit>hilothresh]
    alldat_aaseq_hi=alldat_aaseq[alldat_fit>hilothresh]
    alldat_codseq_hi=alldat_codseq[alldat_fit>hilothresh]
    alldat_fit_hi=alldat_fit[alldat_fit>hilothresh]
    alldat_sefit_hi=alldat_sefit[alldat_fit>hilothresh]

    alldat_ntseq_lo=alldat_ntseq[alldat_fit<=hilothresh]
    alldat_aaseq_lo=alldat_aaseq[alldat_fit<=hilothresh]
    alldat_codseq_lo=alldat_codseq[alldat_fit>hilothresh]
    alldat_fit_lo=alldat_fit[alldat_fit<=hilothresh]
    alldat_sefit_lo=alldat_sefit[alldat_fit<=hilothresh]

    print("hi fitness subset as percentage of whole ", 100*len(alldat_fit_hi)/len(alldat_fit))
    print("lo fitness subset as percentage of whole ", 100*len(alldat_fit_lo)/len(alldat_fit))

    #extract  a training, validation and test set from the sequences
    trainfrac=f_tr 
    valfrac=f_va
    testfrac=f_te
    if (trainfrac+valfrac + testfrac) != 1: 
        print("\nerror_aw: invalid data set partition")

    tmpi=int(np.floor((trainfrac)*len(alldat_fit)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit)))

    traindat_ntseq=alldat_ntseq[:tmpi]
    traindat_aaseq=alldat_aaseq[:tmpi]
    traindat_codseq=alldat_codseq[:tmpi]
    traindat_fit=alldat_fit[:tmpi]
    traindat_sefit=alldat_sefit[:tmpi]

    valdat_ntseq=alldat_ntseq[tmpi+1:tmpj]
    valdat_aaseq=alldat_aaseq[tmpi+1:tmpj]
    valdat_codseq=alldat_codseq[tmpi+1:tmpj]
    valdat_fit=alldat_fit[tmpi+1:tmpj]
    valdat_sefit=alldat_sefit[tmpi+1:tmpj]

    testdat_ntseq=alldat_ntseq[tmpj+1:]
    testdat_aaseq=alldat_aaseq[tmpj+1:]
    testdat_codseq=alldat_codseq[tmpj+1:]
    testdat_fit=alldat_fit[tmpj+1:]
    testdat_sefit=alldat_sefit[tmpj+1:]

    #now do the same for the high and low data subsets
    tmpi=int(np.floor((trainfrac)*len(alldat_fit_hi)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit_hi)))

    traindat_ntseq_hi=alldat_ntseq_hi[:tmpi]
    traindat_aaseq_hi=alldat_aaseq_hi[:tmpi]
    traindat_codseq_hi=alldat_codseq_hi[:tmpi]
    traindat_fit_hi=alldat_fit_hi[:tmpi]
    traindat_sefit_hi=alldat_sefit_hi[:tmpi]

    valdat_ntseq_hi=alldat_ntseq_hi[tmpi+1:tmpj]
    valdat_aaseq_hi=alldat_aaseq_hi[tmpi+1:tmpj]
    valdat_codseq_hi=alldat_codseq_hi[tmpi+1:tmpj]
    valdat_fit_hi=alldat_fit_hi[tmpi+1:tmpj]
    valdat_sefit_hi=alldat_sefit_hi[tmpi+1:tmpj]

    testdat_ntseq_hi=alldat_ntseq_hi[tmpj+1:]
    testdat_aaseq_hi=alldat_aaseq_hi[tmpj+1:]
    testdat_codseq_hi=alldat_codseq_hi[tmpj+1:]
    testdat_fit_hi=alldat_fit_hi[tmpj+1:]
    testdat_sefit_hi=alldat_sefit_hi[tmpj+1:]

    
    #now do the same for the high and low data subsets
    tmpi=int(np.floor((trainfrac)*len(alldat_fit_lo)))
    tmpj=int(np.floor((trainfrac+valfrac)*len(alldat_fit_lo)))

    traindat_ntseq_lo=alldat_ntseq_lo[:tmpi]
    traindat_aaseq_lo=alldat_aaseq_lo[:tmpi]
    traindat_codseq_lo=alldat_codseq_lo[:tmpi]
    traindat_fit_lo=alldat_fit_lo[:tmpi]
    traindat_sefit_lo=alldat_sefit_lo[:tmpi]

    valdat_ntseq_lo=alldat_ntseq_lo[tmpi+1:tmpj]
    valdat_aaseq_lo=alldat_aaseq_lo[tmpi+1:tmpj]
    valdat_codseq_lo=alldat_codseq_lo[tmpi+1:tmpj]
    valdat_fit_lo=alldat_fit_lo[tmpi+1:tmpj]
    valdat_sefit_lo=alldat_sefit_lo[tmpi+1:tmpj]

    testdat_ntseq_lo=alldat_ntseq_lo[tmpj+1:]
    testdat_aaseq_lo=alldat_aaseq_lo[tmpj+1:]
    testdat_codseq_lo=alldat_codseq_lo[tmpj+1:]
    testdat_fit_lo=alldat_fit_lo[tmpj+1:]
    testdat_sefit_lo=alldat_sefit_lo[tmpj+1:]

  
    
    return [alldat_ntseq, alldat_aaseq,alldat_codseq,alldat_fit,alldat_sefit,
            traindat_ntseq,traindat_aaseq,traindat_codseq, traindat_fit,traindat_sefit,
            valdat_ntseq,valdat_aaseq,valdat_codseq, valdat_fit,valdat_sefit,
            testdat_ntseq,testdat_aaseq,testdat_codseq, testdat_fit,testdat_sefit,
            traindat_ntseq_lo,traindat_aaseq_lo,traindat_codseq_lo,traindat_fit_lo,traindat_sefit_lo,
            valdat_ntseq_lo,valdat_aaseq_lo,valdat_codseq_lo,valdat_fit_lo,valdat_sefit_lo,
            testdat_ntseq_lo,testdat_aaseq_lo,testdat_codseq_lo,testdat_fit_lo,testdat_sefit_lo,
            traindat_ntseq_hi,traindat_aaseq_hi,traindat_codseq_hi,traindat_fit_hi,traindat_sefit_hi,
            valdat_ntseq_hi,valdat_aaseq_hi,valdat_codseq_hi,valdat_fit_hi,valdat_sefit_hi,
            testdat_ntseq_hi,testdat_aaseq_hi,testdat_codseq_hi,testdat_fit_hi,testdat_sefit_hi]



#passed datasets are aaseq, fit, sefit, which are dictionaries indexed by nucleotide sequences
#loads data into arrays and randomly shuffles them

#loads the data and sets a random fraction f_te for later testing

#from the remainder, samples and subdivides a fraction of f_tr_va
#into multiple training and validation
#data subsets for k-fold cross-validation

#note that f_tr_va and f_te need not add up to one
 
#sampling strategies for the tranign/validation data are set by text variable sampling mode 
#"random": random sampling, just take the first f_tr_va fraction of the shuffled array 

#"unique_aas": aims to sample only one nucleotide sequence per unique amino acid sequence
#if the desired sample size is too large, samples additional sequences, such that the 
#number of nt sequences per aa sequence is as small as possible 

#'two_syn_aas': aims to sample two synonymous nucleotide sequences per aas

#'maxdiv_nts': samples maximally diverse nucleotide sequences

#'maxdiv_aas_georgiev': samples maximally diverse aas according to the physcochemical 
#aa distance measure published by georgiev
 
#'max_codon_usage': samples aas with the best codon usage (using the most frequent e.coli codons)

#'NNK': samples aas with this codon compression
#'NNS': samples aas with this codon compression
#'NNT': samples aas with this codon compression
#'NNG': samples aas with this codon compression
#'NDT': samples aas with this codon compression
#'Tang': the Tang pattern from Azevedo-Rocha SciRep 2015
#does not work for the Papkou 2023 data because only ca. 3 percent of dhfr data fit it


#both one-hot and integer encodes (nt, aa, and codon-based) the data set to ensure
#that training, validation, and test data sets are the same, used
#for mixed architecture NN models

#one-hot encodes ntseq and aaseq with (flattenflag==True) or without (flattenflag = False)
#flattening

#returns dictionaries tr, va, te containing the data in the different encodings
def dhfr_sample_data_kfold_int_codon_onehot(sampling_mode, foldcross, aaseq, fit, sefit, f_tr_va, f_te, flattenflag):
    
    #first write all data to arrays and encode them
    #this should be done before the splitting of the data 
    #into different sets, becaue otherwise time will be wasted
    #to recode the data
    
    #arrays for the unencoded data
    alldat_ntseq=[]
    alldat_aaseq=[]
    alldat_fit=[]
    alldat_sefit=[]
    
    #arrays for the integer encoded data
    alldat_ntseq_int=[]
    alldat_aaseq_int=[]
    alldat_codseq_int=[]
    
       
    for ntseq in fit.keys():
        alldat_ntseq.append(ntseq)
        alldat_aaseq.append(aaseq[ntseq])
        alldat_fit.append(fit[ntseq])
        alldat_sefit.append(sefit[ntseq])
        
        alldat_ntseq_int.append([dna_to_int[x] for x in ntseq])
        alldat_aaseq_int.append([aa_to_int[x] for x in aaseq[ntseq]])
        alldat_codseq_int.append([codon_to_int[ntseq[i:i+3]] for i in range(0, 7, 3)])   
     
       
    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)
        
    alldat_ntseq_int=np.array(alldat_ntseq_int)
    alldat_aaseq_int=np.array(alldat_aaseq_int)
    alldat_codseq_int=np.array(alldat_codseq_int)
                               

    
    #now 1hot encode the data
    alldat_ntseq_1hot=[]
    alldat_aaseq_1hot=[]

    if flattenflag==True:        
        for ntseq in fit.keys():
            alldat_ntseq_1hot.append(onehot_DNA_flat(ntseq))
            alldat_aaseq_1hot.append(onehot_prot_flat(aaseq[ntseq]))
    elif flattenflag==False: 
        for ntseq in fit.keys():
            alldat_ntseq_1hot.append(ntseq)
            alldat_aaseq_1hot.append(aaseq[ntseq])
        alldat_ntseq_1hot=onehot_DNA(alldat_ntseq_1hot)
        #also one hot encode aa seq even though we currently do not use them
        #note that the papkou 2023 data contains a separate character for stop codons
        alldat_aaseq_1hot=onehot_prot_w_stop(alldat_aaseq_1hot)  
    else:
        print("error_aw: wrong encoding flag")
    
    #since the sequences may have been ordered, randomly permute the 
    #data before doing anything else
    ranper=np.random.permutation(len(alldat_fit))
    alldat_ntseq=[alldat_ntseq[ranper[i]] for i, ntseq in enumerate(alldat_ntseq)]
    alldat_aaseq=[alldat_aaseq[ranper[i]] for i, ntseq in enumerate(alldat_aaseq)]
    alldat_fit=[alldat_fit[ranper[i]] for i, ntseq in enumerate(alldat_fit)]
    alldat_sefit=[alldat_sefit[ranper[i]] for i, ntseq in enumerate(alldat_sefit)]

    
    alldat_ntseq_int=[alldat_ntseq_int[ranper[i]] for i, ntseq in enumerate(alldat_ntseq_int)]
    alldat_ntseq_1hot=[alldat_ntseq_1hot[ranper[i]] for i, ntseq in enumerate(alldat_ntseq_1hot)]

    alldat_aaseq_int=[alldat_aaseq_int[ranper[i]] for i, aaseq in enumerate(alldat_aaseq_int)]
    alldat_aaseq_1hot=[alldat_aaseq_1hot[ranper[i]] for i, aaseq in enumerate(alldat_aaseq_1hot)]
   
    alldat_codseq_int=[alldat_codseq_int[ranper[i]] for i, codseq in enumerate(alldat_codseq_int)]

    #cast to np arrays or the subsetting below won't work
    alldat_ntseq=np.array(alldat_ntseq)
    alldat_aaseq=np.array(alldat_aaseq)
    alldat_fit=np.array(alldat_fit)
    alldat_sefit=np.array(alldat_sefit)
    
    alldat_ntseq_int=np.array(alldat_ntseq_int)
    alldat_ntseq_1hot=np.array(alldat_ntseq_1hot)

    alldat_aaseq_int=np.array(alldat_aaseq_int)
    alldat_aaseq_1hot=np.array(alldat_aaseq_1hot)
 
    alldat_codseq_int=np.array(alldat_codseq_int)
    


    #next subdivide the data into training/validation and test set
    if f_tr_va+ f_te > 1: 
        print("\nerror_aw: invalid data set partition")
    #use the first fraction of the data as training/test set
    #and the last fraction as test set
    tmpi=int(np.floor(f_tr_va*len(alldat_fit)))
    tmpj=int(np.floor((1-f_te)*len(alldat_fit)))
    
    #whatever the sampling mode for the training/validation set is going to be
    #always use a random sample for the test set.
    te_ntseq=alldat_ntseq[tmpj+1:]
    te_aaseq=alldat_aaseq[tmpj+1:]
    te_fit=alldat_fit[tmpj+1:]
    te_sefit=alldat_sefit[tmpj+1:]
    te_ntseq_int=alldat_ntseq_int[tmpj+1:]
    te_ntseq_1hot=alldat_ntseq_1hot[tmpj+1:]
    te_aaseq_int=alldat_aaseq_int[tmpj+1:]
    te_aaseq_1hot=alldat_aaseq_1hot[tmpj+1:]   
    te_codseq_int=alldat_codseq_int[tmpj+1:]
    
    if sampling_mode=='random':
        #the arrays have already been shuffled, so now we need to take only
        #the appropriate number of sequences
        trva_ntseq=alldat_ntseq[:tmpi]
        trva_aaseq=alldat_aaseq[:tmpi]
        trva_fit=alldat_fit[:tmpi]
        trva_sefit=alldat_sefit[:tmpi]
        trva_ntseq_int=alldat_ntseq_int[:tmpi]
        trva_ntseq_1hot=alldat_ntseq_1hot[:tmpi]
        trva_aaseq_int=alldat_aaseq_int[:tmpi]
        trva_aaseq_1hot=alldat_aaseq_1hot[:tmpi]
        trva_codseq_int=alldat_codseq_int[:tmpi]
    #sample only a single nucleotide sequence per aas if possible 
    elif sampling_mode=='unique_aas':
        #a temporary nucleotide sequence array that contains all the sequences
        #that are not in the test set, and from which we will sample sequences
        #for the training and validation set
        #note that tmpj marks the beginning of the test set
        non_te_ntseq=alldat_ntseq[:tmpj]
        non_te_aaseq=alldat_aaseq[:tmpj]
        non_te_fit=alldat_fit[:tmpj]
        non_te_sefit=alldat_sefit[:tmpj]
        non_te_ntseq_int=alldat_ntseq_int[:tmpj]
        non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
        non_te_aaseq_int=alldat_aaseq_int[:tmpj]
        non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
        non_te_codseq_int=alldat_codseq_int[:tmpj]
        
        #compute the actual number of sequences to be sampled
        tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
        #compute the indices of the entries of the nucleotide sequence array to be sampled
        sample_index_arr=sample_one_codon_per_aa(non_te_ntseq, tr_va_samplesize)
        
        trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
        trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
        trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
        trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
        trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
        trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
        trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
        trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
        trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
     
    #aim to sample two synonymous nts per aas    
    elif sampling_mode=='two_syn_aas':
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            #compute the indices of the entries of the nucleotide sequence array to be sampled
            sample_index_arr= sample_two_syn_aas(non_te_ntseq, tr_va_samplesize)
            
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
            
                
        
    elif sampling_mode=='maxdiv_nts':
        
        #a temporary nucleotide sequence array that contains all the sequences
        #that are not in the test set, and from which we will sample sequences
        #for the training and validation set
        #note that tmpj marks the beginning of the test set
        non_te_ntseq=alldat_ntseq[:tmpj]
        non_te_aaseq=alldat_aaseq[:tmpj]
        non_te_fit=alldat_fit[:tmpj]
        non_te_sefit=alldat_sefit[:tmpj]
        non_te_ntseq_int=alldat_ntseq_int[:tmpj]
        non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
        non_te_aaseq_int=alldat_aaseq_int[:tmpj]
        non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
        non_te_codseq_int=alldat_codseq_int[:tmpj]
        
        #compute the actual number of sequences to be sampled
        tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
        #compute the indices of the entries of the nucleotide sequence array to be sampled
        sample_index_arr=sample_max_diverse_ntseq(non_te_ntseq, tr_va_samplesize)
        trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
        trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
        trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
        trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
        trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
        trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
        trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
        trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
        trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
        
    #sampling a maximally diverse set of aas where distance is measured by the 
    #georgiev 2009 embedding
    elif sampling_mode=='maxdiv_aas_georgiev':
         
         #a temporary nucleotide sequence array that contains all the sequences
         #that are not in the test set, and from which we will sample sequences
         #for the training and validation set
         #note that tmpj marks the beginning of the test set
         
         
         non_te_ntseq=alldat_ntseq[:tmpj]
         non_te_aaseq=alldat_aaseq[:tmpj]
         non_te_fit=alldat_fit[:tmpj]
         non_te_sefit=alldat_sefit[:tmpj]
         non_te_ntseq_int=alldat_ntseq_int[:tmpj]
         non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
         non_te_aaseq_int=alldat_aaseq_int[:tmpj]
         non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
         non_te_codseq_int=alldat_codseq_int[:tmpj]
         
         stopctr=0
         index_arr=[]
         for i, s in enumerate(non_te_aaseq):
            if '*' in s:
                stopctr+=1 
            else:
                index_arr.append(i)
         print("warning from 'dhfr_sample_data_kfold_int_codon_onehot': eliminated ",
              stopctr,
              "sequences with stop codons")
         
         #now eliminate all the sequences encoding a stop codon
         trva_ntseq=np.take(non_te_ntseq, index_arr, axis=0)
         trva_aaseq=np.take(non_te_aaseq, index_arr, axis=0)
         trva_fit=np.take(non_te_fit, index_arr, axis=0)
         trva_sefit=np.take(non_te_sefit, index_arr, axis=0)
         trva_ntseq_int=np.take(non_te_ntseq_int, index_arr, axis=0)
         trva_ntseq_1hot=np.take(non_te_ntseq_1hot, index_arr, axis=0)
         trva_aaseq_int=np.take(non_te_aaseq_int, index_arr, axis=0)
         trva_aaseq_1hot=np.take(non_te_aaseq_1hot, index_arr, axis=0)
         trva_codseq_int=np.take(non_te_codseq_int, index_arr, axis=0)    
              
            
         #compute the actual number of sequences to be sampled
         #chose not to compensate for the sequences lost due to stop codons, 
         #because if tr_va=0.5 and te=0.5 this would imply (illegal) sampling from the
         #test set
         tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
         #compute the indices of the entries of the nucleotide sequence array to be
         #sampled        
         sample_index_arr=sample_max_diverse_aaseq_georgiev(trva_ntseq, tr_va_samplesize)
         trva_ntseq=np.take(trva_ntseq, sample_index_arr, axis=0)
         trva_aaseq=np.take(trva_aaseq, sample_index_arr, axis=0)
         trva_fit=np.take(trva_fit, sample_index_arr, axis=0)
         trva_sefit=np.take(trva_sefit, sample_index_arr, axis=0)
         trva_ntseq_int=np.take(trva_ntseq_int, sample_index_arr, axis=0)
         trva_ntseq_1hot=np.take(trva_ntseq_1hot, sample_index_arr, axis=0)
         trva_aaseq_int=np.take(trva_aaseq_int, sample_index_arr, axis=0)
         trva_aaseq_1hot=np.take(trva_aaseq_1hot, sample_index_arr, axis=0)
         trva_codseq_int=np.take(trva_codseq_int, sample_index_arr, axis=0)    
        
        
    #random sample of amino acid sequences with the highest codon usage
    elif sampling_mode=='max_codon_usage':
            
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            #compute the indices of the entries of the nucleotide sequence array to be sampled
            sample_index_arr=sample_aas_codon_usage(non_te_ntseq, tr_va_samplesize)
            
        
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
     
    
    elif sampling_mode=='NNK':
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            
            sample_index_arr=[]
            for i, seq in enumerate(non_te_ntseq):
            #compute the indices of the entries of the nucleotide sequence array to be
            #sampled      
                if re.search(r"^([ACGT][ACGT][GT]){3}", seq): #NNK pattern    
                    sample_index_arr.append(i)
            
            if tr_va_samplesize>len(sample_index_arr):
                #in this case, use the whole sample index array
                print("warning_aw: could only sample ", len(sample_index_arr), "sequences")
            else:
                #now take a random sample of the index array that is of the right size
                sample_index_arr=np.random.choice(sample_index_arr, tr_va_samplesize, replace=False)
    
            
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
            
    elif sampling_mode=='NNS':
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            
            sample_index_arr=[]
            for i, seq in enumerate(non_te_ntseq):
            #compute the indices of the entries of the nucleotide sequence array to be sampled
                if re.search(r"^([ACGT][ACGT][CG]){3}", seq): #NNS pattern    
                    sample_index_arr.append(i)
            
            if tr_va_samplesize>len(sample_index_arr):
                #in this case, use the whole sample index array
                print("warning_aw: could only sample ", len(sample_index_arr), "sequences")
            else:
                #now take a random sample of the index array that is of the right size
                sample_index_arr=np.random.choice(sample_index_arr, tr_va_samplesize, replace=False)
    
            
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
            
    elif sampling_mode=='NNT':
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            
            sample_index_arr=[]
            for i, seq in enumerate(non_te_ntseq):
            #compute the indices of the entries of the nucleotide sequence array to be sampled
                if re.search(r"^([ACGT][ACGT][T]){3}", seq): #NNT pattern      
                    sample_index_arr.append(i)
            
            if tr_va_samplesize>len(sample_index_arr):
                #in this case, use the whole sample index array
                print("warning_aw: could only sample ", len(sample_index_arr), "sequences")
            else:
                #now take a random sample of the index array that is of the right size
                #sample_index_arr=random.sample(sample_index_arr, tr_va_samplesize)
                sample_index_arr=np.random.choice(sample_index_arr, tr_va_samplesize, replace=False)
    
            
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
            
    elif sampling_mode=='NNG':
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            
            sample_index_arr=[]
            for i, seq in enumerate(non_te_ntseq):
            #compute the indices of the entries of the nucleotide sequence array to be sampled
                if re.search(r"^([ACGT][ACGT][G]){3}", seq): #NNG pattern    
                    sample_index_arr.append(i)
            
            if tr_va_samplesize>len(sample_index_arr):
                #in this case, use the whole sample index array
                print("warning_aw: could only sample ", len(sample_index_arr), "sequences")
            else:
                #now take a random sample of the index array that is of the right size
                #sample_index_arr=random.sample(sample_index_arr, tr_va_samplesize)
                sample_index_arr=np.random.choice(sample_index_arr, tr_va_samplesize, replace=False)
    
            
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
                            
            
    elif sampling_mode=='NDT':
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            
            sample_index_arr=[]
            for i, seq in enumerate(non_te_ntseq):
            #compute the indices of the entries of the nucleotide sequence array to be
            #sampled      
                if re.search(r"^([ACGT][AGT]T){3}", seq): #NDT pattern   
                    sample_index_arr.append(i)
            
            if tr_va_samplesize>len(sample_index_arr):
                #in this case, use the whole sample index array
                print("warning_aw: could only sample ", len(sample_index_arr), "sequences")
            else:
                #now take a random sample of the index array that is of the right size
                #sample_index_arr=random.sample(sample_index_arr, tr_va_samplesize)
                sample_index_arr=np.random.choice(sample_index_arr, tr_va_samplesize, replace=False)
    
            
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
            
    
    elif sampling_mode=='Tang': #the Tang pattern from Azevedo-Rocha SciRep 2015
                                #only ca. 3 percent of dhfr from Papkou et al., 2023 fit it 
                    
            #a temporary nucleotide sequence array that contains all the sequences
            #that are not in the test set, and from which we will sample sequences
            #for the training and validation set
            #note that tmpj marks the beginning of the test set
            non_te_ntseq=alldat_ntseq[:tmpj]
            non_te_aaseq=alldat_aaseq[:tmpj]
            non_te_fit=alldat_fit[:tmpj]
            non_te_sefit=alldat_sefit[:tmpj]
            non_te_ntseq_int=alldat_ntseq_int[:tmpj]
            non_te_ntseq_1hot=alldat_ntseq_1hot[:tmpj]
            non_te_aaseq_int=alldat_aaseq_int[:tmpj]
            non_te_aaseq_1hot=alldat_aaseq_1hot[:tmpj]
            non_te_codseq_int=alldat_codseq_int[:tmpj]
            
            #compute the actual number of sequences to be sampled
            tr_va_samplesize=int(np.floor(f_tr_va*len(alldat_fit)))
            
            sample_index_arr=[]
            for i, seq in enumerate(non_te_ntseq):
            #compute the indices of the entries of the nucleotide sequence array to be
            #sampled      
                if re.search(r"^(([ACGT][AGT]T)|([ACG][AC]A)|(ATG)|(TGG)){3}", seq): #the Tang pattern from Azevedo-Rocha SciRep 2015                                                                         
                    sample_index_arr.append(i)
            
            if tr_va_samplesize>len(sample_index_arr):
                #in this case, use the whole sample index array
                print("warning_aw: could only sample ", len(sample_index_arr), "sequences")
            else:
                #now take a random sample of the index array that is of the right size
                #sample_index_arr=random.sample(sample_index_arr, tr_va_samplesize)
                sample_index_arr=np.random.choice(sample_index_arr, tr_va_samplesize, replace=False)
    
            
            trva_ntseq=np.take(non_te_ntseq, sample_index_arr, axis=0)
            trva_aaseq=np.take(non_te_aaseq, sample_index_arr, axis=0)
            trva_fit=np.take(non_te_fit, sample_index_arr, axis=0)
            trva_sefit=np.take(non_te_sefit, sample_index_arr, axis=0)
            trva_ntseq_int=np.take(non_te_ntseq_int, sample_index_arr, axis=0)
            trva_ntseq_1hot=np.take(non_te_ntseq_1hot, sample_index_arr, axis=0)
            trva_aaseq_int=np.take(non_te_aaseq_int, sample_index_arr, axis=0)
            trva_aaseq_1hot=np.take(non_te_aaseq_1hot, sample_index_arr, axis=0)
            trva_codseq_int=np.take(non_te_codseq_int, sample_index_arr, axis=0)
    else:
        print("error_aw in 'dhfr_sample_data_kfold_int_codon_onehot': invalid sampling strategy ")
        sys.exit()
   
    
    
    #now further subdivide the training/validation sets into 
    #k chunks for k-fold cross validation, // is floor division
    num_trva=len(trva_fit)//foldcross    
    
    #hashes indexed by the index of cross-validation for the various data subsets   
    tr_ntseq={} 
    tr_aaseq={}
    tr_fit={}    
    tr_sefit={}
    tr_ntseq_int={}
    tr_ntseq_1hot={}
    tr_aaseq_int={}
    tr_aaseq_1hot={}
    tr_codseq_int={}
    
    va_ntseq={} 
    va_aaseq={}
    va_fit={}    
    va_sefit={}
    va_ntseq_int={}
    va_ntseq_1hot={}
    va_aaseq_int={}
    va_aaseq_1hot={}
    va_codseq_int={}  
    
    
    
    
    
    for i in range(foldcross):
        delindices=[i for i in range(i * num_trva, (i+1) * num_trva)]

        tr_ntseq[i] = np.delete(trva_ntseq, delindices, axis=0)
        tr_aaseq[i] = np.delete(trva_aaseq, delindices, axis=0)
        tr_fit[i] = np.delete(trva_fit, delindices, axis=0)
        tr_sefit[i] = np.delete(trva_sefit, delindices, axis=0)
        tr_ntseq_int[i] = np.delete(trva_ntseq_int, delindices, axis=0)
        tr_ntseq_1hot[i] = np.delete(trva_ntseq_1hot, delindices, axis=0)
        tr_aaseq_int[i] = np.delete(trva_aaseq_int, delindices, axis=0)
        tr_aaseq_1hot[i] = np.delete(trva_aaseq_1hot, delindices, axis=0)
        tr_codseq_int[i] = np.delete(trva_codseq_int, delindices, axis=0)
                
        va_ntseq[i] = trva_ntseq[i * num_trva: (i + 1) * num_trva]
        va_aaseq[i] = trva_aaseq[i * num_trva: (i + 1) * num_trva]
        va_fit[i] = trva_fit[i * num_trva: (i + 1) * num_trva]
        va_sefit[i] = trva_sefit[i * num_trva: (i + 1) * num_trva]
        va_ntseq_int[i] = trva_ntseq_int[i * num_trva: (i + 1) * num_trva]
        va_ntseq_1hot[i] = trva_ntseq_1hot[i * num_trva: (i + 1) * num_trva]
        va_aaseq_int[i] = trva_aaseq_int[i * num_trva: (i + 1) * num_trva]
        va_aaseq_1hot[i] = trva_aaseq_1hot[i * num_trva: (i + 1) * num_trva]
        va_codseq_int[i] = trva_codseq_int[i * num_trva: (i + 1) * num_trva]
        
        
       
        
        #NOTE THAT each element of the tr and va dicts is itself a dictionary whose keys are the 
        #k indices of the k-fold cross-validation and the values are the arrays corresponding to
        #those data sets
        #In contrast, the te dictionary entries are themselves arrays, because there is only one test set 
        tr={"tr_ntseq": tr_ntseq, 
            "tr_aaseq": tr_aaseq, 
            "tr_fit": tr_fit, 
            "tr_sefit": tr_sefit, 
            "tr_ntseq_int": tr_ntseq_int, 
            "tr_ntseq_1hot": tr_ntseq_1hot, 
            "tr_aaseq_int": tr_aaseq_int, 
            "tr_aaseq_1hot": tr_aaseq_1hot,
            "tr_codseq_int": tr_codseq_int}
        va={"va_ntseq": va_ntseq, 
            "va_aaseq": va_aaseq, 
            "va_fit": va_fit, 
            "va_sefit": va_sefit, 
            "va_ntseq_int": va_ntseq_int, 
            "va_ntseq_1hot": va_ntseq_1hot, 
            "va_aaseq_int": va_aaseq_int, 
            "va_aaseq_1hot": va_aaseq_1hot,
            "va_codseq_int": va_codseq_int}
        te={"te_ntseq": te_ntseq, 
            "te_aaseq": te_aaseq, 
            "te_fit": te_fit, 
            "te_sefit": te_sefit, 
            "te_ntseq_int": te_ntseq_int, 
            "te_ntseq_1hot": te_ntseq_1hot, 
            "te_aaseq_int": te_aaseq_int, 
            "te_aaseq_1hot": te_aaseq_1hot,
            "te_codseq_int": te_codseq_int}
                
                
            
    return [tr, va, te] 



#takes a set of nts and corresponding aa sequences, computes the codon usage of 
#each aa sequence, and samples n_sample sequences from them, such that for each
#aa sequence the sequence with the highest codon usage is chosen first 

#does not exclude sequences with stop codons

#ntseq is an array of nucleotide sequences from which to sample
#returns the index of the sequences to be sampled from the array
def sample_aas_codon_usage(ntseq, n_sample):
    #do NOT reshuffle the sequences in here, or the mapping to the 
    #aa sequences etc. in the calling routine will be destroyed,
    #sequences should be randomized before this function is entered
    
    if n_sample>len(ntseq):
        print("error_aw: sample size too large")
        sys.exit()
        
    #first map each nt sequence to its index in the array, for easier later
    #reference
    ntseq_to_i={} 
    for i, snt in enumerate(ntseq):
        ntseq_to_i[snt]=i
    
    #take the codon usage statistics of E.coli and create a dictionary
    #of codons, such that the best codon for each amino acid has the codon usage value of one
    #and all others are below
    #also includes sequences with the stop codon
    cu_norm={}
    for aa in sorted(list(aa_to_int.keys())):
        #if aa == '*': continue
        maxcu=0
        for codon in codon_to_aa.keys():
            if codon_to_aa[codon]==aa:
               if codon_usage_coli[codon] > maxcu:
                   maxcu=codon_usage_coli[codon]
        for codon in codon_to_aa.keys():
            if codon_to_aa[codon]==aa:
                cu_norm[codon]=codon_usage_coli[codon]/maxcu
    
    
    #a dictionary indexed by ntseqs that holds the sum of the codon usage values
    #for the codons in the sequence, divided by the total number of codons in the 
    #sequence
    cu={}
    for nts in ntseq:
        cutmp=0
        for i in range(0, len(nts), 3):
            cutmp+=cu_norm[str(nts[i:i+3])]
        #normalize by the total number of codons in the sequence
        cu[nts]=cutmp/(len(nts)/3)
    
    
    #dict whose keys are amino acid sequences, each value is an array (!)of nucleotide sequences encoding 
    #this aa
    aaseq_dict={}
    for snt in ntseq:
        saa=translate(snt)
        #if this is the first time we see this aa
        if saa not in aaseq_dict: 
            aaseq_dict[saa]=[snt]
        else:
            aaseq_dict[saa].append(snt)
    unique_saa_list=list(aaseq_dict.keys())        
    n_unique_saa=len(unique_saa_list)
    
       
    #now we want to sort the list of coding sequences for each aas by their codon usage
    #sorting is such that the last element of each list is the one with the best
    #codon usage, can be extracted by pop command
    for aas in aaseq_dict.keys():
        #only need to sort if we have at least two elements
        if len(aaseq_dict[aas])>1: 
            #from
            #https://stackoverflow.com/questions/12987178/sort-a-list-based-on-dictionary-values-in-python
            aaseq_dict[aas]=sorted(aaseq_dict[aas], key=lambda x: cu[x])
    


    #an array of nucleotide sequences that point to the indices of the passed
    #nt seq array that are to be chosen
    ntseq_index=[]
    
    #if there are more unique amino acid sequences than samples, then
    #simply choose a random subset of them, without replacement
    if n_sample<=n_unique_saa:
        ranper=np.random.permutation(n_unique_saa)
        for i in range(n_sample):
            sampled_saa=unique_saa_list[ranper[i]]
            #now choose for the nucleotide sequences encoding the 
            #array the last sequence, which will be the one with the
            #highest codon usage and ramove it
            sampled_snt=aaseq_dict[sampled_saa].pop()
            ntseq_index.append(ntseq_to_i[sampled_snt])
    #the harder case where we need to sample more than one nucleotide sequence per 
    #amino acid sequence, sample such that we get for each sampled amino acid sequence
    #the nucleotide sequence that has not been sampled yet with the highest codon usage 
    else:
        samplectr=1
        while samplectr<n_sample:
        #second condition tests whether sample counter is a multiple of
        #number of unique amino acid sequences, if so, create a new random index array
        #to sample amino acid sequences
            if samplectr==1 or ((samplectr-1) % n_unique_saa) == 0:
                ranper=np.random.permutation(n_unique_saa)
    
            #now choose an amino acid without replacement   
            for i in range(n_unique_saa):
                sampled_saa=unique_saa_list[ranper[i]]   
                #choose the nts with the highest codon usage from the nucleotide sequence array for this
                #aa sequence, then pop the element from the array so that it can no longer be sampled          
                n_snt=len(aaseq_dict[sampled_saa])
                if n_snt==0: #we have already popped all elements of the 
                             #list so sample another amino acid sequence
                    continue
                else:
                    sampled_saa=unique_saa_list[ranper[i]]
                    #now choose for the nucleotide sequences encoding the 
                    #array the last sequence, which will be the one with the
                    #highest codon usage and ramove it
                    sampled_snt=aaseq_dict[sampled_saa].pop()
                    ntseq_index.append(ntseq_to_i[sampled_snt])
                    samplectr+=1
                    if samplectr==n_sample+1:
                        break  #break out of for loop, which gets us to the return statement
    
        
    return ntseq_index
                   
#takes a set of nt sequences and corresponding aa sequences and samples n_sample sequences from them, such that
#each aa sequences is represented only once, i.e., each codon is represented only once
#if n_sample is greater than the total number of unique aa sequences, then allow multiple aa
#sequences per codon, but such that the number of ntseq per amino acids is as small as possible
#ntseq is an array of nucleotide sequences from which to sample
#returns the index of the sequences to be sampled from the array


def sample_one_codon_per_aa(ntseq, n_sample):
    #do NOT reshuffle the sequences in here, or the mapping to the 
    #aa sequences etc. in the calling routine will be destroyed,
    #sequences should be randomized before this function is entered
    
    if n_sample>len(ntseq):
        print("error_aw: sample size too large")
        sys.exit()
        
    #first map each nt sequence to its index in the array, for easier later
    #reference
    ntseq_to_i={}
    for i, snt in enumerate(ntseq):
        ntseq_to_i[snt]=i
   
    
    #dict whose keys are amino acid sequences and whose
    #values are arrays (!) of the ntseqs that encode them in the data set
    aaseq_dict={}
    for snt in ntseq:
        saa=translate(snt)
        #if this is the first time we see this aa
        if saa not in aaseq_dict: 
            aaseq_dict[saa]=[snt]
        else:
            aaseq_dict[saa].append(snt)
    unique_saa_list=list(aaseq_dict.keys())        
    n_unique_saa=len(unique_saa_list)
    
    
    if n_sample>n_unique_saa:
        print("'sample_one_codon_per_aa': sample size is larger than no. of unique aa sequences")
    
    #an array of nucleotide sequences that point to the indices of the passed
    #nt seq array that are to be chosen
    ntseq_index=[]
    
    #if there are more unique amino acid sequences than samples, then
    #simply choose a random subset of them, without replacement
    if n_sample<=n_unique_saa:
        ranper=np.random.permutation(n_unique_saa)
        for i in range(n_sample):
            sampled_saa=unique_saa_list[ranper[i]]
            #now choose a random one among the nucleotide sequences
            #encoding this amino acid sequence
            sampled_snt=np.random.choice(aaseq_dict[sampled_saa])
            ntseq_index.append(ntseq_to_i[sampled_snt])
    #the harder case where we need to sample more than one nucleotide sequence per 
    #amino acid sequence, sample such that we get a minimal number of 
    #nucleotide sequences per amino acid sequence
    else:
        samplectr=1
        while samplectr<n_sample:
        #second condition tests whether sample counter is a multiple of
        #number of unique amino acid sequences, if so, create a new random index array
        #to sample amino acid sequences
            if samplectr==1 or ((samplectr-1) % n_unique_saa) == 0:
                ranper=np.random.permutation(n_unique_saa)
    
            #now choose an amino acid without replacement   
            for i in range(n_unique_saa):
                sampled_saa=unique_saa_list[ranper[i]]   
                #choose a random element from the nucleotide sequence arrays for this
                #aa sequence, then pop the element from the array so that it can no longer be sampled          
                n_snt=len(aaseq_dict[sampled_saa])
                if n_snt==0: #we have already popped all elements of the 
                            #list so sample another amino acid sequence
                    continue
                else:
                    k = np.random.randint(n_snt) # get random index
                    #a trick to avoid O(n) cost when popping from inside a list
                    aaseq_dict[sampled_saa][k], aaseq_dict[sampled_saa][-1] = aaseq_dict[sampled_saa][-1], aaseq_dict[sampled_saa][k]    # swap with the last element
                    #pop what is now the last element
                    sampled_snt = aaseq_dict[sampled_saa].pop()                                     # pop last element O(1)
                    ntseq_index.append(ntseq_to_i[sampled_snt])
                    samplectr+=1
                    if samplectr==n_sample+1:
                        break#break out of foor loop, which gets us to the return statement
    return ntseq_index
                   
                                
        

#takes a set of nt and corresponding aa sequences and samples n_sample sequences from them, such that
#each aa sequences is represented by two synonymous sequences wherever possible
#if n_sample is greater than the total number of aa sequences that meet this criterion, then allows
#multiple synonymous aa sequences per codon, but such that the number of synonymous sequences per amino acids is minimal
#ntseq is an array of nucleotide sequences from which to sample
#returns the index of the sequences to be sampled from the array


def sample_two_syn_aas(ntseq, n_sample):
    #do NOT reshuffle the sequences in here, or the mapping to the 
    #aa sequences etc. in the calling routine will be destroyed,
    #sequences should be randomized before this function is entered
    
    if n_sample>len(ntseq):
        print("error_aw: sample size too large")
        sys.exit()
        
    #first map each nt sequence to its index in the array, for easier later
    #reference
    ntseq_to_i={}
    for i, snt in enumerate(ntseq):
        ntseq_to_i[snt]=i
   
    
    #dict whose keys are amino acid sequences and whose
    #values are arrays (!) of the ntseqs that encode them in the data set
    aaseq_dict={}
    for snt in ntseq:
        saa=translate(snt)
        #if this is the first time we see this aa
        if saa not in aaseq_dict: 
            aaseq_dict[saa]=[snt]
        else:
            aaseq_dict[saa].append(snt)
    unique_saa_list=list(aaseq_dict.keys())        
    n_unique_saa=len(unique_saa_list)
    
    if n_sample>n_unique_saa:
        print("'sample_two_syn_aas': sample size is larger than twice the no. of unique aa sequences")
    
    #an array of nucleotide sequences that point to the indices of the passed
    #nt seq array that are to be chosen
    ntseq_index=[]
    
    
    samplectr=1
    #we will perform potentially multiple rounds of sampling without replacement
    sampling_round=0
    while samplectr<n_sample:
        #second condition tests whether sample counter is a multiple of
        #number of unique amino acid sequences, if so, create a new random index array
        #to sample amino acid sequences
        if samplectr==1 or ((samplectr-1) % n_unique_saa) == 0:
            ranper=np.random.permutation(n_unique_saa)
            sampling_round+=1

        #now choose an amino acid without replacement   
        for i in range(n_unique_saa):
            sampled_saa=unique_saa_list[ranper[i]]   
            #choose a random element from the nucleotide sequence arrays for this
            #aa sequence, then pop the element from the array so that it can no longer be sampled
            
            #first determine how many nts (not alrady sampled) encode this aas
            n_snt=len(aaseq_dict[sampled_saa])
            if n_snt==0: #we have already popped all elements of the 
                         #list so sample another amino acid sequence
                continue
            #there is only one nucleotide sequence encoding this amino acid sequence
            #there are two possibilities: 
            if n_snt==1:
                #first, there is only a single ntseq encoding this aa
                #in the passed nt seq array, in this case we should sample this one
                if sampling_round==1:
                    sampled_snt=aaseq_dict[sampled_saa].pop()    
                    ntseq_index.append(ntseq_to_i[sampled_snt])
                    samplectr+=1
                    if samplectr==n_sample+1:
                        break   #break out of foor loop, which takes us to the return statement
                #second, we have already sampled a pair of sequences in a previous sampling 
                #(recall that sampling is without replacement) and there is still one sequence left
                #in this case, sample the sequence too, because otherwise if the sample size is   
                #close to len(ntseq), the routine might get hung because not all sequences can be sampled
                else:
                    sampled_snt=aaseq_dict[sampled_saa].pop()    
                    ntseq_index.append(ntseq_to_i[sampled_snt])
                    samplectr+=1
                    if samplectr==n_sample+1:
                        break   #break out of foor loop, which gets us to the return statement
            #there are at least two sequences left to be sampled 
            else:
                #two possibilities: if we are in the first sampling round
                #sample two ntseqs, if we are not in the first sampling round (meaning two 
                #ntseqs have already been sampled for this aas), sample only one
               
                k = np.random.randint(n_snt) # get random index
                #a trick to avoid O(n) cost when popping from inside a list
                aaseq_dict[sampled_saa][k], aaseq_dict[sampled_saa][-1] = aaseq_dict[sampled_saa][-1], aaseq_dict[sampled_saa][k]    # swap with the last element
                #pop what is now the last element
                sampled_snt = aaseq_dict[sampled_saa].pop()
                ntseq_index.append(ntseq_to_i[sampled_snt])
                samplectr+=1
                if samplectr==n_sample+1:
                    break #break out of for loop, which gets us to the return statement
                    
                #sample a second sequence only of we are in the first sampling round
                if sampling_round==1:
                    #now repeat this process to sample the second sequence but keep in mind that
                    #the total number of nocleotide sequences to be sampled has increased by one
                    k = np.random.randint(n_snt-1) # get random index
                    #a trick to avoid O(n) cost when popping from inside a list
                    aaseq_dict[sampled_saa][k], aaseq_dict[sampled_saa][-1] = aaseq_dict[sampled_saa][-1], aaseq_dict[sampled_saa][k]    # swap with the last element
                    #pop what is now the last element
                    sampled_snt = aaseq_dict[sampled_saa].pop()    
                    ntseq_index.append(ntseq_to_i[sampled_snt])
                    samplectr+=1
                    if samplectr==n_sample+1:
                        break  #break out of for loop, which gets us to the return statement
                
    return ntseq_index
                   
                                
        
     
#takes a set of nt sequences and corresponding aa sequences and samples n_sample sequences from them, 
#such that the resulting set of sequences is maximally diverse

#calculates the mean location of all already sampled sequences in the 
#space of 1-hot encoded sequences, and chooses the next sequence to be
#sampled that is maximally distant from this location

#ntseq is an array of nucleotide sequences from which to sample
#returns the index of the sequences to be sampled from the array
def sample_max_diverse_ntseq(ntseq, n_sample):
    #do NOT reshuffle the sequences in here, or the mapping to the 
    #aa sequences etc. in the calling routine will be destroyed,
    #sequences should be randomized before this function is entered
    
    n_seq=len(ntseq)

    if n_sample>n_seq:
        print("error_aw: sample size too large")
        sys.exit()
         
    #next create a one-hot flattened vector of the ntseqs
    #this is better than integer encoding because the euklidean
    #distance is then like the hamming distance
    ntseq_1hot=[]
    #this array will map the one hot encoded sequence to its index in the passed file
    #problem is that the one hot encoded sequence cannot be a key to a hash, so
    #need to conver it into a string
    ntseq_1hot_to_seq_index={}
    for i, s in enumerate(ntseq):
        s1hot=onehot_DNA_flat(s)
        s1hotstr=''
        for bit in s1hot:
            s1hotstr+=str(int(bit))
        ntseq_1hot.append(s1hot)
        ntseq_1hot_to_seq_index[s1hotstr]=i
        
    
    #from now on work with the 1hot representation
    #first choose a single sampled sequence, in fact, choose its
    #index
    sampled_ntseq=[]
    ranindex=np.random.choice(np.arange(n_seq))
    sampled_ntseq.append(ntseq_1hot[ranindex])
    #the average position of the first sampled sequence in the one-hot sequence space
    #which is itself a vector in this space
    
    #copying is necessary here, because otherwise we would be modifying the old list
    ave_loc=ntseq_1hot[ranindex].copy()
    
    # now pop the sampled sequence from the one hot encoded array
    # swap with the last element, trick to avoid 
    # O(n) cost of popping an internal element
    ntseq_1hot[ranindex], ntseq_1hot[-1] = ntseq_1hot[-1], ntseq_1hot[ranindex]    
    ntseq_1hot.pop() 
    
  
    sctr=1
    #less than below, because we have already sampled a sequence above
    while sctr<n_sample:
        #first calculate the distance of every sequence still to be sampled from
        #the average location of the sampled sequence
        d_to_sample=[]
        for s1hot in ntseq_1hot: 
            d_to_sample.append(euclidean_distance(ave_loc, s1hot))
        #now identify the index of the sequence in the remaining one hot array where
        #this distance is greatest
        #note that argmax will only identify the first of these, it would
        #be more expensive to identify all and choose a random one, but
        #since the sequences were randomized outside this function, this should be acceptable
        maxdist_i=np.argmax(d_to_sample, axis=0)
        currseq=ntseq_1hot[maxdist_i]
        sampled_ntseq.append(currseq)
              
        #now update the average location of the sample, such that the 
        #average is the average over all sampled sequences
        for coord in range(len(ave_loc)):
            ave_loc[coord] = ave_loc[coord]*(sctr/(sctr+1)) + (currseq[coord]/(sctr+1))        
        
        #now eliminate the sampled sequence from the array of sequences to be sampled
        ntseq_1hot[maxdist_i], ntseq_1hot[-1] = ntseq_1hot[-1], ntseq_1hot[maxdist_i]   
        ntseq_1hot.pop() 
        
     
        sctr+=1  
    
    
    #now map all the sampled sequences to the index of the corresponding nucleotide 
    #sequence
    sample_indices=[]
    for s1hot in sampled_ntseq:
        #convert the sequence into a string for indexing
        s1hotstr=''
        for bit in s1hot:
            s1hotstr+=str(int(bit))
        sample_indices.append(ntseq_1hot_to_seq_index[s1hotstr])
    return sample_indices
    
    

    
#takes a set of ntseqs sequences, translates them,  and samples n_sample aa sequences from them,
#such that the resulting set of sequences is maximally diverse
#in terms of the aa distance of the sequences, when aas
#are represented with a feature vector using the georgiev 2009 encoding 

#does NOT sample sequences with a stop codon

#calculates the mean location of all already sampled sequences in the 
#space of sequences encoded as prescribed by georgiev 2009, and chooses the next sequence to be
#sampled such that it is maximally distant from this location, if there is more
#than one such sequence, it chooses an arbitrary one (the first one in the array of sequences)

#ntseq is an array of nucleotide sequences from which to sample
#do this on the level of ntseq because the mapping from aas to nts is not unique
#returns the index of the sequences to be sampled from the array
def sample_max_diverse_aaseq_georgiev(ntseq, n_sample):
    #do NOT reshuffle the sequences in here, or the mapping to the 
    #aa sequences etc. in the calling routine will be destroyed,
    #sequences should be randomized before this function is entered
    
    #note that the passed aaseq is not and should not be altered in this
    #function
    
    #notice also that there may be multiple aas with identical sequences
    #in the passed array, which is unavoidable if the analysis is done
    #on the aa level, this is why one needs to keep track of the nt sequences
    #that encode each aa sequences and do the sampling in parallel for them
    
    ntseq_to_seq_index={}
    for i, snt in enumerate(ntseq):
        ntseq_to_seq_index[snt]=i
    
    aaseq=[translate(s) for s in ntseq]
    
    #first screen for stop codons in any of the sequences
    for s in aaseq:
        if '*' in s:
            print("error in 'sample_max_diverse_aaseq_georgiev': one passed nts encodes stop codon")
            sys.exit()
    
    n_seq=len(aaseq)

    if n_sample>n_seq:
        print("error_aw: sample size too large")
        sys.exit()
         
    #next create a georgiev-encoded flattened vector of the aaseqs  
    aaseq_gg=[]
    
    #an array that will hold the nt sequences corresponding to each
    #amino acid sequence in the passed array
    ntseq_for_mapping=[]
    
    
    for i, s in enumerate(aaseq):
        sgg=georgiev_prot_flat([s])[0]
        aaseq_gg.append(sgg)
        
        ntseq_for_mapping.append(ntseq[i])
        
        
    #from now on work with the georgiev representation
    #first choose a single sampled sequence, in fact, choose its
    #index
    sampled_aaseq=[]
    sampled_ntseq=[]
    ranindex=np.random.choice(np.arange(n_seq))
    sampled_aaseq.append(aaseq_gg[ranindex])
    sampled_ntseq.append(ntseq_for_mapping[ranindex])
    
    
    #the average position of the first sampled sequence in the one-hot sequence space
    #which is itself a vector in this space
    #copying is necessary here, because otherwise we would be modifying the old list
    ave_loc=aaseq_gg[ranindex].copy()
    
    # now pop the sampled sequence from the one hot encoded array
    # swap with the last element, trick to avoid 
    # O(n) cost of popping an internal element
    aaseq_gg[ranindex], aaseq_gg[-1] = aaseq_gg[-1], aaseq_gg[ranindex]    
    aaseq_gg.pop() 
    #introduce a parallel change into the nt sequence vector used for mapping so that
    #the indices are preserved
    ntseq_for_mapping[ranindex], ntseq_for_mapping[-1] = ntseq_for_mapping[-1], ntseq_for_mapping[ranindex]        
    ntseq_for_mapping.pop()
    
        
    sctr=1
    #'<' below, because we have already sampled a sequence above
    while sctr<n_sample:
        #first calculate the distance of every sequence still to be sampled from
        #the average location of the sampled sequence
        d_to_sample=[]
        for s_gg in aaseq_gg: 
            d_to_sample.append(euclidean_distance(ave_loc, s_gg))
        #now identify the index of the sequence in the remaining one hot array where
        #this distance is greatest
        #note that argmax will only identify the first of these, it would
        #be more expensive to identify all and choose a random one, but
        #since the sequences were randomized outside this function, this shoudl be acceptable
        maxdist_i=np.argmax(d_to_sample, axis=0)
        currseq=aaseq_gg[maxdist_i]
        sampled_aaseq.append(currseq)
        sampled_ntseq.append(ntseq_for_mapping[maxdist_i])

              
        #now update the average location of the sample, such that the 
        #average is the average over all sampled sequences
        for coord in range(len(ave_loc)):
            ave_loc[coord] = ave_loc[coord]*(sctr/(sctr+1)) + (currseq[coord]/(sctr+1))        
        
        #now eliminate the sampled sequence from the array of sequences to be sampled
        aaseq_gg[maxdist_i], aaseq_gg[-1] = aaseq_gg[-1], aaseq_gg[maxdist_i]   
        aaseq_gg.pop() 
        
        ntseq_for_mapping[maxdist_i], ntseq_for_mapping[-1] = ntseq_for_mapping[-1], ntseq_for_mapping[maxdist_i]        
        ntseq_for_mapping.pop()
        

        sctr+=1  
    
    
    #now map all the sampled sequences back to the index of the corresponding passed amino acid 
    #sequence
    sample_indices=[]
    for snt in sampled_ntseq:
        #convert the sequence into a string for indexing
        sample_indices.append(ntseq_to_seq_index[snt])
    return sample_indices
    



#########################################################
### one hot encoding of sequences and sequence embedding
#########################################################

#takes a DNA string and one-hot encodes it as a single (!) array where
#A=(1, 0, 0, 0), C=(0,1,0,0) etc.
def onehot_DNA_flat(seq):
    #first convert DNA string into a string of 0123
    DNA_to_int={"A": 0, "C": 1, "G": 2, "T": 3}
    seq_to_list=[DNA_to_int[x] for x in seq]

    onehot=tf.keras.utils.to_categorical(seq_to_list, num_classes=4)
    #the following converts the data back from an ndarry to a list,
    #which is necessary for later steps where these array are concatenated
    #but is inefficient
    return onehot.flatten().tolist()




#same for protein  
def onehot_prot_flat(seq):
    #first convert protein string into a string of 0123s
    aa_to_int = {"A":0, "C":1, "D":2, "E":3, "F":4, "G":5,"H":6, "I":7, "K":8, 
                 "L":9, "M":10, "N":11, "P":12, "Q":13, "R":14, "S":15, "T":16, 
                 "V":17, "W":18, "Y":19, "*": 20}
    seq_to_list=[aa_to_int[x] for x in seq]

    onehot=tf.keras.utils.to_categorical(seq_to_list, num_classes=21)
    #the following converts the data back from an nd arry to a list,
    #which is necessary for later steps where these array are concatenated
    #but is inefficient
    return onehot.flatten().tolist()
    


#takes a python list of DNA strings and one-hot encodes each entry seq as a 2D array where
#A=(1, 0, 0, 0), C=(0,1,0,0) etc.
#returns a list
def onehot_DNA(seqlist):
    #first convert DNA string into a string of 0123
    seq_to_int=[]
    DNA_to_int={"A": 0, "C": 1, "G": 2, "T": 3}
    for seq in seqlist:
        seq_to_int.append([DNA_to_int[x] for x in seq])
        
    onehot=tf.keras.utils.to_categorical(seq_to_int, num_classes=4)
    return onehot.tolist()
  
def onehot_prot(seqlist):
    #first convert protein string into a string of 0123
    aa_to_int = {"A":0, "C":1, "D":2, "E":3, "F":4, "G":5,"H":6, "I":7, "K":8, "L":9, "M":10, "N":11, "P":12, "Q":13, "R":14, "S":15, "T":16, "V":17, "W":18, "Y":19, "*": 20}
    seq_to_int=[]
    for seq in seqlist:
        seq_to_int.append([aa_to_int[x] for x in seq])
    
    onehot=tf.keras.utils.to_categorical(seq_to_int, num_classes=21)
    return onehot.tolist()

#uses stop codon as 21st amino acid
def onehot_prot_w_stop(seqlist):
    #first convert protein string into a string of 0123
    aa_to_int = {"A":0, "C":1, "D":2, "E":3, "F":4, "G":5,"H":6, "I":7, "K":8, "L":9, "M":10, "N":11, "P":12, "Q":13, "R":14, "S":15, "T":16, "V":17, "W":18, "Y":19, "*": 20}
    seq_to_int=[]
    for seq in seqlist:
        seq_to_int.append([aa_to_int[x] for x in seq])
    
    onehot=tf.keras.utils.to_categorical(seq_to_int, num_classes=21)
    return onehot.tolist()
    


#encodes each amino acid in a sequence of length L
#as a 19-dimensional feature vector from Georgiev 2009, using data 
#in code published by Ofer, Bioinformatics 2015
#yields a flattened Lx19 vector for the amino acid
#NOTE: does not allow stop codons in the sequences 
def georgiev_prot_flat(seqlist):    
    seq_to_gg=[]
    for seq in seqlist:
        tmpgg=[]
        for aa in seq:
            if aa=='*':
                print('error_aw: sequence contained stop codon')
                sys.exit()
            else:
                tmpgg.append(ggencoding[aa])
        #now flatten the list (no flatten command in python, only in numpy)
        tmpgg = [item for sublist in tmpgg for item in sublist]
        seq_to_gg.append(tmpgg)
    return seq_to_gg
                      
        
    


#adapted from Chollet 2021, 11.24, p 347
#sequence length is the number of tokens, in my case the length of the aa or DNA sequence
#input_dim is the number of tokens (20 for aa, 4 for DNA)
#output dim is the dimension of the embedding space
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs) #allows to use functions of the layer class
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1] #the last dimension of the input shape, this would be the length of the aa or DNA sequence
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs) #for the embedding of the sequence
        embedded_positions = self.position_embeddings(positions) #for the embedding of the positions
        return embedded_tokens + embedded_positions #note that both representations are added

    #if not all sequences are of equal length, a mask can be applied to ignore 0 entries
    #not relevant for me, but left it in here, because of how it might interact with other layers
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    #for saving model with custom layers, see chollat p 344
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
    
  




#######################################################################################
### Now some hard-coded previously hypertuned and high quality NN models for sampling
#######################################################################################


#input data is a flattened 1hot encoded nucleotide sequence
def dense_stack_v1():
    units=32
    regu=0.0
    n_stacks=3
    dropout=0.0
    learn_rate=0.01
    
    inputs = keras.Input(shape=(36, ))
    x=layers.Dense(units = units, kernel_regularizer = regularizers.l2(regu), activation="relu") (inputs)
    stackin=layers.Dropout(dropout) (x)

    for i in range(n_stacks):
        x=layers.Dense(units = units, kernel_regularizer = regularizers.l2(regu), activation="relu") (stackin)
        x=layers.Dropout(dropout) (x)
        x=layers.Dense(units = units, kernel_regularizer = regularizers.l2(regu), activation="relu") (x)
        x=layers.Dropout(dropout) (x)  
        #a residual connection and normalization step
        x=tf.add(stackin, x)
        stackin = layers.LayerNormalization() (x)
    
    
    #in case we use mixed precision set the output layer to float 32 to prevent numerical
    #instability     
    outputs = layers.Dense(1, dtype='float32') (stackin)
    
    model = keras.Model(inputs, outputs)   
        
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learn_rate), 
                  loss="mse", 
                  metrics=["mae", "mape"])
    return model



#input is integer-encoded codon data, treated internally as if it was a 64-dimensional time series of length 3
#i.e., input data dimension is (3, None) 
def RNN_stack_w_pos_embed_v1():
   
    emb_dim=32
    num_RNN_layers=3
    RNNunits=16
    regu=0.0
    RNN_dropout=0.1
    learn_rate=0.01
   

    
    inputs = keras.Input(shape=(3,))
    embedded=PositionalEmbedding(sequence_length=3, input_dim=64, output_dim = emb_dim) (inputs)
    
    stackin=layers.Bidirectional(layers.LSTM(units=RNNunits,
                                            kernel_regularizer = regularizers.l2(regu),
                                            recurrent_regularizer = regularizers.l2(regu),
                                            recurrent_dropout=RNN_dropout, 
                                            return_sequences=True)) (embedded)
    
    for i in range(1, num_RNN_layers+1): #note that this runs unto num_RNN_layers+1-1=num_RNN_layers, 
                                         #so gives correctly the number of intermediate num_RNN_layers
        x=layers.Bidirectional(layers.LSTM(
                #units=units_RNN,
                units=RNNunits,
                kernel_regularizer = regularizers.l2(regu),
                recurrent_regularizer = regularizers.l2(regu),
                recurrent_dropout=RNN_dropout, 
                return_sequences=True)) (stackin)
        #a residual connection and normalization step
        x=tf.add(stackin, x)
        stackin = layers.BatchNormalization() (x) 

    #the last layer must have return  sequences = False
    x=layers.Bidirectional(layers.LSTM(
                units=RNNunits,
                kernel_regularizer = regularizers.l2(regu),
                recurrent_regularizer = regularizers.l2(regu),
                recurrent_dropout=RNN_dropout, 
                return_sequences=False)) (stackin)
    #cannot use residual connection here because of return_sequences=False -- dimensionality not preserved
    stackin = layers.BatchNormalization() (x) 
 
    #perhaps also add a stack of dense layers here?
    #in case we use mixed precision set the output layer to float 32 to prevent numerical
    #instability  
    outputs = layers.Dense(1, dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)   
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learn_rate), 
                  loss="mse", 
                  metrics=["mae", "mape"])
    return model

# a transformer that gave best results on tuning with dhfr_nns_tune_regr_v1 on Jun 20
def transf_cod_w_pos_embed_v1():
   
    emb_dim=32 
    n_heads=8
    subsp_dim=16
    dense_dim=16
    n_stacks=6
    learn_rate=0.001
   
    if n_stacks<1:
        print("error_aw: number of stacks invalid")
        exit(1)

    inputs = keras.Input(shape=(3,))
    embedded=PositionalEmbedding(sequence_length=3, input_dim=64, output_dim = emb_dim) (inputs)
    
    att_out = layers.MultiHeadAttention(num_heads=n_heads, key_dim=subsp_dim) (embedded, embedded, embedded)
    x=tf.add(embedded, att_out)
    dense_input = layers.LayerNormalization() (x) 
    x = layers.Dense(units = dense_dim, activation = 'relu') (dense_input)
    dense_output = layers.Dense(units = emb_dim, activation = 'relu') (x)
    #and a finalresidual connection and normalization step
    x=tf.add(dense_input, dense_output)
    stack_out = layers.LayerNormalization() (x)
    
    #now iterate this loop if there is more than one stack
    for i in range(n_stacks):
        att_out = layers.MultiHeadAttention(num_heads=n_heads, key_dim=subsp_dim) (stack_out, stack_out, stack_out)
        x=tf.add(stack_out, att_out)
        dense_input = layers.LayerNormalization() (x) 
        x = layers.Dense(units = dense_dim, activation = 'relu') (dense_input)
        dense_output = layers.Dense(units = emb_dim, activation = 'relu') (x)
        #and a final residual connection and normalization step
        x=tf.add(dense_input, dense_output)
        stack_out = layers.LayerNormalization() (x)
        
    
    stack_out=layers.Dropout(rate=0.1) (stack_out)
    
    # the last part here is no longer part of the transformer proper
    #flatten the layers for the final regression output
    x=layers.Flatten() (stack_out)
    outputs = layers.Dense(1) (x)
    
    model = keras.Model(inputs, outputs)   
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = learn_rate), 
                  loss="mse", 
                  metrics=["mae", "mape"])
    return model
    


