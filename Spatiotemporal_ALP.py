import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import normalize
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy.linalg import norm 
import sys

#This functions calculate the alphas
def calculate_alphas(lsts,alphas):
    res = []
    for i in range(len(lsts)):
        r = lsts[i]*alphas[i]
        res.append(r)
    return res

#This function calculates the sigmas for each spatial location
def extract_sigmas(x,sigmas):
    relevant_sigmas = []
    for i in range(len(sigmas)):
        relevant_sigmas.append(sigmas[i][x])
    return relevant_sigmas

#This function aims to find the K neighbors based on correlation
def find_k_cor(corr,k):
    cor_locations = []
    for i in range(len(corr)):
        lst = np.argsort(corr.iloc[i]).values.flatten().tolist()
        neighbors = lst[(-1-k):-1]
        cor_locations.append([i,neighbors])
    return cor_locations

#This function calculate the Epsilon value 
epsilon_factor = 4 
def calcEpsilon(X, dataList):
    #epsilon_factor - a parameter that controls the width of the Gaussian kernel  
    #Compute  the width of the Gaussian kernel based on the given dataset.  
    dist = []
    for x in X:
        dist.append([])
        for y in X:
            _x = np.array(dataList[x])
            _y = np.array(dataList[y])
            dist[x].append(LA.norm(_x - _y))
    temp = list(dist + np.multiply(np.identity(len(X)) ,max(max(dist))))
    mins = []
    for row in temp:
        small = sys.maxsize
        for el in row:
            if(el < small and el != 0):
                small = el
        mins.append(small)
    return max(mins) * epsilon_factor

def kernel_matrix(dataframe,sigma,is_zero):
    if is_zero == True:
        dataframe = dataframe.iloc[:,:-1].to_numpy() #creating numpy array
        pairwise_dists = pdist(dataframe, 'sqeuclidean') #calcuatlting distances using Pdist
        pairwise_dists = squareform(pairwise_dists) # pdist can be converted to a full distance matrix by using squareform
        kernel = np.exp(-pairwise_dists / sigma**2) 
        np.fill_diagonal(kernel, 0, wrap=True) #fill the main diagonal of the given array with zeros
        normalize_kernel = normalize(kernel, axis=1, norm='l1')
        return (normalize_kernel)
    elif is_zero == False:
        dataframe = dataframe.iloc[:,:-1].to_numpy() #creating numpy array
        pairwise_dists = pdist(dataframe, 'sqeuclidean') #calcuatlting distances using Pdist
        pairwise_dists = squareform(pairwise_dists) # pdist can be converted to a full distance matrix by using squareform
        kernel = np.exp(-pairwise_dists / sigma**2) 
        normalize_kernel = normalize(kernel, axis=1, norm='l1')
        return (normalize_kernel)     

#Spatiotemporal ALP training    
def LP_Fusion_Train(dataframes_lst,sigmas_lst,alphas,is_zero):
    f_multiscale = [] #f0,f1,f2...
    f_approx = []
    d = [] #d1(f-f0),d2(f-f1),d3(f-f2)...
    sigmas = []
    counter = 0
    f = dataframes_lst[0].iloc[:,-1].to_numpy() #label
    while counter <=20:
        if counter == 0:      
            #first iteration, computing kernel matrix (K0), f0 and d1.
            #first iteratin: f0 = approx = f*K0
             f0_lst = []
             for i in range(len(dataframes_lst)): 
                 k0_i = kernel_matrix(dataframes_lst[i],sigmas_lst[i],is_zero)
                 f0_i = k0_i.dot(f)
                 f0_lst.append(f0_i)
             res = calculate_alphas(f0_lst,alphas)
             f0_Fusion = [sum(x) for x in zip(*res)]
             di = f- f0_Fusion #d1
             f_multiscale.append(f0_Fusion)
             f_approx.append(f0_Fusion)
             d.append(di)
             sigmas.append(sigmas_lst)
             sigmas_lst = [x/2 for x in sigmas_lst]   
             sigmas.append(sigmas_lst)
             counter +=1       
        else:
            fi_lst = []
            for i in range(len(dataframes_lst)):
                ki = kernel_matrix(dataframes_lst[i],sigmas_lst[i],is_zero) 
                approxi = ki.dot(di)
                fi_lst.append(approxi)
            res = calculate_alphas(fi_lst,alphas)
            approx_i = [sum(x) for x in zip(*res)]
            f_approx.append(approx_i)
            fi_Fusion = [sum(i) for i in zip(*f_approx)]
            f_multiscale.append(fi_Fusion)
            di = f - fi_Fusion
            d.append(di)
            sigmas_lst = [x/2 for x in sigmas_lst] 
            sigmas.append(sigmas_lst)
            counter +=1
    err = [norm(i) for i in d]   
    return f_multiscale,d,err,f,sigmas

#Spatiotemporal ALP testing    
def LP_Fusion_Test(dataframes_train,dataframe_test,sigmas_lst,alphas,is_zero):
        f_multiscale,d,err,f,sigmas = LP_Fusion_Train(dataframes_train,sigmas_lst,alphas,is_zero) #training
        index_min = min(range(len(err)), key=err.__getitem__) #Optimal iteration
        predicted_values = []
        for sample in dataframe_test.values:
            fi_new = []
            for i in range(len(dataframes_train)):
                fij = []
                relevant_sigmas = extract_sigmas(i,sigmas) #return list of sigmas     
                sample = np.array(sample.tolist()).reshape(1,-1)
                pairwise_dists = cdist(sample,dataframes_train[i],'sqeuclidean').reshape(-1)
                for j in range(0,index_min+1):
                    if j ==0:
                        kernel_vector = np.exp(-(pairwise_dists)/(relevant_sigmas[j]**2))  
                        normalize_vector = kernel_vector/kernel_vector.sum(axis=0,keepdims=1)
                        if np.isnan([normalize_vector]).sum()>0:
                            normalize_vector = kernel_vector
                        else:
                            normalize_vector = normalize_vector
                        fij_new = normalize_vector.dot(f)
                        fij.append(fij_new)
                    else:
                        kernel_vector = np.exp(-(pairwise_dists)/(relevant_sigmas[j]**2))  
                        normalize_vector = kernel_vector/kernel_vector.sum(axis=0,keepdims=1)
                        if np.isnan([normalize_vector]).sum()>0:
                            normalize_vector = kernel_vector
                        else:
                            normalize_vector = normalize_vector
                        dij_new = normalize_vector.dot(d[j])
                        fij.append(dij_new)
                fi_new.append(fij)
            res = calculate_alphas(np.array(fi_new),alphas)
            res = [sum(x) for x in zip(*res)]
            final = sum(res)
            predicted_values.append(final)
        return predicted_values  