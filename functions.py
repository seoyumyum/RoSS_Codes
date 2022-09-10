#### Created by Hyungjoo Seo at University of Illinois at Urbana-Champaign --  08/24/2022 ######

import os
import numpy as np
import random
from numpy.linalg import*
import matplotlib.pyplot as plt
from numpy import r_, exp, cos, sin, pi, zeros, ones, hanning, sqrt, log, floor, reshape, mean
from scipy import signal
from numpy.fft import fft
import math
import time
import scipy.optimize as opt

import scipy.io as sio
import scipy.io.wavfile

from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import copy
get_ipython().run_line_magic('matplotlib', 'inline')
import librosa
import librosa.display
import itertools 


def Tshift(arr,tau,fs):
    num = (tau*fs).astype(int)
    arr=np.roll(arr,num)
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),0)
    elif num > 0:
         np.put(arr,range(num),0)
    return arr

def Tshift_Ups(arr,tau,fs,factor):
    arr_ups = signal.resample(arr, len(arr)*factor)
    fs_ups = fs*factor
    
    num = (tau*fs_ups).astype(int)
    
    arr_ups=np.roll(arr_ups,num)

    if num<0:
         np.put(arr_ups,range(len(arr_ups)+num,len(arr_ups)),0)
    elif num > 0:
         np.put(arr_ups,range(num),0)
         
    arr_ups_LPF = butter_lowpass_filter(arr_ups, fs, fs_ups, 10)        
    return arr_ups_LPF[::factor]

def ArrayVec_deg_WB(M,Angle_deg, freq, vp, d):
    phi = 2*np.pi*freq*d*np.sin(Angle_deg*np.pi/180)/vp
    a=[]
    for i in range(M):
        a.append(np.exp(i*-1j*phi))
    return np.asarray(a)

def ArrayVec_deg_WB_CentRef(M,Angle_deg, freq, vp,d):
    phi = 2*np.pi*freq*d*np.sin(Angle_deg*np.pi/180)/vp
    a=[]
    for i in range(M):
        a.append(np.exp(((-1)**i)*-1j*phi/2))
    return np.asarray(a)

def ArrayVec_deg_WB_COS(M,Angle_deg, freq, vp,d):
    phi = 2*np.pi*freq*d*np.cos(Angle_deg*np.pi/180)/vp
    a=[]
    for i in range(M):
        a.append(np.e**((i*1j*phi)))
    return np.asarray(a)



def deg2ind(deg_list):              #Convert angle in degree to index of Angle_Sweep   Updated on 15th Nov. 2020
    if type(deg_list) != list:
        deg_list = np.array([deg_list])
    index_list = np.array([])
    for deg in deg_list:
        index_list = np.append(index_list, (N_Angle-1)*(deg-theta_start)/(theta_end - theta_start))
    index_list = index_list.astype(int)
    return index_list



def MixGen_att(SigVec, Angle, M, fs_true, SNR, factor_ups, vp, d, Range):  ## Range is from X0 to source
    X = np.zeros((M,len(SigVec[0,:])))
    K = len(SigVec[:,0])
    ## Convolutive Mixing using upsampling 
    for i in range(M):
        for j in range(K):
            alpha = Range/(Range-i*d*np.cos(Angle[j]*np.pi/180))
            X[i,:] +=  alpha * Tshift_Ups(SigVec[j,:], -i*d*np.cos(Angle[j]*np.pi/180)/vp, fs_true, factor_ups)
            #X[i,:] += Tshift(SigVec[j,:], -i*d*np.cos(Angle[j]*np.pi/180)/vp, fs_true)

    Disp = 500

    # Noise vector
    #np.random.seed(0)
    SigMaxVar = np.max(np.var(SigVec,axis=1))/2
    noiseVec =  np.random.normal(0,np.sqrt(SigMaxVar/(10**(SNR/10))),size=(M,len(SigVec[0,:])))    #Noise in time domain is real

    X = X + noiseVec
    return X



'''
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #y = signal.lfilter(b, a, data)           #With actual delay
    y = signal.filtfilt(b,a, data)            #without delay
    return y
    
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    #y = signal.lfilter(b, a, data)           #With actual delay
    y = signal.filtfilt(b,a, data)            #without delay
    return y



def butter_bandpass(lowcut, highcut, fs, order=5):
    return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = signal.lfilter(b, a, data)
    y = signal.filtfilt(b, a, data) 
    return y
'''




def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, analog=False, btype='low', output='sos')
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=5):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = signal.sosfiltfilt(sos, data)    #Without actual delay
    #y = signal.sosfilt(sos, data)      #with delay          
    return y
    
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, analog=False, btype='high', output='sos')
    return sos

def butter_highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order=order)
    y = signal.sosfiltfilt(sos, data)    #Without actual delay
    #y = signal.sosfilt(sos, data)      #with delay          
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    #y = signal.sosfilt(sos, data)
    return y


def AoA_clip_number(data):
    if data>180:
        data -= 360
    if data<-180:
        data += 360
    return data
def AoA_clip(data):
    if type(data) != list and type(data) != np.ndarray:
        data = np.array([data])
    for i in range(len(data)):
        if data[i]>180:
            data[i] -= 360
        if data[i]<-180:
            data[i] += 360
    return data




def val_clip(data,d,vp):
    if type(data) != list and type(data) != np.ndarray:
        data = np.array([data])
    for i in range(len(data)):
        if data[i]>d/vp:
            data[i] = d/vp
        if data[i]<-d/vp:
            data[i] = -d/vp
    return data

def eval_SI_SDR(s_ref,s_est):
    s_tilda = np.vdot(s_ref,s_est)*s_ref/np.square(np.linalg.norm(s_ref,ord=2))
    e_tilda = s_est-s_tilda
    return 20*np.log10(np.linalg.norm(s_tilda,ord=2)/np.linalg.norm(e_tilda,ord=2))
def eval_SDR(s_ref,s_est):
    e = s_est-s_ref
    return 20*np.log10(np.linalg.norm(s_ref,ord=2)/np.linalg.norm(e,ord=2))
def eval_SNR(s_ref,s_est):
    s_tilda = np.vdot(s_ref,s_est)*s_ref/np.square(np.linalg.norm(s_ref,ord=2))
    e_tilda = s_est-s_tilda
    return 20*np.log10(np.linalg.norm(s_ref,ord=2)/np.linalg.norm(s_est-s_ref,ord=2))
def rej_outlier(data, m=2.5):
    return data[abs(data - np.mean(data,axis=0)) < m * np.std(data,axis=0)]




def Find_AlignRotAng(Angle_est, index_want):
    if len(Angle_est) > 2:
        k = index_want
        Theta_align = np.average(np.array(list(itertools.combinations(Angle_est, 2))),axis=1)
        Q = np.zeros((len(Theta_align)))
        for i in range(len(Theta_align)):
            Q_list = np.abs(np.abs(AoA_clip(Angle_est - Theta_align[i]))-np.abs(AoA_clip_number(Angle_est[k] - Theta_align[i])))
            #print(np.delete(Q_list,k))
            Q[i] = np.min(np.delete(Q_list,k))

        #print('pick index i :',np.argmax(Q))
        return Theta_align[np.argmax(Q)]
    #elif len(Angle_est) == 3: 


    #    return (np.sum(Angle_est)-Angle_est[index_want])/2

    elif len(Angle_est) == 2:
        return np.average(Angle_est) - 90
    elif len(Angle_est) == 1:
        return False
        


def getAoA(AoA_D_save, prom, height):
    AoA_space = np.linspace(-200,200,401)
    KDE_vec = np.zeros((len(AoA_space),len(AoA_D_save)))
    for i in range(len(AoA_D_save)):
        a = AoA_D_save[i].reshape(-1, 1)
        #kde = KernelDensity(kernel='exponential', bandwidth=2.5).fit(a)
        kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(a)
        KDE_vec[:,i] = np.exp(kde.score_samples(AoA_space.reshape(-1,1))/10)
        #plt.figure(0)
        #plt.plot(AoA_space, KDE_vec[:,i])
    f_score = np.sum(KDE_vec,axis=1)
    #plt.ylim([-50,0])
    plt.figure(1)
    plt.plot(AoA_space, f_score)
    #plt.ylim([-50,0])
    plt.grid()
    peaks, _ = signal.find_peaks(f_score, prominence = prom*np.max(f_score), height=height)
    print('height = ', height)
    print('estimated AoAs: ',-np.sort(-AoA_space[peaks]))

    Angle_est = -np.sort(-AoA_space[peaks])
    peak_prom = signal.peak_prominences(f_score, peaks, wlen=None)[0]
    return Angle_est, peak_prom   

def getScore(AoA_D_save, count):
    AoA_space = np.linspace(-200,200,401)
    KDE_vec = np.zeros((len(AoA_space),len(AoA_D_save)))
    for i in range(len(AoA_D_save)):
        a = AoA_D_save[i].reshape(-1, 1)
        #kde = KernelDensity(kernel='exponential', bandwidth=2.5).fit(a)
        kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(a)
        KDE_vec[:,i] = np.exp(kde.score_samples(AoA_space.reshape(-1,1))/10)
        #plt.figure(0)
        #plt.plot(AoA_space, KDE_vec[:,i])
    f_score = np.sum(KDE_vec,axis=1)
    #plt.ylim([-50,0])
    plt.figure(1)
    plt.plot(AoA_space, f_score)
    #plt.ylim([-50,0])
    plt.grid()

    return f_score

def smooth_1d(data, kernel_size=3):
    kernel = np.ones(kernel_size) / kernel_size
    smooth_1d = np.convolve(data, kernel, mode='same')
    return smooth_1d

######### For RoSS ###############
def delta_to_AoA(delta, vp, d, fs_MIC):
    return np.arccos(-1*(vp/d)*val_clip(delta/fs_MIC,d,vp))*180/np.pi

def AoA_to_delta(AoA, vp, d, fs_MIC):
    return -fs_MIC*np.cos(AoA*np.pi/180)*(d/vp)

def rot_in_del(delta, vp, d, fs_MIC, dtheta, sign, ):  # sign=1 -> positive, sign=-1 -> negative
    if sign == 1:
        theta_G = delta_to_AoA(delta, vp, d, fs_MIC) + dtheta
    if sign == -1:
        theta_G = -delta_to_AoA(delta, vp, d, fs_MIC) + dtheta
    return AoA_to_delta(theta_G, vp, d, fs_MIC)
def findarg_X(hist_X,delta):
    return np.argmin(np.abs(hist_X - delta))

def AoA_pair(A_1d_save, i_srt, i_end, prom, hist_X, vp, d, fs_MIC, dtheta_step):
    AoA_cand = []
    peaks, _ = signal.find_peaks(A_1d_save[i_srt], prominence = prom*np.max(A_1d_save[i_srt]))
    #print(hist_X[peaks])
    for i in range(len(peaks)):
        arg_p = findarg_X(hist_X,rot_in_del(hist_X[peaks[i]],  vp,d,fs_MIC, (i_end-i_srt)*dtheta_step, 1))
        arg_n = findarg_X(hist_X,rot_in_del(hist_X[peaks[i]],  vp,d,fs_MIC, (i_end-i_srt)*dtheta_step, -1))
        #print(hist_X[arg_p], A_1d_save[1][arg_p])
        #print(hist_X[arg_n], A_1d_save[1][arg_n])
        if A_1d_save[i_end][arg_p] < A_1d_save[i_end][arg_n]:
            AoA_L = delta_to_AoA(hist_X[peaks[i]], vp, d, fs_MIC)
        else:
            AoA_L = -delta_to_AoA(hist_X[peaks[i]], vp, d, fs_MIC)
        #print(AoA_L)
        if i_srt == 0: 
            AoA_cand.append(AoA_L)
        else:
            arg_p_b4 = findarg_X(hist_X,rot_in_del(hist_X[peaks[i]],  vp,d,fs_MIC, -dtheta_step, 1))
            arg_n_b4 = findarg_X(hist_X,rot_in_del(hist_X[peaks[i]],  vp,d,fs_MIC, -dtheta_step, -1))

            if A_1d_save[i_srt-1][arg_p] < A_1d_save[i_srt-1][arg_n]:
                AoA_cand.append(AoA_L + i_srt*dtheta_step)
            else:
                AoA_cand.append(AoA_L + i_srt*dtheta_step)

    AoA_cand = np.array(AoA_cand).flatten()
    return AoA_clip(AoA_cand)


def ITD_cosine(rot_ang_list, theta, vp, d, fs_MIC):
    return -1*(fs_MIC*d/vp)*np.cos((np.pi/180)*(rot_ang_list-theta))
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def whiten(X):
    # Calculate the covariance matrix
    Xcov = np.cov(X, rowvar=True, bias=True)
    EigVal, EigVec = np.linalg.eigh(Xcov)
    SigmaInv = np.diag(1/EigVal**0.5)
    whiteM = EigVec@SigmaInv@np.transpose(EigVec).conj()            ## ZCA implementation
    #whiteM = SigmaInv@np.transpose(EigVec)            
    Xw = whiteM@X
    return Xw, whiteM

def whiten_complex(X, K):
    # Calculate the covariance matrix
    M = len(X)
    if M>K: 
        Xcov = np.cov(X, rowvar=True, bias=True)
        EigVal, EigVec = np.linalg.eigh(Xcov)
        k = np.argsort(EigVal)
        E = EigVal[k]
        bl_nv = np.sqrt(E[-K:]-E[:-K].mean())
        bl = 1/bl_inv
        whiteM = np.diag(bl).dot(EigVec[:,k[-K:]].conj().T)   
        Xw = whiteM@X
    else:
        W_inv      = sqrtm.linalg.sqrtm((X.dot(X.conj().T))/len(t))
        whiteM  = inv(W_inv)
        Xw = whiteM@X
    return Xw, whiteM


def center(X):
    mean = np.mean(X, axis=1, keepdims=True)
    centered =  X - mean 
    return centered, mean