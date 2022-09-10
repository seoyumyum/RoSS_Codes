from __future__ import division
import pdb,os,time,warnings

from numpy.random import rand
from scipy.linalg import sqrtm
import numpy as np
from scipy.io.wavfile import read,write
from scipy import signal,stats 
import math
import numpy.ma as ma
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
import math
from math import log

import random
from numpy.linalg import*
from numpy import r_, exp, cos, sin, pi, zeros, ones, hanning, sqrt, log, floor, reshape, mean
from numpy.fft import fft
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import copy
import librosa
import librosa.display

import importlib
import functions as lib_HS 
importlib.reload(lib_HS)

import itertools 


#import tfanalysis as tfanalysis
#import tfsynthesis as tfsynthesis
#import twoDsmooth as twoDsmooth
#importlib.reload(tfanalysis)
#importlib.reload(tfsynthesis)
#importlib.reload(twoDsmooth)
#from tfanalysis import tfanalysis
#from tfsynthesis import tfsynthesis
#from twoDsmooth import twoDsmooth

from scipy import ndimage


#time-frequency analysis
#X is the time domain signal
#AWIN is an analysis window
#TIMESTEP is the # of samples between adjacent time windows.
#NUMFREQ is the # of frequency components per time point.
#
#TFMAT complex matrix time-freq representation

def tfanalysis(x,awin,timestep,numfreq):
    nsamp=x.size
    wlen=awin.size
    x=np.reshape(x,-1,'F')
    awin=np.reshape(awin,-1,'F') #make inputs go column-wise
    numtime=math.ceil((nsamp-wlen+1)/timestep)
    tfmat=np.zeros((numfreq,numtime+1))+0j
    sind=None
    for i in range(0,numtime):
        sind=((i)*timestep)
        tfmat[:,i]=fft(x[sind:(sind+wlen)]*awin,numfreq)
    i=numtime+1
    sind=((i)*timestep)
    lasts = min(sind,x.size-1)
    laste=min((sind+wlen),x.size-1)
    tfmat[:,-1]=fft(np.hstack((x[lasts:laste],np.zeros(wlen-(laste-lasts))))*awin,numfreq)
    return tfmat



################################################

#time-frequency synthesis
#TIMEFREQMAT is the complex matrix time-freq representation
#SWIN is the synthesis window
#TIMESTEP is the # of samples between adjacent time windows.
#NUMFREQ is the # of frequency components per time point.
#
#X contains the reconstructed signal.

def tfsynthesis(timefreqmat,swin,timestep,numfreq):
    timefreqmat=np.asarray(timefreqmat)
    swin=np.reshape(swin,-1,'F')
    winlen=swin.size
    (numfreq, numtime)=timefreqmat.shape
    ind=np.fmod(np.array(range(0,winlen)),numfreq)
    x=np.zeros(((numtime-1)*timestep+winlen))
    for i in range(0,numtime):
        temp=numfreq*np.real(ifft(timefreqmat[:,i]))
        sind=((i)*timestep)
        for i in range(0,winlen):
            x[sind+i]=x[sind+i]+temp[ind[i]]*swin[i]
    return x


###########################################


# Smoothening for better identification of the peaks in a graph. Could have used Gaussian Kernels to do the same
# but it seemed better visual effects were given when this algorithm was followed ( Again, based on original CASA495)
#MAT is the 2D matrix to be smoothed.
#KER is either
#(1)a scalar
#(2)a matrix which is used as the averaging kernel.

def twoDsmooth(mat,ker):
    try:
        len(ker)
        kmat = ker
        
    except:
        kmat = np.ones((ker,ker))
        kmat = kmat / pow(ker, 2)

    [kr,kc] = list(kmat.shape)
    if (kr%2 == 0):
        conmat = np.ones((2,1))
        kmat = signal.convolve2d(kmat,conmat,'symm','same')
        kr = kr + 1

    if (kc%2 == 0):
        conmat = np.ones((2,1))
        kmat = signal.convolve2d(kmat,conmat,'symm','same')
        kc = kc + 1

    [mr,mc] = list(mat.shape)
    fkr = math.floor(kr/2)
    fkc = math.floor(kc/2)
    rota = np.rot90(kmat,2)
    mat=signal.convolve2d(mat,rota,'same','symm')
    return mat

#########################################################
def getHist(x, fs, N_fft, timestep,p,q,d,smooth):
    eps=2.2204e-16
    numfreq = N_fft
    wlen = numfreq
    awin=np.hamming(wlen) #analysis window is a Hamming window Looks like Sine on [0,pi]
    x1=x[0,:]
    x2=x[1,:]
    ### normalizing needed? for now let's not.
    x1=x1/np.max(x1) # Dividing by maximum to normalise
    x2=x2/np.max(x2) # Dividing by maximum to normalise
    tf1=tfanalysis(x1,awin,timestep,numfreq) #time-freq domain
    tf2=tfanalysis(x2,awin,timestep,numfreq) #time-freq domain
    x1=np.asmatrix(x1)
    x2=np.asmatrix(x2)
    tf1=np.asmatrix(tf1)
    tf2=np.asmatrix(tf2)

    #removing DC component
    tf1=tf1[1:,:]
    tf2=tf2[1:,:]
    #eps is the a small constant to avoid dividing by zero frequency in the delay estimation

    #calculate pos/neg frequencies for later use in delay calc ??

    a=np.arange(1,((numfreq/2)+1))
    b=np.arange((-(numfreq/2)+1),0)
    freq=(np.concatenate((a,b)))*((2*np.pi)/numfreq) #freq looks like saw signal

    a=np.ones((tf1.shape[1],freq.shape[0]))
    freq=np.asmatrix(freq)
    a=np.asmatrix(a)
    for i in range(a.shape[0]):
        a[i]=np.multiply(a[i],freq)
    fmat=a.transpose()

    
    ####################################################

    #2.calculate alpha and delta for each t-f point
    #2) For each time/frequency compare the phase and amplitude of the left and
    #   right channels. This gives two new coordinates, instead of time-frequency 
    #   it is phase-amplitude differences.

    R21 = (tf2+eps)/(tf1+eps)
    #2.1HERE WE ESTIMATE THE RELATIVE ATTENUATION (alpha)
    a=np.absolute(R21) #relative attenuation between the two mixtures
    #alpha = a
    alpha=a-1./a #'alpha' (symmetric attenuation)
    #2.2HERE WE ESTIMATE THE RELATIVE DELAY (delta)
    delta = -(np.imag((np.log(R21)/fmat)))                  ### delta = delay * fs
    
    # imaginary part, 'delta' relative delay
    ####################################################

    # 3.calculate weighted histogram
    # 3) Build a 2-d histogram (one dimension is phase, one is amplitude) where 
    #    the height at any phase/amplitude is the count of time-frequency bins that
    #    have approximately that phase/amplitude.

    #p=1; q=0;
    #p=2; q=2;
    h1=np.power(np.multiply(np.absolute(tf1),np.absolute(tf2)),p) #refer to run_duet.m line 45 for this. 
                                                                  #It's just the python translation of matlab 
    h2=np.power(np.absolute(fmat),q)

    tfweight=np.multiply(h1,h2) #weights vector 
    maxa=0.7;
    maxd=1.25*fs*d/343;#histogram boundaries for alpha, delta

    abins=35;
    dbins=181#2*50;#number of hist bins for alpha, delta


    # only consider time-freq points yielding estimates in bounds
    amask=(abs(alpha)<maxa)&(abs(delta)<maxd)
    amask=np.logical_not(amask)
    alphavec = np.asarray(ma.masked_array(alpha, mask=(amask)).transpose().compressed())[0]
    deltavec = np.asarray(ma.masked_array(delta, mask=(amask)).transpose().compressed())[0]
    tfweight = np.asarray(ma.masked_array(tfweight, mask=(amask)).transpose().compressed())[0]
    # to do masking the same way it is done in Matlab/Octave, after applying a mask we must take transpose and compress

    #determine histogram indices (sampled indices?)

    alphaind=np.around((abins-1)*(alphavec+maxa)/(2*maxa))
    deltaind=np.around((dbins-1)*(deltavec+maxd)/(2*maxd))

    #FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
    #A(alphaind(k),deltaind(k)) = tfweight(k), S is abins-by-dbins
    A=sp.sparse.csr_matrix((tfweight, (alphaind, deltaind)),shape=(abins,dbins)).todense()
    #smooththehistogram-localaverage3-by-3neighboringbins

    A=twoDsmooth(A,smooth)
    X=np.linspace(-maxd,maxd,dbins)
    Y=np.linspace(-maxa,maxa,abins)

    #A_1d = np.max(A,axis=0)          #### 220207 -- changed from max to average to deal with reverb
    A_1d = np.average(A,axis=0)

 
    return A, A_1d, X, Y


########################################################################



################################################
#     setp 1,2,3
################################################ 
# 1. analyze the signals - STFT
# 1) Create the spectrogram of the Left and Right channels.

#constants used

### peak_mode =0 : prominence will be used for peak detection
### peak_mode =1 : height will be used.

def DUET(x, fs, N_fft, timestep,p,q,d,smooth,prom_const, max_const,peak_mode,graph_mode):
    eps=2.2204e-16
    numfreq = N_fft
    wlen = numfreq
    awin=np.hamming(wlen) 
    x1=x[0,:]
    x2=x[1,:]

    x1=x1/np.max(x1) 
    x2=x2/np.max(x2) 
    tf1=tfanalysis(x1,awin,timestep,numfreq) #time-freq domain
    tf2=tfanalysis(x2,awin,timestep,numfreq) #time-freq domain
    x1=np.asmatrix(x1)
    x2=np.asmatrix(x2)
    tf1=np.asmatrix(tf1)
    tf2=np.asmatrix(tf2)

    #removing DC component
    tf1=tf1[1:,:]
    tf2=tf2[1:,:]
    #eps is the a small constant to avoid dividing by zero frequency in the delay estimation


    a=np.arange(1,((numfreq/2)+1))
    b=np.arange((-(numfreq/2)+1),0)
    freq=(np.concatenate((a,b)))*((2*np.pi)/numfreq) #freq looks like saw signal

    a=np.ones((tf1.shape[1],freq.shape[0]))
    freq=np.asmatrix(freq)
    a=np.asmatrix(a)
    for i in range(a.shape[0]):
        a[i]=np.multiply(a[i],freq)
    fmat=a.transpose()

    
    ####################################################

    #2.calculate alpha and delta for each t-f point
    #2) For each time/frequency compare the phase and amplitude of the left and
    #   right channels. This gives two new coordinates, instead of time-frequency 
    #   it is phase-amplitude differences.

    R21 = (tf2+eps)/(tf1+eps)
    #2.1HERE WE ESTIMATE THE RELATIVE ATTENUATION (alpha)
    a=np.absolute(R21) #relative attenuation between the two mixtures
    alpha=a-1./a #'alpha' (symmetric attenuation)
    #2.2HERE WE ESTIMATE THE RELATIVE DELAY (delta)
    delta = -(np.imag((np.log(R21)/fmat)))                  ### delta = delay * fs
    
    # imaginary part, 'delta' relative delay
    ####################################################

    # 3.calculate weighted histogram
    # 3) Build a 2-d histogram (one dimension is phase, one is amplitude) where 
    #    the height at any phase/amplitude is the count of time-frequency bins that
    #    have approximately that phase/amplitude.

    #p=1; q=0;
    #p=2; q=2;
    h1=np.power(np.multiply(np.absolute(tf1),np.absolute(tf2)),p) 
                                                                  #It's just the python translation of matlab 
    h2=np.power(np.absolute(fmat),q)

    tfweight=np.multiply(h1,h2) #weights vector 
#    maxa=0.7;
#    maxd=2*3.6;#histogram boundaries for alpha, delta

#    abins=35;
#    dbins=2*50;#number of hist bins for alpha, delta

    maxa=0.7;
    maxd=1.25*fs*d/343;#histogram boundaries for alpha, delta

    abins=35;
    dbins=181#2*50;#number of hist bins for alpha, delta
    

    # only consider time-freq points yielding estimates in bounds
    amask=(abs(alpha)<maxa)&(abs(delta)<maxd)
    amask=np.logical_not(amask)
    alphavec = np.asarray(ma.masked_array(alpha, mask=(amask)).transpose().compressed())[0]
    deltavec = np.asarray(ma.masked_array(delta, mask=(amask)).transpose().compressed())[0]
    tfweight = np.asarray(ma.masked_array(tfweight, mask=(amask)).transpose().compressed())[0]
    # to do masking the same way it is done in Matlab/Octave, after applying a mask we must take transpose and compress

    #determine histogram indices 

    alphaind=np.around((abins-1)*(alphavec+maxa)/(2*maxa))
    deltaind=np.around((dbins-1)*(deltavec+maxd)/(2*maxd))

    #FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
    #A(alphaind(k),deltaind(k)) = tfweight(k), S is abins-by-dbins
    A=sp.sparse.csr_matrix((tfweight, (alphaind, deltaind)),shape=(abins,dbins)).todense()
    #smooththehistogram-localaverage3-by-3neighboringbins

    A=twoDsmooth(A,smooth)
    X=np.linspace(-maxd,maxd,dbins)
    Y=np.linspace(-maxa,maxa,abins)
    
    A_1d = np.average(A,axis=0)

    if peak_mode==0:
        col_peak,_=signal.find_peaks(A_1d,prominence=np.max(A_1d)*prom_const)
    else:
        col_peak,_=signal.find_peaks(A_1d,height=np.max(A_1d)*max_const)


    row_peak = []
    for idx in col_peak:
        row_peak.append(np.argmax(A[:,idx]))

    if graph_mode == 1:
        fig = plt.figure(0)
        ax = fig.add_subplot(111, projection='3d')
        X_grid, Y_grid = np.meshgrid(X, Y)
        ax.plot_wireframe(X_grid,Y_grid,A)
        plt.xlabel('delta')
        plt.ylabel('alpha')
        plt.tight_layout()
        plt.show()

        # You can have a look at the histogram to look at the local peaks and what not

        ## Peak location detection
        plt.figure(1)
        #plt.imshow(A,origin='upper')
        #plt.colorbar()
        #plt.figure(1)
        plt.plot(X,A_1d)
        plt.xlabel('delta')
        plt.tight_layout()
        plt.show()        
        
        
        
        
        
    ######################################    step 4,5,6,7
    ######################################
    #4.peak centers (determined from histogram) THIS IS DONE BY HUMAN.
    #4) Determine how many peaks there are in the histogram.
    #5) Find the location of each peak. 

    numsources=len(col_peak);

    peakdelta=X[col_peak]
    peakalpha=Y[row_peak]
    print('number of peaks: ', len(peakdelta))
    #convert alpha to a

    peaka=(peakalpha+np.sqrt(np.square(peakalpha)+4))/2;
    peaka=np.asarray(peaka)
 
    

    ##################################################
    #5.determine masks for separation
    #6) Assign each time-frequency frame to the nearest peak in phase/amplitude 
    #  space. This partitions the spectrogram into sources (one peak per source)

    test = float("inf")
    bestsofar=test*np.ones(tf1.shape)
    bestind=np.zeros(tf1.shape)

    for i in range(peakalpha.size):
        score = np.power(abs(np.multiply(peaka[i]*np.exp(-1j*fmat*peakdelta[i]),tf1)-tf2),2)/(1+peaka[i]*peaka[i])
        mask=score<bestsofar
        np.place(bestind,mask,i+1)
        s_mask=np.asarray(ma.masked_array(score, mask=np.logical_not(mask)).compressed())[0]
        np.place(bestsofar,mask,s_mask)

    ###################################################
    #6.&7.demix with ML alignment and convert to time domain
    #7) Then you create a binary mask (1 for each time-frequency point belonging to my source, 0 for all other points)
    #8) Mask the spectrogram with the mask created in step 7.
    #9) Rebuild the original wave file from 8.
    #10) Listen to the result.

    est=np.zeros((numsources,x1.shape[1]))
    (row,col)=bestind.shape
    for i in range(0,numsources):
        mask=ma.masked_equal(bestind,i+1).mask
        # here, 'h' stands for helper; we're using helper variables to break down the logic of
        # what's going on. Apologies for the order of the 'h's
        h1=np.zeros((1,tf1.shape[1]))
        h3=np.multiply((peaka[i]*np.exp(1j*fmat*peakdelta[i])),tf2)
        h4=((tf1+h3)/(1+peaka[i]*peaka[i]))
        h2=np.multiply(h4,mask)
        h=np.concatenate((h1,h2))

        esti=tfsynthesis(h,math.sqrt(2)*awin/1024,timestep,numfreq)

        #add back into the demix a little bit of the mixture
        #as that eliminates most of the masking artifacts

        est[i]=esti[0:x1.shape[1]]
        #write('out'+str(i),fs,np.asarray(est[i]+0.05*x1)[0])
    return est, peakdelta, A, A_1d



def IDUET_SS(x, fs, N_fft, timestep,p,q,d,smooth,col_peak): 
    if type(col_peak) != list and type(col_peak) != np.ndarray:
        col_peak = np.array([col_peak])
    eps=2.2204e-16
    numfreq = N_fft
    wlen = numfreq
    awin=np.hamming(wlen) 
    x1=x[0,:]
    x2=x[1,:]

    x1=x1/np.max(x1) 
    x2=x2/np.max(x2) 
    tf1=tfanalysis(x1,awin,timestep,numfreq) #time-freq domain
    tf2=tfanalysis(x2,awin,timestep,numfreq) #time-freq domain
    x1=np.asmatrix(x1)
    x2=np.asmatrix(x2)
    tf1=np.asmatrix(tf1)
    tf2=np.asmatrix(tf2)

    #removing DC component
    tf1=tf1[1:,:]
    tf2=tf2[1:,:]
    #eps is the a small constant to avoid dividing by zero frequency in the delay estimation


    a=np.arange(1,((numfreq/2)+1))
    b=np.arange((-(numfreq/2)+1),0)
    freq=(np.concatenate((a,b)))*((2*np.pi)/numfreq) #freq looks like saw signal

    a=np.ones((tf1.shape[1],freq.shape[0]))
    freq=np.asmatrix(freq)
    a=np.asmatrix(a)
    for i in range(a.shape[0]):
        a[i]=np.multiply(a[i],freq)
    fmat=a.transpose()

    
    ####################################################

    #2.calculate alpha and delta for each t-f point
    #2) For each time/frequency compare the phase and amplitude of the left and
    #   right channels. This gives two new coordinates, instead of time-frequency 
    #   it is phase-amplitude differences.

    R21 = (tf2+eps)/(tf1+eps)
    #2.1HERE WE ESTIMATE THE RELATIVE ATTENUATION (alpha)
    a=np.absolute(R21) #relative attenuation between the two mixtures
    alpha=a-1./a #'alpha' (symmetric attenuation)
    #2.2HERE WE ESTIMATE THE RELATIVE DELAY (delta)
    delta = -(np.imag((np.log(R21)/fmat)))                  ### delta = delay * fs
    
    # imaginary part, 'delta' relative delay
    ####################################################

    # 3.calculate weighted histogram
    # 3) Build a 2-d histogram (one dimension is phase, one is amplitude) where 
    #    the height at any phase/amplitude is the count of time-frequency bins that
    #    have approximately that phase/amplitude.

    #p=1; q=0;
    #p=2; q=2;
    h1=np.power(np.multiply(np.absolute(tf1),np.absolute(tf2)),p) #refer to run_duet.m line 45 for this. 
                                                                  #It's just the python translation of matlab 
    h2=np.power(np.absolute(fmat),q)

    tfweight=np.multiply(h1,h2) #weights vector 
    maxa=0.7;
    maxd=1.25*fs*d/343;#histogram boundaries for alpha, delta

    abins=35;
    dbins=181#2*50;#number of hist bins for alpha, delta


    # only consider time-freq points yielding estimates in bounds
    amask=(abs(alpha)<maxa)&(abs(delta)<maxd)
    amask=np.logical_not(amask)
    alphavec = np.asarray(ma.masked_array(alpha, mask=(amask)).transpose().compressed())[0]
    deltavec = np.asarray(ma.masked_array(delta, mask=(amask)).transpose().compressed())[0]
    tfweight = np.asarray(ma.masked_array(tfweight, mask=(amask)).transpose().compressed())[0]
    # to do masking the same way it is done in Matlab/Octave, after applying a mask we must take transpose and compress

    #determine histogram indices (sampled indices?)

    alphaind=np.around((abins-1)*(alphavec+maxa)/(2*maxa))
    deltaind=np.around((dbins-1)*(deltavec+maxd)/(2*maxd))

    #FULL-SPARSE TRICK TO CREATE 2D WEIGHTED HISTOGRAM
    #A(alphaind(k),deltaind(k)) = tfweight(k), S is abins-by-dbins
    A=sp.sparse.csr_matrix((tfweight, (alphaind, deltaind)),shape=(abins,dbins)).todense()
    #smooththehistogram-localaverage3-by-3neighboringbins

    A=twoDsmooth(A,smooth)
    X=np.linspace(-maxd,maxd,dbins)
    Y=np.linspace(-maxa,maxa,abins)
    '''
    if peak_mode==0:
        col_peak,_=signal.find_peaks(np.max(A,axis=0),prominence=np.max(A)*prom_const)
    else:
        col_peak,_=signal.find_peaks(np.max(A,axis=0),height=np.max(A)*max_const)
 
    row_peak = []
    for idx in col_peak:
        row_peak.append(np.argmax(A[:,idx]))

    if graph_mode == 1:
        fig = plt.figure(0)
        ax = fig.add_subplot(111, projection='3d')
        X_grid, Y_grid = np.meshgrid(X, Y)
        ax.plot_wireframe(X_grid,Y_grid,A)
        plt.xlabel('delta')
        plt.ylabel('alpha')
        plt.tight_layout()
        plt.show()

        # You can have a look at the histogram to look at the local peaks and what not

        ## Peak location detection
        plt.figure(1)
        #plt.imshow(A,origin='upper')
        #plt.colorbar()
        #plt.figure(1)
        plt.plot(X,np.max(A,axis=0))
        plt.xlabel('delta')
        plt.tight_layout()
        plt.show()        
    '''    
    A_1d = np.average(A,axis=0)    
    row_peak = []
    for idx in col_peak:
        row_peak.append(np.argmax(A[:,idx]))        
        
        
    ######################################    step 4,5,6,7
    ######################################
    #4.peak centers (determined from histogram) THIS IS DONE BY HUMAN.
    #4) Determine how many peaks there are in the histogram.
    #5) Find the location of each peak. 

    numsources=len(col_peak);

    peakdelta=X[col_peak]
    peakalpha=Y[row_peak]
    print('number of peaks: ', len(peakdelta))
    #convert alpha to a

    peaka=(peakalpha+np.sqrt(np.square(peakalpha)+4))/2;
    peaka=np.asarray(peaka)
 
    

    ##################################################
    #5.determine masks for separation
    #6) Assign each time-frequency frame to the nearest peak in phase/amplitude 
    #  space. This partitions the spectrogram into sources (one peak per source)

    test = float("inf")
    bestsofar=test*np.ones(tf1.shape)
    bestind=np.zeros(tf1.shape)

    for i in range(peakalpha.size):
        score = np.power(abs(np.multiply(peaka[i]*np.exp(-1j*fmat*peakdelta[i]),tf1)-tf2),2)/(1+peaka[i]*peaka[i])
        mask=score<bestsofar
        np.place(bestind,mask,i+1)
        s_mask=np.asarray(ma.masked_array(score, mask=np.logical_not(mask)).compressed())[0]
        np.place(bestsofar,mask,s_mask)

    ###################################################
    #6.&7.demix with ML alignment and convert to time domain
    #7) Then you create a binary mask (1 for each time-frequency point belonging to my source, 0 for all other points)
    #8) Mask the spectrogram with the mask created in step 7.
    #9) Rebuild the original wave file from 8.
    #10) Listen to the result.

    est=np.zeros((numsources,x1.shape[1]))
    (row,col)=bestind.shape
    for i in range(0,numsources):
        mask=ma.masked_equal(bestind,i+1).mask
        # here, 'h' stands for helper; we're using helper variables to break down the logic of
        # what's going on. Apologies for the order of the 'h's
        h1=np.zeros((1,tf1.shape[1]))
        h3=np.multiply((peaka[i]*np.exp(1j*fmat*peakdelta[i])),tf2)
        h4=((tf1+h3)/(1+peaka[i]*peaka[i]))
        h2=np.multiply(h4,mask)
        h=np.concatenate((h1,h2))

        esti=tfsynthesis(h,math.sqrt(2)*awin/1024,timestep,numfreq)

        #add back into the demix a little bit of the mixture
        #as that eliminates most of the masking artifacts

        est[i]=esti[0:x1.shape[1]]
        #write('out'+str(i),fs,np.asarray(est[i]+0.05*x1)[0])
    return est, A, A_1d, X, Y







def ConvBSS_IVA_NG(X, fs, N_fft=1024, N_hop=256, K=None, lr = 0.1, maxiter=1000, tol=1e-6):
    
    """  Fast algorithm for Frecuency Domain Blind source separation
            based on Independent Vector Analysis

        Parameters
        --------------------------------------------
        X : array containing mixtures, shape (# of mixture, # of samples) 

        fs : sampling frequency of mixture measurements in [Hz]

        N_fft : int, optional
                # of fft points (default = 1024)

        N_hop : int, optional
                Hop length of STFT (default = 256)

        K : int, optional
            # of sources. If None, same as the # of mixtures

        lr : float, optional
            Learning rate. (default = 0.1)

        max_iter: int, optional
                  Maximum number of iterations. (default = 1000)
        tol : float, optional 
              When the increment of likelihood is less than tol, the algorithm terminates (default = 1e-6)
        --------------------------------------------

        Returns
        --------------------------------------------
        y_t : Matrix containing separated sources, shape (# of sources, # of samples) 

        Y : STFT of y_t, shape(# of sources, # of freq bins, # of time bins)

        A : Matrix whose each column vector containing independent components (unscaled)
            shape (# of mixtures, # of sources, # of freq bins)
        --------------------------------------------



      - Original script in Matlab in Nov. 2, 2005 - Copyright: Taesu Kim
        Url: https://github.com/teradepth/iva/blob/master/matlab/ivabss.m

      - Citation: T. Kim, H. T. Attias, S. Lee and T. Lee, 
        "Blind Source Separation Exploiting Higher-Order Frequency Dependencies," 
        in IEEE Transactions on Audio, Speech, and Language Processing, 
        vol. 15, no. 1, pp. 70-79, Jan. 2007, doi: 10.1109/TASL.2006.872618.

      - Author (This python script): Hyungjoo Seo <seoyumyum@gmail.com>
        Date: 2/2/2022                                                      """

    
    M,_ = X.shape

    if K==None:
        K = M
    
    
    ## Perform short-time Fourier-Transform
    f, t, X_ft = signal.stft(X, fs, window = 'hamming', nperseg=N_fft, noverlap = N_fft-N_hop)

  
    epsi = 1e-6                                  ## For preventing overflow
    pObj = float("inf")                          ## Initialte with infinity 
    W = np.zeros((K,M,len(f)),dtype='complex')
    A = np.zeros((M,K,len(f)),dtype='complex')
    Wp = np.zeros((K,K,len(f)),dtype='complex')
    dWp = np.zeros(Wp.shape,dtype='complex')
    Q = np.zeros((K, M, len(f)),dtype='complex')
    Xp = np.zeros((K, len(f), len(t) ),dtype='complex')
    Y = np.zeros((K, len(f), len(t) ),dtype='complex')   
    Ysq = np.zeros((K, len(t)),dtype='complex')
    Ysq1 = np. zeros((K, len(t)),dtype='complex')

    ### Whiten (PCA) at each frequency:
    for i in range(len(f)):
        Xmean, _ = lib_HS.center(X_ft[:,i,:])
        Xp[:,i,:], Q[:,:,i] = lib_HS.whiten(Xmean)
        Wp[:,:,i] = np.eye(K)



    ### Learning algorithm
    for iter in range(maxiter):
        dlw = 0
        for i in range(len(f)):
            Y[:,i,:] = Wp[:,:,i]@Xp[:,i,:]

        Ysq = np.sum(abs(Y)**2,axis=1)**0.5
        Ysq1 = (Ysq + epsi)**-1

        for i in range(len(f)):
            ## Calculate multivariate score function and gradients
            Phi = Ysq1*Y[:,i,:]
            dWp[:,:,i] = (np.eye(K) - Phi@Y[:,i,:].T.conj()/len(t))@Wp[:,:,i]
            dlw = dlw + np.log(abs(np.linalg.det(Wp[:,:,i])) + epsi)

        ## update unmixing matrices
        Wp = Wp + lr*dWp

        Obj = (sum(sum(Ysq))/len(t)-dlw)/(K*len(f))
        dObj = pObj-Obj
        pObj = Obj

        if iter%20 == 1:
            print(iter, 'iterations: Objective=', Obj, ', dObj=', dObj);


        if abs(dObj)/abs(Obj) < tol:
            print('Converged')
            break
        iter += 1


    ## Correct scaling of unmixing filter coefficients
    for i in range(len(f)):
        W[:,:,i] = Wp[:,:,i]@Q[:,:,i]
        A[:,:,i] = np.linalg.pinv(W[:,:,i])           ## This is optional.
        W[:,:,i] = np.diag(np.diag(A[:,:,i]))@W[:,:,i]

    ## Calculate outputs
    for i in range(len(f)):
        Y[:,i,:] = W[:,:,i]@X_ft[:,i,:]

    ## Recover signal to time domain
    y_t = np.zeros((K,len(X[0,:])))
    for i in range(K):
        _, y = signal.istft(Y[i,:,:], fs, nperseg=N_fft, noverlap = N_fft-N_hop)
        y_t[i,:]=y[:len(X[0,:])]

    return y_t, Y, A



def ConvBSS_JadeICA(X, fs_true, N_fft=1024, N_hop=int(0.25*1024), M=4, K=None, f_LPF=None):
    
    ### STFT block ###
    f, t, _ = signal.stft(X[0,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)
    X_ft = np.zeros((M,len(f),len(t)), dtype='complex')
    print('(M,f,t): ', X_ft.shape)
    for i in range(M):
        _, _, X_ft[i,:,:] = signal.stft(X[i,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)   
    # hopsize H = nperseg - noverlap


    #magnitudeZxx = np.abs(X_ft[0,:,:])
    #log_spectrogramZxx = librosa.amplitude_to_db(magnitudeZxx)#-60

    #plt.figure(figsize=(10,4))
    #librosa.display.specshow(log_spectrogramZxx, sr=fs_true, x_axis='time', y_axis='linear', hop_length=N_hop) #,cmap=plt.cm.gist_heat)
    #plt.xlabel("Time")
    #plt.ylabel("Frequency")
    #plt.colorbar(format='%+2.0f dB')
    #plt.title("Spectrogram (dB)")


    ### Separation Step Using JadeICA ###

    A_f = np.zeros((len(f),M,K), dtype='complex')
    Y_ft = np.zeros((len(f), K, len(t)), dtype='complex')
    W_f = np.zeros((len(f), K, K), dtype='complex')
    V_f = np.zeros((len(f), K, M), dtype='complex')

    # at each f
    for i in range(len(f)):
        A_tmp,Y_tmp,V_tmp,W_tmp = jade(X_ft[:,i,:],m=None,max_iter=3000)
        A_f[i,:,:] = A_tmp    # freq x M x K   (estimated mixing matrix)
        Y_ft[i,:,:] = Y_tmp    # freq x K x T   (estimated source signals)
        W_f[i,:,:] = W_tmp    # freq x K x K   (unmixing matrix)
        V_f[i,:,:] = V_tmp    # freq x K x M   (Sphering/Whitening matrix)


    ### Permutation Resolve method: PowerRatio-correlation approach ###

    # Calculate PowerRatio
    PowR = np.zeros((len(f), K, len(t)))

    for freq in range(len(f)):
        for tau in range(len(t)):
            PowSum=0
            for i in range(K):    
                PowR[freq,i,tau] = np.linalg.norm(A_f[freq,:,i]*Y_ft[freq,i,tau],ord=2)**2    
                PowSum += PowR[freq,i,tau]
            PowR[freq,:,tau] /= PowSum     

            
    ##### Correlation Algorithm - centroid 

    cent_k = np.sum(PowR,axis=0)/len(f)    ## initialization: K x T

    #for freq in f:
    Sigma_f = np.arange(K)
    Y_ft_PowR = copy.deepcopy(Y_ft)
    A_f_PowR = copy.deepcopy(A_f)

    index_argmax = np.zeros((len(f),K),dtype='int')

    flag = 0
    N_trial = 100
    for n in range(N_trial):
        #print('(---n =',n,'---)')    
        #index_argmax_tmp = copy.deepcopy(index_argmax)
        cent_k_temp = copy.deepcopy(cent_k)
        PowR_sum = np.zeros((K,len(t)))
        for freq in range(len(f)):
            #print('---f =',freq,'---')
            index_argmax[freq,:]=Sigma_f

            Sum_max = -1
            for i in np.array(list(itertools.permutations(Sigma_f))):

                Sum_l = np.zeros(K)
                for l in range(K):
                    Sum_l[l] = np.corrcoef(PowR[freq,i[l],:],cent_k[Sigma_f[l],:])[1,0]
                Sum = np.sum(Sum_l)

                if Sum > Sum_max:
                    Sum_max = Sum
                    index_argmax[freq,:] = i

            PowR_sum += PowR[freq,index_argmax[freq,:],:]

        cent_k = PowR_sum/len(f)

        #print(cent_k[:,:3])
        ##print('(iteration = ',n,')','loss = ',np.sum(np.abs(cent_k_temp-cent_k)))
        if np.sum(np.abs(cent_k_temp-cent_k))<1e-5:
            flag += 1
        #print('flagCount = ', flag)
        if flag == 5:
            print('flagCount = ', flag)
            break

    for freq in range(len(f)):        
        Y_ft_PowR[freq,:,:] = Y_ft[freq,index_argmax[freq,:] ,:]
        A_f_PowR[freq,:,:] = A_f[freq,:,index_argmax[freq,:]].T
        
    ### Scaling after Clustering
    Y_ft_scale = copy.deepcopy(Y_ft_PowR)
    for freq in range(len(f)):
        for i in range(K):
            Y_ft_scale[freq,i,:] = A_f_PowR[freq,0,i]*Y_ft_PowR[freq,i,:]

    ## istft check
    y_t = np.zeros((K,len(X[0,:])))
    for i in range(K):
        _, y = signal.istft(Y_ft_scale[:,i,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)
        y_t[i,:]=y[:len(X[0,:])]
        if f_LPF != None:
            y_t[i,:] = lib_HS.butter_lowpass_filter(y_t[i,:], f_LPF, fs_true, 10)   ## LPF 
       
    return y_t,Y_ft_scale, A_f_PowR 



def ConvBSS_JadeICA_AoA(X, fs_true, N_fft=1024, N_hop=int(0.25*1024), M=4, K=None, f_LPF=None):
    
    ### STFT block ###
    f, t, _ = signal.stft(X[0,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)
    X_ft = np.zeros((M,len(f),len(t)), dtype='complex')
    #print('(M,f,t): ', X_ft.shape)
    for i in range(M):
        _, _, X_ft[i,:,:] = signal.stft(X[i,:], fs_true, nperseg=N_fft, noverlap = N_fft-N_hop)   


    ### Perform JadeICA at each frequency ###

    A_f = np.zeros((len(f),M,K), dtype='complex')

    # at each f
    for i in range(len(f)):
        A_tmp,_,_,_ = jade(X_ft[:,i,:],m=None,max_iter=3000)
        A_f[i,:,:] = A_tmp    # freq x M x K   (estimated mixing matrix)

    return A_f




def jade(X,m=None,max_iter=100,nem=None,tol=None):
    """Source separation of complex signals via Joint Approximate 
        Diagonalization of Eigen-matrices or JADE
    Parameters
    ----------
    X : array, shape (n_mixtures, n_samples)
        Matrix containing the mixtures
    m : int, optional
        Number of sources. If None, equals the number of mixtures
    max_iter : int, optional
        Maximum number of iterations
    nem : int, optional
        Number of eigen-matrices to be diagonalized
    tol : float, optional
        Threshold for stopping joint diagonalization
    Returns
    -------
    A : array, shape (n_mixtures, n_sources) 
        Estimate of the mixing matrix
    S : array, shape (n_sources, n_samples) 
        Estimate of the source signals
    V : array, shape (n_sources, n_mixtures)
        Estimate of the un-mixing matrix
    W : array, shape (n_components, n_mixtures)
        Sphering matrix
    Notes
    -----
    Original script in Matlab - version 1.6. Copyright: JF Cardoso.
    Url: http://perso.telecom-paristech.fr/~cardoso/Algo/Jade/jade.m
    Citation: Cardoso, Jean-Francois, and Antoine Souloumiac. 'Blind 
    beamforming for non-Gaussian signals.'IEE Proceedings F (Radar 
    and Signal Processing). Vol. 140. No. 6. IET Digital Library, 
    1993.
    Author (python script): Alex Bujan <afbujan@gmail.com>
    Date: 20/01/2016
    """

    n,T = X.shape

    if m==None:
        m = n

    if nem==None:
        nem = m

    if tol==None:
        tol = 1/(np.sqrt(T)*1e2)

    '''
    whitening
    '''

    X-=X.mean(1,keepdims=True)

    if m<n:
        #assumes white noise
        D,U     = eig((X.dot(X.conj().T))/T)
        k       = np.argsort(D)
        puiss   = D[k]
        ibl     = np.sqrt(puiss[-m:]-puiss[:-m].mean())
        bl      = 1/ibl
        W       = np.diag(bl).dot(U[:,k[-m:]].conj().T)
        IW      = U[:,k[-m:]].dot(np.diag(ibl))
    else:
        #assumes no noise
        IW      = sqrtm((X.dot(X.conj().T))/T)
        W       = inv(IW)

    Y    = W.dot(X)

    '''
    Cumulant estimation
    '''

    #covariance
    R    = Y.dot(Y.conj().T)/T
    #pseudo-covariance
    C    = Y.dot(Y.T)/T

    Q    = np.zeros((m*m*m*m,1),dtype=complex)
    idx  = 0

    for lx in range(m):
        Yl = Y[lx]
        for kx in range(m):
            Ykl = Yl*Y[kx].conj()
            for jx in range(m):
                Yjkl = Ykl*Y[jx].conj()
                for ix in range(m):
                    Q[idx] = Yjkl.dot(Y[ix].T)/T-\
                               R[ix,jx]*R[lx,kx]-\
                               R[ix,kx]*R[lx,jx]-\
                               C[ix,lx]*np.conj(C[jx,kx])
                    idx+=1

    '''
    computation and reshaping of the significant eigen matrices
    '''
    D,U = eig(Q.reshape((m*m,m*m)))
    K   = np.argsort(abs(D))
    la  = abs(D)[K]
    M   = np.zeros((m,nem*m),dtype=complex)
    h   = (m*m)-1
    for u in np.arange(0,nem*m,m):
        M[:,u:u+m] = la[h]*U[:,K[h]].reshape((m,m))
        h-=1

    '''
    joint approximate diagonalization of the eigen-matrices
    '''
    B       = np.array([[1,0,0],[0,1,1],[0,0,0]])+\
              1j*np.array([[0,0,0],[0,0,0],[0,-1,1]])
    Bt      = B.conj().T
    V       = np.eye(m).astype(complex)

    encore  = True

    #Main loop

    for n_iter in range(max_iter):

        for p in range(m-1):

            for q in np.arange(p+1,m):

                Ip = np.arange(p,nem*m,m)
                Iq = np.arange(q,nem*m,m)

                #Computing the Givens angles
                g       = np.vstack([M[p,Ip]-M[q,Iq],M[p,Iq],M[q,Ip]])
                D,vcp   = eig(np.real(B.dot(g.dot(g.conj().T)).dot(Bt)))
                K       = np.argsort(D)
                la      = D[K]

                angles  = vcp[:,K[2]]

                if angles[0]<0:
                    angles*=-1

                c    = np.sqrt(.5+angles[0]/2)
                s    = .5*(angles[1]-1j*angles[2])/c

                #updates matrices M and V by a Givens rotation
                if abs(s)>tol:
                    pair            = np.hstack((p,q))
                    G               = np.vstack(([c,-s.conj()],[s,c]))
                    V[:,pair]       = V[:,pair].dot(G)
                    M[pair,:]       = G.conj().T.dot(M[pair,:])
                    ids             = np.hstack((Ip,Iq))
                    M[:,ids]        = np.hstack((c*M[:,Ip]+s*M[:,Iq],\
                                                 -s.conj()*M[:,Ip]+c*M[:,Iq]))
                else:
                    encore          = False

        if encore==False:
            break

    if n_iter+1==max_iter:
        warnings.warn('JadeICA did not converge. Consider increasing '
                      'the maximum number of iterations or decreasing the '
                      'threshold for stopping joint diagonalization.')

    '''
    estimation of the mixing matrix and sources
    '''
    A    = IW.dot(V)
    S    = V.conj().T.dot(Y)

    return A,S,W,V

