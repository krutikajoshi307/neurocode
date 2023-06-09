# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:53:04 2023

@author: Fractal
"""

# =============================================================================
# GET DATA BLOCK- get EEG, muscle (GG= genioglossus muscle example) and fiber photometry signals
# EEG: can be either raw or filtered for frequency. Current code uses example case with sample rate of 5000Hz, 1-4Hz delta
# GG: smoothed: IIR 7th order butterworth, 67-1500Hz, smoothing value: 0.02
# Fiber photometry- raw gCAMP and baseline UV signal. Code lets you choose best fit method 
# =============================================================================


def get_EEG(filename):
    EEG=pd.read_csv(filename,header=None)       #Pandas Dataframe
    EEG=EEG.to_numpy()                          # numpy Array
    EEG=EEG[~np.isnan(EEG)]                     # removes empty rows/cols (NaNs)
    
    sg.theme('DarkBlue') #Adding theme          # GUI for user input- raw or z-scored
    layout=[
        [sg.Text('Choose an option for EEG data processing type')],
        [sg.Button('z-score'),sg.Button('raw')]
    ] 
    window3=sg.Window('Data Type Input',layout)
    event,y =window3.read()
    window3.close()
    if event =='z-score':
        numerator = np.subtract(EEG, np.nanmean(EEG))
        zscore = np.divide(numerator, np.nanstd(EEG))
        EEG=zscore
        return EEG
    elif event=='raw':
        EEG=EEG[~np.isnan(EEG)]
        return EEG
    else:
        print('Enter either z_score or raw')
    return 

def get_GG(filename):
    GG=pd.read_csv(filename,header=None)
    GG=GG.to_numpy()
    GG=GG[~np.isnan(GG)]
    GG=np.divide(GG,norm)
    
    sg.theme('DarkBlue') #Adding theme
    layout=[
        [sg.Text('Choose an option for GG data processing type')],
        [sg.Button('z-score'),sg.Button('raw')]
    ] 
    window4=sg.Window('Input data',layout)
    event,y =window4.read()
    window4.close()
    if event =='z-score':
        numerator = np.subtract(GG, np.nanmean(GG))
        zscore = np.divide(numerator, np.nanstd(GG))
        GG=zscore
        return GG
    elif event=='raw':
        GG=GG[~np.isnan(GG)]
        return GG
    else:
        print('Enter either z_score or raw')
    return 
    
    
def get_gCAMP_dF_F(gGAMP_filename, UV_filename):        # Pandas df for gCAMP and UV files
    gCAMP=pd.read_csv(gGAMP_filename,header=None)
    gCAMP=gCAMP.to_numpy()
    gCAMP=gCAMP[~np.isnan(gCAMP)]
    
    UV=pd.read_csv(UV_filename,header=None)
    UV=UV.to_numpy()
    UV=UV[~np.isnan(UV)]
    
    indexes=np.arange(1,len(gCAMP)+1)
    time=indexes/500
    time_hours=time/3600
    
    
    sg.theme('DarkBlue') #Adding theme
    layout=[
        [sg.Text('Choose an fit method for FP data processing type')],
        [sg.Button('Highpass'),sg.Button('Polyfit'),sg.Button('Expfit: 2 order single'),sg.Button('Expfit: double') ]
    ] 
    window5=sg.Window('Data Type Input',layout)
    event,z =window5.read()
    window5.close()
    
    if event=='Highpass':                               # Filter math adapted from Akam, T., & Walton, M. E. (2019). pyPhotometry: Open source Python based hardware and software for fiber photometry data acquisition. Scientific Reports, 9, 3521. 
                                                        # https://doi.org/10.1038/s41598-019-39724-y
        gCAMP_denoised = medfilt(gCAMP, kernel_size=5)
        UV_denoised = medfilt(UV, kernel_size=5)
        b,a = butter(2, 10, btype='low', fs=sr)
        gCAMP_denoised = filtfilt(b,a, gCAMP_denoised)
        UV_denoised = filtfilt(b,a, UV_denoised)
        d,c = butter(2, 0.001, btype='high', fs=sr)
        gCAMP_highpass = filtfilt(d,c, gCAMP_denoised, padtype='even')
        UV_highpass = filtfilt(d,c, UV_denoised, padtype='even')
        slope, intercept, r_value, p_value, std_err = linregress(x=UV_highpass, y=gCAMP_highpass)
        gCAMP_est_motion = intercept + slope * UV_highpass
        gCAMP_corrected = gCAMP_highpass - gCAMP_est_motion
        #f,e = butter(2, 0.001, btype='low', fs=sr)
        #baseline_fluorescence = filtfilt(f,e, gCAMP_denoised, padtype='even')
        gCAMP_dF_F = gCAMP_corrected/gCAMP_highpass      # old version: replace gCAMP_highpass with baseline_fluorescence
        numerator = np.subtract(gCAMP_dF_F, np.nanmean(gCAMP_dF_F))
        zscore = np.divide(numerator, np.nanstd(gCAMP_dF_F))
        return zscore,gCAMP_dF_F,gCAMP_highpass,UV_highpass
    
    elif event=='Polyfit':
        gCAMP_denoised = medfilt(gCAMP, kernel_size=5)
        UV_denoised = medfilt(UV, kernel_size=5)
        b,a = butter(2, 10, btype='low', fs=sr)
        gCAMP_denoised = filtfilt(b,a, gCAMP_denoised)
        UV_denoised = filtfilt(b,a, UV_denoised)
        coefs_gCAMP = np.polyfit(time,gCAMP_denoised, deg=4)
        gCAMP_polyfit = np.polyval(coefs_gCAMP,time)
        coefs_UV = np.polyfit(time, UV_denoised, deg=4)
        UV_polyfit = np.polyval(coefs_UV, time)
        gCAMP_ps = gCAMP_denoised - gCAMP_polyfit
        UV_ps = UV_denoised - UV_polyfit
        slope, intercept, r_value, p_value, std_err = linregress(x=UV_ps, y=gCAMP_ps)
        gCAMP_est_motion = intercept + slope * UV_ps
        gCAMP_corrected = gCAMP_ps - gCAMP_est_motion
        #d,c = butter(2, 0.001, btype='low', fs=sr)
        #baseline_fluorescence = filtfilt(d,c, gCAMP_denoised, padtype='even')
        gCAMP_dF_F = gCAMP_corrected/gCAMP_polyfit          # old version: replace gCAMP_polyfit with baseline_fluorescence
        numerator = np.subtract(gCAMP_dF_F, np.nanmean(gCAMP_dF_F))
        zscore = np.divide(numerator, np.nanstd(gCAMP_dF_F))
        return zscore,gCAMP_dF_F,gCAMP_ps,UV_ps
    
    elif event=='Expfit: 2 order single':
        gCAMP_denoised = medfilt(gCAMP, kernel_size=5)
        UV_denoised = medfilt(UV, kernel_size=5)
        b,a = butter(2, 10, btype='low', fs=sr)
        gCAMP_denoised = filtfilt(b,a, gCAMP_denoised)
        UV_denoised = filtfilt(b,a, UV_denoised)
        
        # The exponential curve for fitting:
        def exp_func(x, a, b, c):
            return a*np.exp(-b*x) + c
       
        # Fit curve to gCAMP and UV signal.
        gCAMP_parms, parm_cov = curve_fit(exp_func, time, gCAMP_denoised, p0=[1,1e-3,1],bounds=([0,0,0],[4,0.1,4]), maxfev=1000)
        gCAMP_expfit = exp_func(time, *gCAMP_parms)
        UV_parms, parm_cov = curve_fit(exp_func, time, UV_denoised, p0=[1,1e-3,1],bounds=([0,0,0],[4,0.1,4]), maxfev=1000)
        UV_expfit = exp_func(time, *UV_parms)
        # Subtract fit line from signal
        gCAMP_es = gCAMP_denoised - gCAMP_expfit
        UV_es = UV_denoised - UV_expfit
        # Motion correction
        slope, intercept, r_value, p_value, std_err = linregress(x=UV_es, y=gCAMP_es)
        gCAMP_est_motion = intercept + slope * UV_es
        gCAMP_corrected = gCAMP_es - gCAMP_est_motion
        #d,c = butter(2, 0.001, btype='low', fs=sampling_rate)
        #baseline_fluorescence = filtfilt(d,c, gCAMP_denoised, padtype='even')
        gCAMP_dF_F = gCAMP_corrected/gCAMP_expfit # old version: replace gCAMP_expfit with baseline_fluorescence
        numerator = np.subtract(gCAMP_dF_F, np.nanmean(gCAMP_dF_F))
        zscore = np.divide(numerator, np.nanstd(gCAMP_dF_F))
        return zscore,gCAMP_dF_F,gCAMP_es,UV_es
    
    elif event=='Expfit: double':
         gCAMP_denoised = medfilt(gCAMP, kernel_size=5)
        UV_denoised = medfilt(UV, kernel_size=5)
        b,a = butter(2, 10, btype='low', fs=sampling_rate)
        gCAMP_denoised = filtfilt(b,a, gCAMP_denoised)
        UV_denoised = filtfilt(b,a, UV_denoised)
        
        # The double exponential curve we are going to fit.
        def double_exp_func(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
            tau_fast = tau_slow*tau_multiplier
            return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)
    
        # Fit curve to gCAMP and UV signal.
        
        max_sig=np.max(gCAMP_denoised)
        inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
        bounds = ([0      , 0      , 0      , 600  , 0],
                  [max_sig, max_sig, max_sig, 36000, 1])
        
        gCAMP_params, param_cov = curve_fit(double_exp_func, time, gCAMP_denoised, 
                                  p0=inital_params, bounds=bounds, maxfev=1000)
        gCAMP_expfit = double_exp_func(time, *gCAMP_params)
        
        max_sig=np.max(UV_denoised)
        inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
        bounds = ([0      , 0      , 0      , 600  , 0],
                  [max_sig, max_sig, max_sig, 36000, 1])
        
        UV_params, param_cov = curve_fit(double_exp_func, time, UV_denoised, 
                                  p0=inital_params, bounds=bounds, maxfev=1000)
        UV_expfit = double_exp_func(time, *UV_params)
        
        # Subtract fit line from signal
        gCAMP_es = gCAMP_denoised - gCAMP_expfit
        UV_es = UV_denoised - UV_expfit
        # Motion correction
        slope, intercept, r_value, p_value, std_err = linregress(x=UV_es, y=gCAMP_es)
        gCAMP_est_motion = intercept + slope * UV_es
        gCAMP_corrected = gCAMP_es - gCAMP_est_motion
        #d,c = butter(2, 0.001, btype='low', fs=sampling_rate)
        #baseline_fluorescence = filtfilt(d,c, gCAMP_denoised, padtype='even')
        gCAMP_dF_F = gCAMP_corrected/gCAMP_expfit
        numerator = np.subtract(gCAMP_dF_F, np.nanmean(gCAMP_dF_F))
        zscore = np.divide(numerator, np.nanstd(gCAMP_dF_F))
        return zscore,gCAMP_dF_F,gCAMP_es,UV_es
    
    return
            
def one_epoch():                                # Line graphs containing all signals looped through for all user provided epochs. 
    for i in range(0,n):
        plt.figure(figsize=(14, 8), dpi=600)
        plt.subplot(3,1,1)
        py.plot(time[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],EEG[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],color= '#fb7104', label='EEG delta power')
        plt.title('Quiet wake to Active wake epoch {}'.format(i+1),fontdict={'fontname':'Calibri', 'fontsize':20})
        py.ylabel('Power')
        plt.axvline(timestamps[i]/sr,color='k',linestyle='dashed') # draws vertical dashed line at epoch timestamp
        py.xlim((timestamps[i]-(sr*t1))/sr,(timestamps[i]+(sr*t2))/sr)
        plt.xticks([(timestamps[i]-(sr*t1))/sr,timestamps[i]/sr,(timestamps[i]+(sr*t2))/sr])
        plt.legend(loc=1)
        
        plt.subplot(3,1,2)
        plt.subplots_adjust(hspace=0.5)
        py.plot(time[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],GG[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],color= '#037727', label='GG sm')
        py.ylabel('Volts')
        plt.axvline(timestamps[i]/sr,color='k',linestyle='dashed')
        py.xlim((timestamps[i]-(sr*t1))/sr,(timestamps[i]+(sr*t2))/sr)
        plt.xticks([(timestamps[i]-(sr*t1))/sr,timestamps[i]/sr,(timestamps[i]+(sr*t2))/sr])
        plt.legend(loc=1)
        
        plt.subplot(3,1,3)
        plt.subplots_adjust(hspace=0.5)
        py.plot(time[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],zdF_F[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],color= '#fb040c', label='zscore')
        py.xlabel('Time (seconds)')
        py.ylabel('df/f z-score')
        plt.axvline(timestamps[i]/sr,color='k',linestyle='dashed')
        py.xlim((timestamps[i]-(sr*t1))/sr,(timestamps[i]+(sr*t2))/sr)
        plt.xticks([(timestamps[i]-(sr*t1))/sr,timestamps[i]/sr,(timestamps[i]+(sr*t2))/sr])
        plt.legend(loc=1)
        plt.show()
    return

def heat_map():
    for i in range(0,n):
        EEGtrans=np.expand_dims(EEG[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],axis=0) # converts column vector to row vector
        GGtrans=np.expand_dims(GG[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],axis=0)
        zdF_Ftrans=np.expand_dims(zdF_F[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)],axis=0)
        
        plt.figure(figsize=(14, 8), dpi=600)
        plt.subplot(3,1,1) 
        plt.subplots_adjust(hspace=0.5)
        plt.imshow(EEGtrans, aspect = "auto", cmap="hot", interpolation = "nearest",label='EEG delta power') # see imshow documentation. Displays data as an image, i.e., on a 2D regular raster.
        plt.title('Quiet wake to Active wake epoch {}'.format(i+1),fontdict={'fontname':'Calibri', 'fontsize':20})
        py.xlabel('Time (seconds)')
        plt.colorbar(fraction= 0.05,pad=0.05)
        py.ylabel(' EEG delta power')
        plt.axvline(sr*t1,color='k',linestyle='dashed')
        plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2])
        
        plt.subplot(3,1,2)
        plt.subplots_adjust(hspace=0.5)
        py.imshow(GGtrans, aspect = "auto", cmap="hot", interpolation = "nearest",label='GG sm')
        py.xlabel('Time (seconds)')
        py.ylabel('GGsm (Volts)')
        plt.axvline(sr*t1,color='k',linestyle='dashed')
        plt.colorbar(fraction= 0.05,pad=0.05)
        plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2])
       
        plt.subplot(3,1,3)
        plt.subplots_adjust(hspace=0.5)
        py.imshow(zdF_Ftrans, aspect = "auto", cmap="hot", interpolation = "nearest",label='df/f z-score')
        py.xlabel('Time (seconds)')
        py.ylabel('df/f z-score')
        plt.axvline(sr*t1,color='k',linestyle='dashed')
        plt.colorbar(fraction= 0.05,pad=0.05)
        plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2])
        plt.show()
    return
        
def save_data(): # saves data in current folder. Use folder address after r'C:\
    for i in range(0,n):
        EEG_epoch= EEG[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)]
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\EEG_c{0}_{1}.csv'.format(case,i+1),EEG_epoch,delimiter='',newline='\n')
        GG_epoch=GG[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)]
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\GG_c{0}_{1}.csv'.format(case,i+1),GG_epoch,delimiter='',newline='\n')
        dF_F_epoch=dF_F[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)]
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\dF_F_c{0}_{1}.csv'.format(case,i+1),dF_F_epoch,delimiter='',newline='\n')
        zdF_F_epoch=zdF_F[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)]
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\zdF_F_c{0}_{1}.csv'.format(case,i+1),zdF_F_epoch,delimiter='',newline='\n')
        gCAMP_epoch=gCAMP[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)]
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\gCAMP_c{0}_{1}.csv'.format(case,i+1),gCAMP_epoch,delimiter='',newline='\n')
        UV_epoch=UV[timestamps[i]-(sr*t1):timestamps[i]+(sr*t2)]
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\UV_c{0}_{1}.csv'.format(case,i+1),UV_epoch,delimiter='',newline='\n')
    
    return  
        
# =============================================================================
#    RUN BLOCK
# =============================================================================


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import pylab as py
    from scipy.signal import medfilt,butter,filtfilt
    from scipy.stats import linregress
    from scipy.optimize import curve_fit
    import PySimpleGUI as sg
    
    
    sg.theme('DarkBlue')                                        ## GUI for user input- # of epochs in a case and sample rate
    layout=[
        [sg.Text('Enter # of epochs in case')],
        [sg.Text('case #',size=(9,1)),sg.InputText()],
        [sg.Text('# of epochs',size=(9,1)),sg.InputText()],
        [sg.Text('sample rate',size=(9,1)),sg.InputText()],
        [sg.Submit(),sg.Cancel()]
    ] 
    window1=sg.Window('Input data',layout)
    event,n =window1.read()
    window1.close()
    case=int(n[0])
    n=int(n[1])
    sr=int(n[2])
    
    
    sg.theme('DarkBlue') 
        
    layout=[
        [sg.Text('Enter epoch times and offset (in seconds)')],   # GUI for user input timestamps (upto 15) ( in seconds) corresponding to the event of choice
        [sg.Text('Epoch 1',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 2',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 3',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 4',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 5',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 6',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 7',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 8',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 9',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 10',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 11',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 12',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 13',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 14',size=(7,1)),sg.InputText()],
        [sg.Text('Epoch 15',size=(7,1)),sg.InputText()],
        [sg.Text('t1 (pre)',size=(7,1)),sg.InputText()],          # specifies time period prior to event
        [sg.Text('t2 (post)',size=(7,1)),sg.InputText()],         # specifies time period prior after event
        [sg.Text('Offset',size=(7,1)),sg.InputText()],            # Time removed from the start of recording. To remove noise and time before gCAMP and UV ramp
        [sg.Text('GG norm',size=(7,1)),sg.InputText()],           # Either max amp or NREM normalization factor as decided by user. 
        
        [sg.Submit(),sg.Cancel()]
    ]    
         
    window2=sg.Window('Input data',layout)
    event,x =window2.read()
    window2.close()
    
    t1=int(x[15])
    t2=int(x[16])
    offset=float(x[17])
    norm=float(x[18])
    epochs=[]    
    for i in range(0,n):
            epoch_i = float(x[i])
            epochs.append(epoch_i)
            
    timestamps=[]
    for i in range(0,n):
            timestamp_i=((epochs[i]-offset)*sr)  
            timestamp_i=int(timestamp_i)
            timestamps.append(timestamp_i)
        
# =============================================================================
#     ## get the data 
# =============================================================================
    EEG = get_EEG('EEG.txt')
    GG =get_GG('GG.txt')
    zdF_F,dF_F,gCAMP,UV =get_gCAMP_dF_F('gCAMP.txt','UV.txt')
    
    indexes=np.arange(1,len(EEG)+1)
    time=indexes/sr
# =============================================================================
            
    # RUN BLOCK-----FUNCTIONS    
        
    one_epoch()
    heat_map()
    save_data()
   
            
        
        
        
