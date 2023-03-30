# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:02:31 2023

@author: Fractal
"""
def line_plot(): # all epochs combined and averaged so variance might be high. Alternate scenario would be to combine n epochs per case to form 3 cases(3 lines) and average those
        
        l1=list(range(0,len(EEG_mean)))
        fig=plt.figure(figsize=(14,8),dpi=600)
        plt.subplot(3,1,1)
        plt.title('Quiet wake to Active wake group data (mean +/- SD)',fontdict={'fontname':'Calibri', 'fontsize':20})
        plt.subplots_adjust(hspace=0.5)
        py.ylabel('Power')
        plt.plot(EEG_mean,color='Black',marker='',linestyle='solid',linewidth=2,label='EEG delta power')
        plt.fill_between(l1, EEG_mean-EEG_SD,EEG_mean+EEG_SD,color='lightcoral',alpha=0.5)
        plt.axvline(x=(t1*len(EEG_mean))/(t1+t2),color='k',linestyle='dashed')
        py.xlim(0,len(EEG_mean)) 
        plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2])
        plt.legend(loc=1)
        
        l2=list(range(0,len(GG_mean)))
        plt.subplot(3,1,2)
        plt.subplots_adjust(hspace=0.5)
        #py.ylim(-5,10)
        plt.plot(GG_mean,color='Black',marker='',linestyle='solid',linewidth=2,label='GG_sm')
        plt.fill_between(l2, GG_mean-GG_SD,GG_mean+GG_SD,color='lightcoral',alpha=0.5)
        py.ylabel('Volts')
        plt.axvline(x=(t1*len(GG_mean))/(t1+t2),color='k',linestyle='dashed')
        py.xlim(0,len(GG_mean))
        plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2])
        plt.legend(loc=1)
        
        
        plt.subplot(3,1,3)
        plt.subplots_adjust(hspace=0.5)
        plt.plot(zdF_F_mean,color='Black',marker='',linestyle='solid',linewidth=2,label='dF/F')
        plt.fill_between(l2, zdF_F_mean-zdF_F_SD,zdF_F_mean+zdF_F_SD,color='lightcoral',alpha=0.5)
        py.ylabel('dF/F z-score')
        py.ylim(-2,2)
        plt.axvline(x=(t1*len(zdF_F_mean))/(t1+t2),color='k',linestyle='dashed')
        py.xlim(0,len(zdF_F_mean))
        plt.legend(loc=1)
        plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2])
        py.xlabel('Time(sec)')
        
        fig.savefig('line_plot.svg')
        return     

def heat_map():
    
    fig=plt.figure(figsize=(14, 8), dpi=600)
    ax1=sns.heatmap(GG_map,cmap='viridis')
    plt.title('Quiet wake to Active wake- GGsm',fontdict={'fontname':'Calibri', 'fontsize':20})
    py.xlabel('Time (seconds)')
    plt.axvline(x=(t1*len(GG_mean))/(t1+t2),color='k',linestyle='dashed')
    py.xlim(0,len(GG_mean))
    plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2], rotation=0)
    fig.savefig('GG_heatmap.svg')
    
    fig=plt.figure(figsize=(14, 8), dpi=600)
    ax1=sns.heatmap(zdF_F_map,cmap='viridis')
    plt.title('Quiet wake to Active wake- dF/F',fontdict={'fontname':'Calibri', 'fontsize':20})
    py.xlabel('Time (seconds)')
    plt.axvline(x=(t1*len(zdF_F_mean))/(t1+t2),color='k',linestyle='dashed')
    py.xlim(0,len(zdF_F_mean))
    plt.xticks([0,sr*t1,sr*(t1+t2)],[-t1,0,t2], rotation=0)
    fig.savefig('zdF_F_heatmap.svg')
    
    return

def bar_graph():                        #barplot: bars, plot: lines connecting pre and post, stripplot: dots connecting lines
    
    if event1=='Amp': 
   
        fig=plt.figure(figsize=(14,8),dpi=600)
        plt.subplot(1,3,1)
        py.ylim(0,1)
        plt.subplots_adjust(wspace=0.5)
        sns.set_context('paper',font_scale=2)# to set themes and font sizes, Types are paper, talk and poster
        clrs=['lightcoral','lightcoral']
        ax=sns.barplot(data=[EEG_pre_mean_series,EEG_post_mean_series], capsize=0.2,errwidth=2,errcolor='black',palette=clrs, edgecolor='black', linewidth=2,saturation=1)# by default, the estimator will be the mean
        ax=sns.stripplot(data=[EEG_pre_mean_series,EEG_post_mean_series],color='#555454', size=4,jitter=0.05)
        ax.set(ylabel='Power (normalized)')
        ax.set(xlabel='Quiet wake        Active wake')
        ax.set(xticklabels=[])
        plt.title('EEG delta power',fontsize=20)
        print('\n')
        print('Stats for EEG delta power, epochs averaged: ')
        print('N=', len(EEG_pre_mean),'df=',len(EEG_pre_mean)-1)
        print(stats.ttest_rel(EEG_pre_mean,EEG_post_mean,alternative='two-sided'))
        print('\n')
    
        plt.subplot(1,3,2)
        plt.subplots_adjust(wspace=0.5)
        sns.set_context('paper',font_scale=2)# to set themes and font sizes, Types are paper, talk and poster
        clrs=['lightcoral','lightcoral']
        ax=sns.barplot(data=[GG_pre_mean_series,GG_post_mean_series], capsize=0.2,errwidth=2,errcolor='black',palette=clrs, edgecolor='black', linewidth=2,saturation=1)# by default, the estimator will be the mean
        ax=sns.stripplot(data=[GG_pre_mean_series,GG_post_mean_series],color='#555454',size=4,jitter=0.05)
        ax.set(ylabel='Amplitude(Volts)')
        ax.set(xlabel='Quiet wake        Active wake')
        ax.set(xticklabels=[])
        plt.title('GG sm',fontsize=20)
        print('Stats for GG sm, epochs averaged: ')
        print('N=', len(GG_pre_mean),'df=',len(GG_pre_mean)-1)
        print(stats.ttest_rel(GG_pre_mean,GG_post_mean,alternative='two-sided'))
        print('\n')
    
        plt.subplot(1,3,3)
        plt.subplots_adjust(wspace=0.5)
        py.ylim(-1,3)
        sns.set_context('paper',font_scale=2)# to set themes and font sizes, Types are paper, talk and poster
        clrs=['lightcoral','lightcoral']
        ax=sns.barplot(data=[zdF_F_pre_mean_series,zdF_F_post_mean_series], capsize=0.2,palette=clrs,errwidth=2,errcolor='black', edgecolor='black', linewidth=2,saturation=1)# by default, the estimator will be the mean
        ax=sns.stripplot(data=[zdF_F_pre_mean_series,zdF_F_post_mean_series],color='#555454',size=4,jitter=0.05)
        ax.set(ylabel='z-score')
        ax.set(xlabel='Quiet wake        Active wake')
        ax.set(xticklabels=[])
        plt.title('z-score',fontsize=20)
        print('Stats for z-score, epochs averaged: ')
        print('N=', len(zdF_F_pre_mean),'df=',len(zdF_F_pre_mean)-1)
        print(stats.ttest_rel(zdF_F_pre_mean,zdF_F_post_mean,alternative='two-sided'))
        print('\n')
        
        fig.savefig('epochs group graph.svg')
        return
    
    elif event1=='AUC':
        
        fig=plt.figure(figsize=(14,8),dpi=600)
        plt.subplot(1,3,1)
        py.ylim(0,1)
        plt.subplots_adjust(wspace=0.5)
        sns.set_context('paper',font_scale=2)# to set themes and font sizes, Types are paper, talk and poster
        clrs=['lightcoral','lightcoral']
        ax=sns.barplot(data=[EEG_pre_mean_series,EEG_post_mean_series], capsize=0.2,errwidth=2,errcolor='black',palette=clrs, edgecolor='black', linewidth=2,saturation=1)# by default, the estimator will be the mean
        ax=sns.stripplot(data=[EEG_pre_mean_series,EEG_post_mean_series],color='#555454', size=4,jitter=0.05)
        ax.set(ylabel='Power (normalized)')
        ax.set(xlabel='Quiet wake        Active wake')
        ax.set(xticklabels=[])
        plt.title('EEG delta power',fontsize=20)
        print('\n')
        print('Stats for EEG delta power, epochs averaged: ')
        print('N=', len(EEG_pre_mean),'df=',len(EEG_pre_mean)-1)
        print(stats.ttest_rel(EEG_pre_mean,EEG_post_mean,alternative='two-sided'))
        print('\n')
    
        plt.subplot(1,3,2)
        plt.subplots_adjust(wspace=0.5)
        sns.set_context('paper',font_scale=2)# to set themes and font sizes, Types are paper, talk and poster
        clrs=['lightcoral','lightcoral']
        ax=sns.barplot(data=[AUC_GG_pre_series,AUC_GG_post_series], capsize=0.2,errwidth=2,errcolor='black',palette=clrs, edgecolor='black', linewidth=2,saturation=1)# by default, the estimator will be the mean
        ax=sns.stripplot(data=[AUC_GG_pre_series,AUC_GG_post_series],color='#555454',size=4,jitter=0.05)
        ax.set(ylabel='AUC(amp* time in sec)')
        ax.set(xlabel='Quiet wake        Active wake')
        ax.set(xticklabels=[])
        plt.title('GG AUC',fontsize=20)
        print('Stats for GG AUC, epochs averaged: ')
        print('N=', len(AUC_GG_pre),'df=',len(AUC_GG_pre)-1)
        print(stats.ttest_rel(AUC_GG_pre,AUC_GG_post,alternative='two-sided'))
        print('\n')
    
        plt.subplot(1,3,3)
        plt.subplots_adjust(wspace=0.5)
        py.ylim(-1,3)
        sns.set_context('paper',font_scale=2)# to set themes and font sizes, Types are paper, talk and poster
        clrs=['lightcoral','lightcoral']
        ax=sns.barplot(data=[zdF_F_pre_mean_series,zdF_F_post_mean_series], capsize=0.2,palette=clrs,errwidth=2,errcolor='black', edgecolor='black', linewidth=2,saturation=1)# by default, the estimator will be the mean
        ax=sns.stripplot(data=[zdF_F_pre_mean_series,zdF_F_post_mean_series],color='#555454',size=4,jitter=0.05)
        ax.set(ylabel='z-score')
        ax.set(xlabel='Quiet wake        Active wake')
        ax.set(xticklabels=[])
        plt.title('z-score',fontsize=20)
        print('Stats for z-score, epochs averaged: ')
        print('N=', len(zdF_F_pre_mean),'df=',len(zdF_F_pre_mean)-1)
        print(stats.ttest_rel(zdF_F_pre_mean,zdF_F_post_mean,alternative='two-sided'))
        print('\n')
        
        fig.savefig('epochs group graph.svg')
        return
    return

def save_grouped_data():
   
    if event1=='Amp': 
    
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\EEG_grouped_mean.csv',EEG_mean,delimiter='',newline='\n')
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\GG_grouped_mean.csv',GG_mean,delimiter='',newline='\n')
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\zdF_F_grouped_mean.csv',zdF_F_mean,delimiter='',newline='\n')
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\bar_grouped_data.csv',grouped_bar_graph_amp,delimiter='',header='EEG_pre_mean\n EEG_post_mean\n GG_pre_mean\n GG_post_mean\n zdF_F_pre_mean\n zdF_F_post_mean\n ')
        return
    
    if event1=='AUC':
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\EEG_grouped_mean.csv',EEG_mean,delimiter='',newline='\n')
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\GG_grouped_mean.csv',GG_mean,delimiter='',newline='\n')
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\zdF_F_grouped_mean.csv',zdF_F_mean,delimiter='',newline='\n')
        np.savetxt(r'C:\Users\Fractal\Desktop\Python\Fiber Photometry\bar_grouped_data.csv',grouped_bar_graph_AUC,delimiter='',header='EEG_pre_mean\n EEG_post_mean\n AUC_GG_pre_\n AUC_GG_post\n zdF_F_pre_mean\n zdF_F_post_mean\n ')
        return
    return

        

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import numpy as np
    import pylab as py
    import seaborn as sns
    import PySimpleGUI as sg
    from scipy.integrate import simpson
    from numpy import trapz
    
    
                                                ## GUI for user input. Different pre and post times can be used for heat maps/line graphs and bar graphs
    sg.theme('DarkBlue')                        #Adding theme
    layout=[
        [sg.Text('Group details')],
        [sg.Text('# of epochs per case',size=(15,1)),sg.InputText()],
        [sg.Text('total # of cases',size=(15,1)),sg.InputText()],
        [sg.Text('t1(pre)- line graph & heat map',size=(25,1)),sg.InputText()],
        [sg.Text('t2(post)-line graph & heat map',size=(25,1)),sg.InputText()],
        [sg.Text('t1(pre)- bar graph',size=(20,1)),sg.InputText()],
        [sg.Text('t2(post)-bar graph',size=(20,1)),sg.InputText()],
        [sg.Text('sample rate',size=(20,1)),sg.InputText()],
        [sg.Submit(),sg.Cancel()]
    ] 
    window=sg.Window('Input data',layout)
    event,x =window.read()
    window.close()
    n=int(x[0])
    case=int(x[1])
    t1=int(x[2])
    t2=int(x[3])
    t1bar=int(x[4])
    t2bar=int(x[5])
    sr=int(x[6])
    
    sg.theme('DarkBlue') #Adding theme
    layout=[
        [sg.Text('Choose an option for GG data processing type')],    # USer can choose if the muscle output for the bar graph be in amplitude or area-under-curve
        [sg.Button('Amp'),sg.Button('AUC')]
    ] 
    window1=sg.Window('Input data',layout)
    event1,y =window1.read()
    window1.close()
     
   
    
    # GET THE DATA
    EEG_full=[]
    EEG_pre=[]
    EEG_post=[]
    EEG_pre_mean=[]
    EEG_post_mean=[]
        
    for case in range(1,case+1):
        for i in range(0,n):
            z=len(EEG_full)
            z=z+1
            EEG=pd.read_csv('EEG_c{0}_{1}.csv'.format(case,i+1),header=None)
            EEG=EEG.rename(columns={0:'epoch {}'.format(z)})
            EEGpre=EEG[(t1-t1bar)*sr:(t1*sr)]
            EEGpost=EEG[(t1*sr):(t1+t2bar)*sr]
            EEGpremean=np.mean(EEGpre,axis=0)
            EEGpostmean=np.mean(EEGpost,axis=0)
            
            EEG_full.append(EEG)
            EEG_pre.append(EEGpre)
            EEG_post.append(EEGpost)
            EEG_pre_mean.append(EEGpremean)
            EEG_post_mean.append(EEGpostmean)   
            EEG_pre_mean_series=pd.Series(EEG_pre_mean)
            EEG_post_mean_series=pd.Series(EEG_post_mean)
                
    EEG_mean=np.mean(EEG_full, axis=0)
    EEG_mean=EEG_mean.flatten()
    EEG_SD=np.std(EEG_full, axis=0)
    EEG_SD=EEG_SD.flatten()
    
    
    
    GG_full=[]
    GG_map=[]
    GG_pre=[]
    GG_post=[]
    GG_pre_mean=[]
    GG_post_mean=[]
    ##############
    AUC_GG_pre=[]
    AUC_GG_post=[]
    ###############
    for case in range(1,case+1):
        for i in range(0,n):
            z=len(GG_full)
            z=z+1
            GG=pd.read_csv('GG_c{0}_{1}.csv'.format(case,i+1),header=None)
            GG=GG.rename(columns={0:'epoch {}'.format(z)})
            GGpre=GG[(t1-t1bar)*sr:(t1*sr)]
            GGprenumpy=np.array(GGpre)
            GGpost=GG[(t1*sr):(t1+t2bar)*sr]
            GGpostnumpy=np.array(GGpost)
            GGpremean=np.mean(GGpre,axis=0)
            GGpostmean=np.mean(GGpost,axis=0)
            
            ########################################################
            ## FOR AUC only:
            # Compute the AUC using the composite trapezoidal rule.
            AUCGGpre = np.trapz(GGprenumpy, dx=len(GGprenumpy),axis=0)
            AUCGGpost = np.trapz(GGpostnumpy, dx=len(GGpostnumpy),axis=0)
            AUCGGpre=AUCGGpre/(len(GGprenumpy)*sr)
            AUCGGpost=AUCGGpost/(len(GGpostnumpy)*sr)
            ########################################################
            
            GG_full.append(GG)
            GG_trans=GG.transpose()
            GG_map.append(GG_trans)
            GG_pre.append(GGpre)
            GG_post.append(GGpost)
            GG_pre_mean.append(GGpremean)
            GG_post_mean.append(GGpostmean)   
            GG_pre_mean_series=pd.Series(GG_pre_mean)
            GG_post_mean_series=pd.Series(GG_post_mean)
            ###################################################
            AUC_GG_pre.append(AUCGGpre)
            AUC_GG_post.append(AUCGGpost)
            AUC_GG_pre_series=pd.Series(AUC_GG_pre)
            AUC_GG_post_series=pd.Series(AUC_GG_post)
            ###################################################
                
    GG_mean=np.mean(GG_full, axis=0)
    GG_mean=GG_mean.flatten()
    GG_SD=np.std(GG_full, axis=0)
    GG_SD=GG_SD.flatten()
    GG_map=pd.concat(GG_map)
    
    

    zdF_F_full=[]
    zdF_F_map=[]
    zdF_F_pre=[]
    zdF_F_post=[]
    zdF_F_pre_mean=[]
    zdF_F_post_mean=[]
 
    for case in range(1,case+1):
        for i in range(0,n):
            z=len(zdF_F_full)
            z=z+1
            zdF_F=pd.read_csv('zdF_F_c{0}_{1}.csv'.format(case,i+1),header=None)
            zdF_F=zdF_F.rename(columns={0:'epoch {}'.format(z)})
            zdF_Fpre=zdF_F[(t1-t1bar)*sr:(t1*sr)]
            zdF_Fpost=zdF_F[(t1*sr):(t1+t2bar)*sr]
            zdF_Fpremean=np.mean(zdF_Fpre,axis=0)
            zdF_Fpostmean=np.mean(zdF_Fpost,axis=0)
            
            zdF_F_full.append(zdF_F)
            zdF_F_trans=zdF_F.transpose()
            zdF_F_map.append(zdF_F_trans)
            zdF_F_pre.append(zdF_Fpre)
            zdF_F_post.append(zdF_Fpost)
            zdF_F_pre_mean.append(zdF_Fpremean)
            zdF_F_post_mean.append(zdF_Fpostmean)   
            zdF_F_pre_mean_series=pd.Series(zdF_F_pre_mean)
            zdF_F_post_mean_series=pd.Series(zdF_F_post_mean)
                
    zdF_F_mean=np.mean(zdF_F_full, axis=0)
    zdF_F_mean=zdF_F_mean.flatten()
    zdF_F_SD=np.std(zdF_F_full, axis=0)
    zdF_F_SD=zdF_F_SD.flatten()
    zdF_F_map=pd.concat(zdF_F_map)
    
    grouped_bar_graph_amp=pd.concat([EEG_pre_mean_series,EEG_post_mean_series,GG_pre_mean_series,GG_post_mean_series,zdF_F_pre_mean_series,zdF_F_post_mean_series])
    grouped_bar_graph_AUC=pd.concat([EEG_pre_mean_series,EEG_post_mean_series,AUC_GG_pre_series,AUC_GG_post_series,zdF_F_pre_mean_series,zdF_F_post_mean_series])
    
    
     # RUN BLOCK-----FUNCTIONS    
        
    line_plot()
    heat_map()
    bar_graph()
    save_grouped_data()
   
