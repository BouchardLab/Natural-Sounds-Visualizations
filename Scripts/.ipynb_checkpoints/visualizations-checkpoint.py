#Install and import relevant packages
import numpy as np
from scipy.io.wavfile import read
import h5py
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy import integrate
import IPython.display
from IPython.display import Audio
from IPython.display import display

#Mods
h5_subfolder = 'sound' #Change if sound data is nested in .h5 file under a different name. 
h5_sampling_rate = 192000 #Sampling rate in Hz of sound in .h5 files
psd_log = True #True if you would like PSD plots to be log-scale/dB
ultrasound_threshold = 32000 #Hz threshold above which 90% of the PSD must be to be considered ultrasound
show_power_spec = True #for compplot
play_sounds=True #for compplot



############################################################################################
#Loading in sounds as ndarrays
def processing1(path):
    if path[-3:] == "wav":
        a = read(path)
        f = sf.SoundFile(path)
        data = np.array(a[1],dtype=float)
        fs = f.samplerate
    elif path[-2:] == "h5":
        f = h5py.File(path, 'r')
        sound = f.get(h5_subfolder)
        data = np.array(sound)
    else:
        print('Your filename/path doesn\'t end in \'.wav\' or \'.h5\'')  
    return data

def processing2(path):
    if path[-3:] == "wav":
        a = read(path)
        f = sf.SoundFile(path)
        data = np.array(a[1],dtype=float)
        fs = f.samplerate
    elif path[-2:] == "h5":
        f = h5py.File(path, 'r')
        sound = f.get(h5_subfolder)
        data = np.array(sound)
        fs = h5_sampling_rate
    else:
        print('Your filename/path doesn\'t end in \'.wav\' or \'.h5\'')  
    return data,fs
############################################################################################
#Oscillogram
def display_oscillogram(sound):
    fig = plt.figure()
    plt.plot(np.arange(sound.shape[0]),sound) #plots 0,1,2,3,.. vs. time series data (frames) 
    plt.ylabel("Amplitude (AU)") #Units: arbitrary units propt to voltage
    plt.xlabel("Frame (Sampling Rate: "+str(sampling_rate/1000)+" kHz)")
    plt.title("Oscillogram (Amp vs. Time)")
    plt.show()
#############################################################################################
#Power Spectral Density (PSD)
def PSD(sound,logf=True,logPb=True):
    fs = sampling_rate
    f,P = periodogram(sound,fs=fs) #extract frequency & PSD 

    #Calculated Cum. Integral
    P_cumtrapz = integrate.cumtrapz(P, f, initial=0) #calculate cumultative integral (of PSD)
    P_cumtrapz = P_cumtrapz/np.max(P_cumtrapz)
    
    if logPb==True:
        logP = 10*np.log10(np.abs(P)/np.max(np.max(np.abs(P))))
        logP_cumtrapz = integrate.cumtrapz(logP-logP.min(), f, initial=0) #calculate cumultative integral (of PSD)
        logP_cumtrapz = logP_cumtrapz/np.max(logP_cumtrapz)
        
    #Plotting
    fig, ax = plt.subplots()
    if logPb==True:
        
        lns1 = ax.plot(f[1:], logP[1:], 'b',label='PSD',linewidth = 0.25)
        ax2 = ax.twinx()
    
        lns2 = ax2.plot(f, logP_cumtrapz,'r-',label='Cum. Integral')
        lns3 = ax2.plot([ultrasound_threshold,ultrasound_threshold],[0,np.max(P_cumtrapz)],'g',
                    label = 'Threshold = '+str(ultrasound_threshold/1000)+' kHz')
        
        ax.set_ylabel(r"PSD (dB)", color='b')
        
        below = logP_cumtrapz[np.max(np.where(f <= ultrasound_threshold))]
        above = 1 - below
    else: 
        lns1 = ax.plot(f, P, 'b',label='PSD',linewidth = 0.25)
        ax2 = ax.twinx()
    
        lns2 = ax2.plot(f, P_cumtrapz,'r-',label='Cum. Integral')
        lns3 = ax2.plot([ultrasound_threshold,ultrasound_threshold],[0,np.max(P_cumtrapz)],'g',
                    label = 'Threshold = '+str(ultrasound_threshold/1000)+' kHz')
        
        ax.set_ylabel(r"PSD (V$^2$/Hz)", color='b')
        
        below = P_cumtrapz[np.max(np.where(f <= ultrasound_threshold))]
        above = 1 - below
    
    if logf==True:
        ax.set_xscale("log")
        ax2.set_xscale("log")

    ax2.set_title('PSD Plot, fs = '+str(fs/1000)+' kHz\n'+str(round(below*100))+'% below, '+str(round(above*100))+'% above threshold')
    ax.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Relative Cum.Int.of PSD (%)', color='r')

    #Legend
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=1,facecolor='white', framealpha = 1,prop={'size': 6})
#############################################################################################
#Comparison Plots for 1/2 Sounds, includes spectogram, oscillogram, & PSD
def pulse1(POI_,sampling_rate, m_channel_=False):
    print('Sound:')
    IPython.display.display(Audio(POI_, rate=sampling_rate))
    if m_channel_:
        print('Marker Pulses:')
        display(Audio(m_channel, rate=sampling_rate))

def compplot(sound1,sampling_rate1,
             sound2=0,sampling_rate2=192000,
             vmin1=-100,vmin2=-100,
             logfs=False,logfp=True,logPb=True,
            s1frange=False,s2frange=False):
    
    if show_power_spec == False:
        fig, axs = plt.subplots(2,2,sharex=False,figsize=(15,7),constrained_layout=True,
                                gridspec_kw={'height_ratios': [1,2]})
        k1 = sound1.shape[0]
        k2 = sound2.shape[0]
        x1=(np.arange(k1))/sampling_rate1
        x2=(np.arange(k2))/sampling_rate2
        axs[0,0].plot(x1,sound1)
        axs[0,1].plot(x2,sound2)

        axs[0,0].set_ylabel("Amplitude (AU)") #Units: arbitrary units or decibels
        axs[0,0].set_title("Sound 1: Oscillogram (Amp vs. Time)")
        axs[0,0].set_xlim([0,k1/sampling_rate1])
        
        axs[0,1].set_ylabel("Amplitude (AU)") #Units: arbitrary units or decibels
        axs[0,1].set_title("Sound 2: Oscillogram (Amp vs. Time)")
        axs[0,1].set_xlim([0,k2/sampling_rate2])

        axs[1,0].specgram(sound1,Fs=sampling_rate1,cmap='jet',vmin=vmin1)
        axs[1,0].set_xlabel('Time (s)')
        axs[1,0].set_ylabel("Freq (Hz)")
        axs[1,0].set_title('Sound 1: Spectrogram (Freq vs. Time)')
        
        axs[1,1].specgram(sound2,Fs=sampling_rate2,cmap='jet',vmin=vmin2)
        axs[1,1].set_xlabel('Time (s)')
        axs[1,1].set_ylabel("Freq (Hz)")
        axs[1,1].set_title('Sound 2: Spectrogram (Freq vs. Time)')
        
        if logfs == True: 
            axs[1,0].set_yscale("log")
            axs[1,1].set_yscale("log")
        if s1frange != False:
            axs[1,0].set_ylim(s1frange)
        if s2frange != False:
            axs[1,1].set_ylim(s2frange)
        #if sound2 is None:
        #    axs[0,1].remove()
        #    axs[1,1].remove()
        plt.show()
        if play_sounds:
            pulse1(sound1,sampling_rate1)
            pulse1(sound2,sampling_rate2)
        
    if show_power_spec == True:
        fig, axs = plt.subplots(3,2,sharex=False,figsize=(15,10.5),constrained_layout=True,
                                gridspec_kw={'height_ratios': [1,2,1]})
        k1 = sound1.shape[0]
        k2 = sound2.shape[0]
        x1=(np.arange(k1))/sampling_rate1
        x2=(np.arange(k2))/sampling_rate2
        axs[0,0].plot(x1,sound1)

        axs[0,1].plot(x2,sound2)

        axs[0,0].set_ylabel("Amplitude (AU)") #Units: arbitrary units or decibels
        axs[0,0].set_title("Sound 1: Oscillogram (Amp vs. Time)")
        axs[0,0].set_xlim([0,k1/sampling_rate1])

        axs[0,1].set_ylabel("Amplitude (AU)") #Units: arbitrary units or decibels
        axs[0,1].set_title("Sound 2: Oscillogram (Amp vs. Time)")
        axs[0,1].set_xlim([0,k2/sampling_rate2])

        axs[1,0].specgram(sound1,Fs=sampling_rate1,cmap='jet',vmin=vmin1)
        axs[1,0].set_xlabel('Time (s)')
        axs[1,0].set_ylabel("Freq (Hz)")
        axs[1,0].set_title('Sound 1: Spectrogram (Freq vs. Time)')

        axs[1,1].specgram(sound2,Fs=sampling_rate2,cmap='jet',vmin=vmin2)
        axs[1,1].set_xlabel('Time (s)')
        axs[1,1].set_ylabel("Freq (Hz)")
        axs[1,1].set_title('Sound 2: Spectrogram (Freq vs. Time)')
        if logfs == True: 
            axs[1,0].set_yscale("log")
            axs[1,1].set_yscale("log")
        if s1frange != False:
            axs[1,0].set_ylim(s1frange)
        if s2frange != False:
            axs[1,1].set_ylim(s2frange)
        
        
        sound = sound1
        fs = sampling_rate1
        f,P = periodogram(sound,fs=fs) #extract frequency & PSD 

        #Calculated Cum. Integral
        P_cumtrapz = integrate.cumtrapz(P, f, initial=0) #calculate cumultative integral (of PSD)
        P_cumtrapz = P_cumtrapz/np.max(P_cumtrapz)

        if logPb==True:
            logP = 10*np.log10(np.abs(P)/np.max(np.max(np.abs(P))))
            logP_cumtrapz = integrate.cumtrapz(logP-logP.min(), f, initial=0) #calculate cumultative integral (of PSD)
            logP_cumtrapz = logP_cumtrapz/np.max(logP_cumtrapz)

        #Plotting
        if logPb==True:

            lns1 = axs[2,0].plot(f[1:], logP[1:], 'b',label='PSD',linewidth = 0.25)
            ax4 = axs[2,0].twinx()

            lns2 = ax4.plot(f, logP_cumtrapz,'r-',label='Cum. Integral')
            lns3 = ax4.plot([ultrasound_threshold,ultrasound_threshold],[0,np.max(P_cumtrapz)],'g',
                        label = 'Threshold = '+str(ultrasound_threshold/1000)+' kHz')

            axs[2,0].set_ylabel(r"PSD (dB)", color='b')

            below = logP_cumtrapz[np.max(np.where(f <= ultrasound_threshold))]
            above = 1 - below
        else: 
            lns1 = axs[2,0].plot(f, P, 'b',label='PSD',linewidth = 0.25)
            ax4 = axs[2,0].twinx()

            lns2 = ax4.plot(f, P_cumtrapz,'r-',label='Cum. Integral')
            lns3 = ax4.plot([ultrasound_threshold,ultrasound_threshold],[0,np.max(P_cumtrapz)],'g',
                        label = 'Threshold = '+str(ultrasound_threshold/1000)+' kHz')

            axs[2,0].set_ylabel(r"PSD (V$^2$/Hz)", color='b')

            below = P_cumtrapz[np.max(np.where(f <= ultrasound_threshold))]
            above = 1 - below

        if logfp==True:
            axs[2,0].set_xscale("log")
            ax4.set_xscale("log")

        ax4.set_title('PSD Plot, fs = '+str(fs/1000)+' kHz\n'+str(round(below*100))+'% below, '+str(round(above*100))+'% above threshold')
        axs[2,0].set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Relative Cum.Int.of PSD (%)', color='r')

        #Legend
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax4.legend(lns, labs, loc=1,facecolor='white', framealpha = 1,prop={'size': 6})

        sound=sound2
        fs = sampling_rate2
        f,P = periodogram(sound,fs=fs) #extract frequency & PSD 

        #Calculated Cum. Integral
        P_cumtrapz = integrate.cumtrapz(P, f, initial=0) #calculate cumultative integral (of PSD)
        P_cumtrapz = P_cumtrapz/np.max(P_cumtrapz)

        if logPb==True:
            logP = 10*np.log10(np.abs(P)/np.max(np.max(np.abs(P))))
            logP_cumtrapz = integrate.cumtrapz(logP-logP.min(), f, initial=0) #calculate cumultative integral (of PSD)
            logP_cumtrapz = logP_cumtrapz/np.max(logP_cumtrapz)

        #Plotting
        if logPb==True:

            lns1 = axs[2,1].plot(f[1:], logP[1:], 'b',label='PSD',linewidth = 0.25)
            ax5 = axs[2,1].twinx()

            lns2 = ax5.plot(f, logP_cumtrapz,'r-',label='Cum. Integral')
            lns3 = ax5.plot([ultrasound_threshold,ultrasound_threshold],[0,np.max(P_cumtrapz)],'g',
                        label = 'Threshold = '+str(ultrasound_threshold/1000)+' kHz')

            axs[2,1].set_ylabel(r"PSD (dB)", color='b')

            below = logP_cumtrapz[np.max(np.where(f <= ultrasound_threshold))]
            above = 1 - below
        else: 
            lns1 = axs[2,1].plot(f, P, 'b',label='PSD',linewidth = 0.25)
            ax5 = axs[2,1].twinx()

            lns2 = ax5.plot(f, P_cumtrapz,'r-',label='Cum. Integral')
            lns3 = ax5.plot([ultrasound_threshold,ultrasound_threshold],[0,np.max(P_cumtrapz)],'g',
                        label = 'Threshold = '+str(ultrasound_threshold/1000)+' kHz')

            axs[2,1].set_ylabel(r"PSD (V$^2$/Hz)", color='b')

            below = P_cumtrapz[np.max(np.where(f <= ultrasound_threshold))]
            above = 1 - below

        if logfp==True:
            axs[2,1].set_xscale("log")
            ax5.set_xscale("log")

        ax5.set_title('PSD Plot, fs = '+str(fs/1000)+' kHz\n'+str(round(below*100))+'% below, '+str(round(above*100))+'% above threshold')
        axs[2,1].set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Relative Cum.Int.of PSD (%)', color='r')

        #Legend
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax5.legend(lns, labs, loc=1,facecolor='white', framealpha = 1,prop={'size': 6})
        
        
        all_axes = fig.get_axes()
        for axis in all_axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
                all_axes[-1].add_artist(legend)
        #if sound2 == 0:
         #   axs[0,1].remove()
         #   axs[1,1].remove()
         #   axs[2,1].remove()
         #   ax5.remove()
            
        plt.show()
        if play_sounds:
            pulse1(sound1,sampling_rate)
            pulse1(sound2,sampling_rate)