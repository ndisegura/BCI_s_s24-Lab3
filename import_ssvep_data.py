# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:24:41 2024

@author: asegura2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def load_ssvep_data(subject,data_directory):
    data_file=f'{data_directory}SSVEP_S{subject}.npz'
    
    # Load dictionary
    data_dict = np.load(data_file,allow_pickle=True)
    
    return data_dict

    

def plot_raw_data(data,subject,channels_to_plot):

    channels=data['channels']
    eeg=data['eeg']
    fs=data['fs']
    eeg_time=np.arange(0,len(eeg[0])*1/fs,1/fs)
    
    event_samples=data['event_samples']
    event_duration=data['event_durations']
    event_type=data['event_types']
    
    is_channel_match=np.zeros(len(eeg[0]),dtype=bool)
    
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle(f'SSVEP Subject {subject} Raw Data')
    
    #PLot Event types
    
    for event_index, event_freq in enumerate(event_type):
        start_time=eeg_time[event_samples[event_index]]
        end_time=eeg_time[event_samples[event_index]+int(event_duration[event_index])]
        axs[0].plot([start_time,end_time],[event_freq,event_freq], 'b')
    axs[0].set_ylabel('Flash Frequency')
    axs[0].set_xlabel('Time (s)')
    axs[0].grid()
        
    #PLot EEG Data
    for channel_index, channel_member in enumerate(channels_to_plot):
        
        is_channel_match=channels==channel_member #Boolean indexing across rows for item in list
        
        selected_channel_data=eeg[is_channel_match]/10-6 #Divide by 10-6 to obtain voltage in uV
        
        axs[1].plot(eeg_time, np.squeeze(selected_channel_data),label=channel_member)
    axs[1].set_ylabel('Voltage (uV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
        


#%%

def epoch_ssvep_data(data,epoch_start_time,epoch_end_time):
    
    channels=data['channels']
    eeg=data['eeg']
    fs=data['fs']
    eeg_time=np.arange(0,len(eeg[0])*1/fs,1/fs)

    event_samples=data['event_samples']
    event_duration=data['event_durations']
    event_type=data['event_types']

    channel_count= len(channels)
    event_count = len(event_samples)


    samples_per_second = int(1/(eeg_time[1]-eeg_time[0]))
    seconds_per_epoch = epoch_end_time-epoch_start_time
    samples_per_epoch = int(samples_per_second * seconds_per_epoch)

    #create empty epoch array
    eeg_epochs = np.zeros((event_count, channel_count, samples_per_epoch))

    for event_index, event in enumerate(event_samples):
        
        # get eeg data_dict within the timebounds of the event
        data_to_add = eeg[:,event:event+samples_per_epoch]
        
        # add eeg data_dict into epoch
        eeg_epochs[event_index,:,:] = data_to_add

    #get time relative to each event 
    epoch_time = eeg_time[:samples_per_epoch]
    
    #create boolean array true if 15hz flash during epoch
    is_trial_15Hz = event_type== '15hz'
  
    return(eeg_epochs,epoch_time,is_trial_15Hz)
    
def get_frequency_spectrum(eeg_epochs,fs):
    
    # Take FFT of signal
    eeg_epochs_fft=np.fft.rfft(eeg_epochs)
    #Compute FFT Magnitude from Complex values
    eeg_epochs_fft_magnitude=np.absolute(eeg_epochs_fft-eeg_epochs_fft)
    #Compute Frequencies
    fft_frequencies=np.arange(0,fs/2,(fs/2)/eeg_epochs_fft_magnitude.shape[2])
    
    return eeg_epochs_fft,fft_frequencies 

def plot_power_spectrum(eeg_epochs_fft,fft_frequencies,is_trial_15Hz,channels,channels_to_plot,subject=1):
    
    #Find the 12Hz trials
    is_trial_12Hz=is_trial_15Hz==False
    #separate 12Hz and 15Hz epochs
    eeg_epochs_fft_12Hz=eeg_epochs_fft[is_trial_12Hz]
    eeg_epochs_fft_15Hz=eeg_epochs_fft[is_trial_15Hz]
    
    #Compute FFT Magnitude from Complex values for 12Hz
    eeg_epochs_fft_magnitude_12hz=np.absolute(eeg_epochs_fft_12Hz)
    eeg_epochs_fft_magnitude_15hz=np.absolute(eeg_epochs_fft_15Hz)
    
    
    
    
