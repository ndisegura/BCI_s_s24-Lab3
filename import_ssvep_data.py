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
    
    fig, axs = plt.subplots(2)
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
        
        selected_channel_data=eeg[is_channel_match]
        
        axs[1].plot(eeg_time, np.squeeze(selected_channel_data),label=channel_member)
    axs[1].set_ylabel('Voltage (uV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
        
# ```
# Fields can then be extracted like this:
# ```python
# # extract variables from dictionary
# eeg = data['eeg']
# channels = data['channels']
# fs = ...
# ```