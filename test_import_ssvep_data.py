# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:25:50 2024

@author: asegura
"""

# Inport the necessry modules
import os
import sys
import matplotlib.pyplot as plt
import import_ssvep_data

#Make sure relative path work
cwd=os.getcwd()
sys.path.insert(0,f"{cwd}\course_software\SsvepData\\")

#Close previosly drawn plots
plt.close('all')

#Build data file string
data_directory=f'{cwd}/course_software/SsvepData/'
subject=1
#data_file=f'{cwd}{data_directory}SSVEP_S{subject}.npz'

#%% Cell1 Load the Data

data_dict=import_ssvep_data.load_ssvep_data(subject,data_directory)

#%% Cell2 Plot the data

import_ssvep_data.plot_raw_data(data_dict,subject,['Fz','Oz'])

#%% Cell3 Epoch the data_dict

#epoch start times
epoch_start_time = 0
epoch_end_time= 20

#epoch data

eeg_epochs,epoch_time,is_trial_15Hz=import_ssvep_data.epoch_ssvep_data(data_dict,epoch_start_time,epoch_end_time)

#%% Cell4 Compute FFT

eeg_epochs_fft,fft_frequencies=import_ssvep_data.get_frequency_spectrum(eeg_epochs,data_dict['fs'])

#%% Cell 5: Plot the Power Spectra

channels=data_dict['channels']
channels_to_plot=['Fz','Oz']
spectrum_db_12Hz,spectrum_db_15Hz=import_ssvep_data.plot_power_spectrum(eeg_epochs_fft,fft_frequencies,is_trial_15Hz,channels,channels_to_plot,subject)
