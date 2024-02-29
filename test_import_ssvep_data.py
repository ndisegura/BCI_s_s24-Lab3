# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:25:50 2024

@author: asegura
"""

# Inport the necessry modules
import os
import sys
import matplotlib.pyplot as plt
import import_ssvep_data as ss

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

data_dict=ss.load_ssvep_data(subject,data_directory)

#%% Cell2 Plot the data

ss.plot_raw_data(data_dict,subject,['Fz','Oz'])

#%% Cell3 Epoch the data_dict

#epoch start times
epoch_start_time = 0
epoch_end_time= 20

#epoch data

eeg_epochs,epoch_time,is_trial_15Hz=ss.epoch_ssvep_data(data_dict,epoch_start_time,epoch_end_time)
