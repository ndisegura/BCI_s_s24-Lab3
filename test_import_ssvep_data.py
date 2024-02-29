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