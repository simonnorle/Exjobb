# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 18:16:47 2025

@author: Simon
"""

import numpy as np
import pandas as pd
time_hours = 8760
Freq_data = r'C:\Users\Simon\Documents\Uppsala\Exjobb\Frequency data 2021 (3min)2.xlsx'
Frequency = np.array(pd.read_excel(Freq_data, usecols="D",skiprows=0))
minutes = np.array(pd.read_excel(Freq_data, usecols="C",skiprows=0))
minutes_copy = minutes
#%%
for m in range(time_hours*20):
    if m < time_hours*20-1 and minutes[m] == 58 and minutes[m+1] != 1:
        minutes = np.insert(minutes, m+1, 1)
        Frequency = np.insert(Frequency, m+1, 0)
    elif m < time_hours*20-1 and minutes[m+1] != minutes[m]+3 and minutes[m] < 58:
        minutes = np.insert(minutes, m+1, minutes[m]+3)
        Frequency = np.insert(Frequency, m+1, 0)
    

    
null_index = [i for i,x in enumerate(Frequency) if x == 0]      

#%%

test = np.array([52,58,4,10,13])
rang = len(test)
freq = np.array([1]*len(test))
for i in range(5):
    if test[i] == 58 and test[i+1] != 1:
        test = np.insert(test, i+1, 1)
        freq = np.insert(freq, i+1, 0)
    elif test[i+1] != test[i]+3:
        test = np.insert(test, i+1, test[i]+3)
        freq = np.insert(freq, i+1, 0)
    
#%%
minute_index = [i for i,x in enumerate(minutes) if x==1] #All indecies for the first minute of each hour
data_interval = 3*60 #Time interval between data points [s] Kan vara en inparameter istället(?)
hourly_data_points = len(Frequency)/time_hours # Fungerar inte att ha en float i range()
FCR_D_count = int()
FCR_U_count = int()
FCR_N_count = int()
Flex_frac_FCR_D = [0]*time_hours # len() fungerar endast om det är en list
Flex_frac_FCR_U = [0]*time_hours
Flex_frac_FCR_N = [0]*time_hours
for i in range(len(minute_index)): # -1 då index börjar från 0. Gäller för int(), men behövs det för len(list)?
    for j in range(minute_index[i],minute_index[i+1]-1): #Behövs -1 i slutet?
        if Frequency[j,0] > 50.1:
            FCR_D_count += 1
        elif Frequency[j,0] < 49.9 and Frequency[j,0] != 0:
            FCR_U_count += 1
        elif 49.9 <= Frequency[j,0] <= 49.999 or 50.001 <= Frequency[j,0] <= 50.1:
            FCR_N_count += 1
        Flex_frac_FCR_D[i] = FCR_D_count/(60*60/data_interval) # Gives fraction of the hour that the flexibility service was operating 
        Flex_frac_FCR_U[i] = FCR_U_count/(60*60/data_interval)
        Flex_frac_FCR_N[i] = FCR_N_count/(60*60/data_interval)
        FCR_D_count = 0
        FCR_U_count = 0 #Resets counters
        FCR_N_count = 0 
