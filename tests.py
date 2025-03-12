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
data_interval = 3*60
FCR_D_count = int(0)
FCR_U_count = int(0)
FCR_N_count = int(0)
Flex_frac_FCR_D = [0]*time_hours # len() fungerar endast om det är en list
Flex_frac_FCR_U = [0]*time_hours
Flex_frac_FCR_N = [0]*time_hours
for i in range(time_hours): # -1 då index börjar från 0. Gäller för int(), men behövs det för len(list)?
    for j in range(i*20, (i+1)*20-1): #Behövs -1 i slutet?
        if Frequency[j] > 50.1:
            FCR_D_count += 1
        elif Frequency[j] < 49.9 and Frequency[j] != 0:
            FCR_U_count += 1
        elif 49.9 <= Frequency[j] <= 49.999 or 50.001 <= Frequency[j] <= 50.1:
            FCR_N_count += 1
        Flex_frac_FCR_D[i] = FCR_D_count/(60*60/data_interval) # Gives fraction of the hour that the flexibility service was operating 
        Flex_frac_FCR_U[i] = FCR_U_count/(60*60/data_interval)
        Flex_frac_FCR_N[i] = FCR_N_count/(60*60/data_interval)
        FCR_D_count = 0
        FCR_U_count = 0 #Resets counters
        FCR_N_count = 0 
FCR_D_nonzero = [i for i,x in enumerate(Flex_frac_FCR_D) if x != 0]
index_FCR_D = [i for i,x in enumerate(Frequency) if x > 50.1]
index_FCR_U = [i for i,x in enumerate(Frequency) if x < 49.9 and x != 0]
FCR_N_nonzero = [i for i,x in enumerate(Flex_frac_FCR_N) if x != 0]
FCR_U_nonzero = [i for i,x in enumerate(Flex_frac_FCR_U) if x != 0]

#%%
Freq_dataHz = r'C:\Users\Simon\Documents\Uppsala\Exjobb\Freq 10 Hz\2021-01\2021-01-01.csv'
Freq_10Hz = np.array(pd.read_excel(Freq_dataHz, usecols="C"))

data_interval = 3*60
FCR_D_count = int(0)
FCR_U_count = int(0)
FCR_N_count = int(0)
Flex_frac_FCR_D = [0]*time_hours # len() fungerar endast om det är en list
Flex_frac_FCR_U = [0]*time_hours
Flex_frac_FCR_N = [0]*time_hours
for i in range(time_hours): # -1 då index börjar från 0. Gäller för int(), men behövs det för len(list)?
    for j in range(i*20, (i+1)*20-1): #Behövs -1 i slutet?
        if Frequency[j] > 50.1:
            FCR_D_count += 1
        elif Frequency[j] < 49.9 and Frequency[j] != 0:
            FCR_U_count += 1
        elif 49.9 <= Frequency[j] <= 49.999 or 50.001 <= Frequency[j] <= 50.1:
            FCR_N_count += 1
        Flex_frac_FCR_D[i] = FCR_D_count/(60*60/data_interval) # Gives fraction of the hour that the flexibility service was operating 
        Flex_frac_FCR_U[i] = FCR_U_count/(60*60/data_interval)
        Flex_frac_FCR_N[i] = FCR_N_count/(60*60/data_interval)
        FCR_D_count = 0
        FCR_U_count = 0 #Resets counters
        FCR_N_count = 0 
#%%
first_minutes = [i for i,x in enumerate(minutes) if x == 1]
diff_minutes = []
for i in range(len(minutes)):
    diff_minutes.append(minutes[i+1]-minutes[i])
#%%
diff_index = [i for i,x in enumerate(diff_minutes) if x != 3 and x != -57]

for d in range(len(diff_index)):
    Frequency = np.insert(Frequency, diff_index[d], 0)
    minutes = np.insert(minutes, diff_index[d], 100)
#%%

test = np.array([52,4,10,13])
rang = len(test)
freq = np.array([1]*len(test))
for i in range(7):
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

#%% 
import styrning as st
import parameters as params
pem2 = params.Electrolyzer(10) #Alternate electrolyzer for efficiency curve
pem2.efficiency('No plot', 10000) # For the purpose of higher resolution H2-efficiency
FCR_read = r'C:\Users\Simon\Documents\Uppsala\Exjobb\FCR_2021.xlsx' #Onödigt att läsa in dessa hela tiden, kanske flytta utanför?
FCR_D_power = np.array(pd.read_excel(FCR_read,usecols="T",skiprows=0))
FCR_U_power = np.array(pd.read_excel(FCR_read,usecols="M",skiprows=0))
FCR_D_price = np.array(pd.read_excel(FCR_read,usecols="P",skiprows=0))
FCR_U_price = np.array(pd.read_excel(FCR_read,usecols="I",skiprows=0))
#FCR_N_power = np.array(pd.read_excel(FCR_read,usecols="F",skiprows=0))
#FCR_N_price = np.array(pd.read_excel(FCR_read,usecols="B",skiprows=0))
test = st.styrning2_alt(i1=0, H2_demand=[50]*24, H2_storage=[150]*24, h2st_size_vector=[300], P_bau=[5]*24, P_pv=[0.5]*24, P_max=10, h2_prod=pem2.h2_prod, FCR_D=True, FCR_U=False, FCR_N=False, Flex_frac_FCR_D = [0.1]*24, Flex_frac_FCR_U=[0]*24, FCR_D_power=FCR_D_power, FCR_D_price=FCR_D_price, FCR_U_power=FCR_U_power, FCR_U_prize=FCR_U_price)