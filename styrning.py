#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:20:11 2025

@author: simonnorle
"""

import numpy as np
import pandas as pd
import parameters as param

from bisect import bisect_left

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before
    
    
# Alternativ formulering som beräknar för alla timmar för en dag, designad för att anropas på rad 189 i simulation.py
def styrning2_alt(
        i1: int, # Int, first hour of the day relative to the entire year
        H2_demand: list, #List, the hourly hydrogen demand [kg/h] istället, är det elz_dispatch.iloc[:,6] "Demand"?
        H2_storage: list, #List, the hourly amount of hydrogen in storage [kg], bör vara  h2_storage_list[i1:i2]
        h2st_size_vector: list, #List with one value, the maximum hydrogen storage capacity [kg]
        P_bau: list, # List of the "default" electrolyzer power [MW], bör vara electrolyzer[i1:i2]
        P_pv: list, #List of the hourly generated power from the PV panels [kW]
        P_max: float, #Float, maximum rated power of the electrolyzer [MW]
        h2_prod: list, #List, H2 production at a given partial load ranging from 0-100% with step size 1/len(h2_prod)
        FCR_D: bool, # Boolean, True if participating as FCR-D down, otherwise False
        FCR_U: bool, # Boolean, True if participating as FCR-D up, otherwise False
        FCR_N: bool, # Boolean, True if participating as FCR-N, otherwise False, ska denna vara med?
        Flex_frac_FCR_D: list, #List, the hourly fraction of the time the flexibility resource FCR_D was active
        Flex_frac_FCR_U: list, #List, the hourly fraction of the time the flexibility resource FCR_U was active
        FCR_D_power: list, #List, the hourly power demand for FCR-D down in bidding region SE3
        FCR_D_price: list, #List, the hourly price/MW for FCR-D down in bidding region SE3
        FCR_U_power: list, #List, the hourly power demand for FCR-D up in bidding region SE3
        FCR_U_prize: list,  #List, the hourly price/MW for FCR-D up in bidding region SE3
):
    electrolyzer = [0]*24 # Creates list of same size (no of hours in a day)
    P_activated = [0]*24
    P_reserved = [0]*24
    Income_flex = [0]*24 
    deltaP_max = [0]*24
    H2_produced = [0]*24
    H2_max = [0]*24
    H2_bau = [0]*24
    H2_flex = [0]*24
    FCR_D_orginal = FCR_D
    FCR_U_orginal = FCR_U
    FCR_N_orginal = FCR_N
    GRÄNS = 0
    
    for i in range(int(24)): #endast för en dag, så 24
        FCR_D = FCR_D_orginal #Resets boolean value to the imput value
        FCR_U = FCR_U_orginal
        index_max = [j for j,x in enumerate(H2_storage) if x >= h2st_size_vector[0]] #Returns list of indices where the storage is full
        #FCR_D = False if bool(index_max) == True and i <= index_max[-1] else FCR_D # Disables FCR-D down as long as the planned storage is full, enables FCR-D down once the storage won't be full for the rest of the day
        if bool(index_max) == True and i <= index_max[-1]:
            FCR_D = False
        else:
            FCR_D = FCR_D_orginal
        if H2_storage[i] >= H2_demand[i]: #Branch 1
            FCR_D = False if H2_storage[i] == h2st_size_vector[0] else FCR_D == FCR_D_orginal #Borde fungera som en rad, No increased production if the storage is full
            if FCR_D == True and FCR_U == False: #Behöver det finnas en övre gräns för P_ele (finns realistiskt iaf)?
                H2_max[i] = h2st_size_vector[0] + H2_storage[i] - H2_demand[i] ##Alternativt uttryck från vätgasbalans
                H2_closest = take_closest(h2_prod, H2_max[i])
                H2_index = [k for k,x in enumerate(h2_prod) if H2_closest == h2_prod[k]] #Finds the index where the produced hydrogen is the same, requires uniqe values(?)
                P_H2_max = P_max * (H2_index[0]+1) / len(h2_prod) 
                # P_H2_max = ([k for k,x in enumerate(h2_prod) if H2_max[i] == h2_prod[k]]+1)/len(h2_prod)*P_max #Alternativ omskrivning
                deltaP_max[i] = P_H2_max - P_bau[i]
                P_reserved[i] = deltaP_max[i] # Maximum possible power reserved as frequency regulation
                Income_flex[i] = P_reserved[i] * FCR_D_price[i1+i]
                P_activated[i] = deltaP_max[i] * Flex_frac_FCR_D[i1+i] 
                electrolyzer[i] = P_bau[i] + P_activated[i]
                H2_flex[i] = h2_prod[round(len(h2_prod) * electrolyzer[i] / P_max)] #Hydrogen production at given average partial load
                H2_bau[i] = h2_prod[round(len(h2_prod) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                H2_storage = [x + H2_produced[i] if j > index_max[-1] else x for j,x in enumerate(H2_storage)] # Increases storage for all indices after the last time the storage is full 
            elif FCR_U == True and FCR_D == False:         #Gällande ovan rad, P_bau ska kanske vara tidigare stegs effekt P_ele [i-1]?
                 if P_pv[i1+i] >= GRÄNS: # Gränsvärde för att undvika kallstarter om P_pv = 0.
                     P_reserved[i] = - (P_bau[i] - P_pv[i1+i]) #Reglerar ned så mycket som möjligt, med P_pv som undre gräns. Negativ för att indikera minskning
                     Income_flex[i] = abs(P_reserved[i] * FCR_U_prize[i1+i]) #Income is positive
                     P_activated[i] = P_reserved[i] * Flex_frac_FCR_U[i1+i] # Negative as it's a reduction in used energy
                     electrolyzer[i] = P_bau[i] + P_activated[i] #Average electrolyzer power usage for the hour
                     H2_flex[i] = h2_prod[round(len(h2_prod) * electrolyzer[i] / P_max)] #Hydrogen production at given average partial load
                     H2_bau[i] = h2_prod[round(len(h2_prod) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                     H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                     H2_storage = [x + H2_produced[i] if j>= i else x for j,x in enumerate(H2_storage)] #Adds to all future storage, not retroactively
                 else: #P_pv[i1+1] < GRÄNS
                     P_reserved[i] = - (P_bau[i] - GRÄNS) #Negative as it regulates down
                     Income_flex[i] = abs(P_reserved[i] * FCR_U_prize[i1+i])
                     P_activated[i] = P_reserved[i] * Flex_frac_FCR_U[i1+i] # Negative as it's a reduction in used energy
                     electrolyzer[i] = P_bau[i] + P_activated[i] #Average electrolyzer power usage for the hour
                     H2_produced[i] = h2_prod[len(h2_prod) * electrolyzer[i] / P_max] # Actual hydrogen production at given average partial load  
                     H2_flex[i] = h2_prod[round(len(h2_prod) * electrolyzer[i] / P_max)] #Hydrogen production at given average partial load
                     H2_bau[i] = h2_prod[round(len(h2_prod) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                     H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                     H2_storage = [x + P_activated[i] * H2_produced[i] if j>= i else x for j,x in enumerate(H2_storage)] #Adds to all future storage, not retroactively
            else:
                  electrolyzer[i] = P_bau[i] #Om inte på stödtjänstmarknaden gäller business as usual
              
        else: #Branch 2, implicit H2_storage[i] < H2_demand[i]
            if FCR_D == True and FCR_U == False:
                #Styrningen kommer nog se till att metaniseringsbehovet alltid täcks, så P_bau räcker i basfallet
                # Ska denna vara nästan samma som FCR_D i första grenen?
                H2_max[i] = h2st_size_vector[0] + H2_demand[i] - H2_storage[i] # Maximum possible hourly increase in hydrogen production [kg]
                
                if H2_max[i] <= max(h2_prod): #Checks if the theoretical maximum hydrogen production is possible in this system
                    H2_closest = take_closest(h2_prod, H2_max[i])
                    H2_index = [k for k,x in enumerate(h2_prod) if H2_closest == h2_prod[k]] #Finds the index where the produced hydrogen is the same, requires uniqe values(?)
                    P_H2_max = H2_index[0]+1 / len(h2_prod) * P_max
                else:
                    P_H2_max = P_max #Otherwise upper technical limit
                deltaP_max[i] = P_H2_max - P_bau[i]
                P_reserved[i] = deltaP_max[i] # Maximum possible power reserved as frequency regulation
                Income_flex[i] = P_reserved[i] * FCR_D_price[i1+i]
                P_activated[i] = deltaP_max[i] * Flex_frac_FCR_D[i1+i] 
                electrolyzer[i] = P_bau[i] + P_activated[i] #Average electrolyzer power usage for the hour
                H2_flex[i] = h2_prod[round(len(h2_prod) * electrolyzer[i] / P_max)] #Hydrogen production at given average partial load
                H2_bau[i] = h2_prod[round(len(h2_prod) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                H2_storage = [x + H2_produced[i] if j > index_max[-1] else x for j,x in H2_storage] # Increases storage for all indices after the last time the storage is full 
            
            elif FCR_U == True and FCR_D == False:
                H2_max[i] = H2_demand[i] - H2_storage[i] #Ska kanske heta H2_min?
                H2_closest = take_closest(h2_prod, H2_max[i])
                H2_index = [k for k,x in enumerate(h2_prod) if H2_closest == h2_prod[k]] #Finds the index where the produced hydrogen is the same, requires uniqe values(?)
                P_H2_max = P_max * (H2_index[0]+1) / len(h2_prod) 
                if P_H2_max < P_pv[i1+1] and P_pv[i1+i] >= GRÄNS:
                    deltaP_max[i] = P_pv[i1+i]
                elif P_H2_max < GRÄNS and P_pv[i1+i] < GRÄNS:
                    deltaP_max[i] = GRÄNS
                else:
                    deltaP_max[i] = P_H2_max  # The minimal electrolyzer power that leaves h2_storage non-negative (at zero)
                P_reserved[i] = -(P_bau[i] - deltaP_max[i])
                Income_flex[i] = abs(P_reserved[i] * FCR_U_prize[i1+i])
                P_activated[i] = P_reserved[i] * Flex_frac_FCR_U[i1+i]
                electrolyzer[i] = P_bau[i] + P_activated[i] #Average electrolyzer power usage for the hour
                H2_flex[i] = h2_prod[round(len(h2_prod) * electrolyzer[i] / P_max)] #Hydrogen production at given average partial load
                H2_bau[i] = h2_prod[round(len(h2_prod) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                H2_storage = [x + H2_produced[i] if j>= i else x for j,x in enumerate(H2_storage)] #Adds to all future storage, not retroactively
        
            else:
                electrolyzer[i] = P_bau[i]
    return electrolyzer, Income_flex, P_activated, H2_storage    #Används senare till beräkningar


# %%
import numpy as np
import pandas as pd
import parameters as param
pem2 = param.Electrolyzer(10) #Alternate electrolyzer for efficiency curve
pem2.efficiency('No plot', 10000) # For the purpose of higher resolution H2-efficiency
FCR_read = r'C:\Users\Simon\Documents\Uppsala\Exjobb\FCR_2021.xlsx' #Onödigt att läsa in dessa hela tiden, kanske flytta utanför?
FCR_D_power = np.array(pd.read_excel(FCR_read,usecols="T",skiprows=0))
FCR_U_power = np.array(pd.read_excel(FCR_read,usecols="M",skiprows=0))
FCR_D_price = np.array(pd.read_excel(FCR_read,usecols="P",skiprows=0))
FCR_U_price = np.array(pd.read_excel(FCR_read,usecols="I",skiprows=0))
#FCR_N_power = np.array(pd.read_excel(FCR_read,usecols="F",skiprows=0))
#FCR_N_price = np.array(pd.read_excel(FCR_read,usecols="B",skiprows=0))
test = styrning2_alt(i1=0, H2_demand=[50]*24, H2_storage=[150]*24, h2st_size_vector=[300], P_bau=[5]*24, P_pv=[0.5]*24, P_max=10, h2_prod=pem2.h2_prod, FCR_D=True, FCR_U=True, FCR_N=False, Flex_frac_FCR_D = [0.1]*24, Flex_frac_FCR_U=[0]*24, FCR_D_power=FCR_D_power, FCR_D_price=FCR_D_price, FCR_U_power=FCR_U_power, FCR_U_prize=FCR_U_price)
# %% 

def FCR_data(
        time_hours: int, #Int, number of hours for the given year
        ):
    FCR_read = r'C:\Users\Simon\Documents\Uppsala\Exjobb\FCR_2021.xlsx' #Onödigt att läsa in dessa hela tiden, kanske flytta utanför?
    FCR_D_power = np.array(pd.read_excel(FCR_read,usecols="T",skiprows=0))
    FCR_U_power = np.array(pd.read_excel(FCR_read,usecols="M",skiprows=0))
    FCR_D_price = np.array(pd.read_excel(FCR_read,usecols="P",skiprows=0))
    FCR_U_price = np.array(pd.read_excel(FCR_read,usecols="I",skiprows=0))
    FCR_N_power = np.array(pd.read_excel(FCR_read,usecols="F",skiprows=0))
    FCR_N_price = np.array(pd.read_excel(FCR_read,usecols="B",skiprows=0))
    Freq_data = r'C:\Users\Simon\Documents\Uppsala\Exjobb\Frequency data 2021 (3min).xlsx'
    Frequency = np.array(pd.read_excel(Freq_data, usecols="C",skiprows=0))
    data_interval = 3*60 #Time interval between data points [s] Kan vara en inparameter istället(?)
    hourly_data_points = len(Frequency)/time_hours # Fungerar inte att ha en float i range()
    FCR_D_count = int()
    FCR_U_count = int()
    FCR_N_count = int()
    Flex_frac_FCR_D = [0]*time_hours # len() fungerar endast om det är en list
    Flex_frac_FCR_U = [0]*time_hours
    Flex_frac_FCR_N = [0]*time_hours
    for i in range(time_hours-1): # -1 då index börjar från 0. Gäller för int(), men behövs det för len(list)?
        for j in range(i*hourly_data_points,(i+1)*hourly_data_points-1): #Behövs -1 i slutet?
            if Frequency[j] > 50.1:
                FCR_D_count += 1
            elif Frequency[j] < 49.9:
                FCR_U_count += 1
            elif 49.9 <= Frequency[j] <= 49.999 or 50.001 <= Frequency[j] <= 50.1:
                FCR_N_count += 1
        Flex_frac_FCR_D[i] = FCR_D_count/(60*60/data_interval) # Gives fraction of the hour that the flexibility service was operating 
        Flex_frac_FCR_U[i] = FCR_U_count/(60*60/data_interval)
        Flex_frac_FCR_N[i] = FCR_N_count/(60*60/data_interval)
        FCR_D_count = 0
        FCR_U_count = 0 #Resets counters
        FCR_N_count = 0 
    return Flex_frac_FCR_D, Flex_frac_FCR_U, Flex_frac_FCR_N, FCR_D_power, FCR_D_price, FCR_U_power, FCR_U_price, FCR_N_power, FCR_N_price

# %%
import numpy as np
import pandas as pd
time_hours = 8760
Freq_data = r'C:\Users\Simon\Documents\Uppsala\Exjobb\Frequency data 2021 (3min)2.xlsx'
Frequency = np.array(pd.read_excel(Freq_data, usecols="D",skiprows=0))
minutes = np.array(pd.read_excel(Freq_data, usecols="C",skiprows=0))
minutes_copy = minutes
for m in range(time_hours*20):
    if m < time_hours*20-1 and minutes[m+1] != minutes[m]+3 and minutes[m] < 58:
        minutes = np.insert(minutes, m+1, minutes[m]+3)
        Frequency = np.insert(Frequency, m+1, 0)
    elif m < time_hours*20-1 and minutes[m] == 58 and minutes[m+1] != 1:
        minutes = np.insert(minutes, m+1, 1)
        Frequency = np.insert(Frequency, m+1, 0)

    
null_index = [i for i,x in enumerate(Frequency) if x == 0]     