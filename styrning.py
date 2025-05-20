#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:20:11 2025

@author: simonnorle
"""

import numpy as np
import pandas as pd
# import parameters as param

from bisect import bisect_left
# import time
# start = time.time()
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
 

def FCR_data(year
        ):
    #FCR_read = r'C:\Users\Simon\Documents\Uppsala\Exjobb\FCR_2021.xlsx' #Onödigt att läsa in dessa hela tiden, kanske flytta utanför?
   
    FCR_read = r'/Users/simonnorle/Documents/Uppsala - år 5/Exjobb/FCR_'+ str(year) +'.xlsx'

    FCR_D_power = np.array(pd.read_excel(FCR_read,usecols="T",skiprows=0))
    FCR_U_power = np.array(pd.read_excel(FCR_read,usecols="M",skiprows=0))
    FCR_D_price = np.array(pd.read_excel(FCR_read,usecols="P",skiprows=0))
    FCR_U_price = np.array(pd.read_excel(FCR_read,usecols="I",skiprows=0))
    FCR_N_power = np.array(pd.read_excel(FCR_read,usecols="F",skiprows=0))
    FCR_N_price = np.array(pd.read_excel(FCR_read,usecols="B",skiprows=0))
    
    return FCR_D_power, FCR_D_price, FCR_U_power, FCR_U_price, FCR_N_power, FCR_N_price

def Frequency(
        year,
        day):


    # year = 2022
    # day = 1
    import datetime
   # import time    #Used for benchmark testing
    import numpy as np
    import pandas as pd
    #start = time.time()
    date = datetime.date(year, 1, 1) + datetime.timedelta(day)
    folder_date_str = date.strftime("%Y-%m")
    file_date_str = date.strftime("%Y-%m-%d")

    time_hours = 24
    if folder_date_str == "2023-06" or folder_date_str == "2023-07" or folder_date_str == "2023-08" or folder_date_str == "2023-11" or folder_date_str == "2023-12":
        Freq_data = r'/Users/simonnorle/Documents/Uppsala - år 5/Exjobb/'+str(year)+' 10Hz/' + str(folder_date_str) + '/' + 'Taajuusdata'+ str(file_date_str) +'.csv'
    else:
        Freq_data = r'/Users/simonnorle/Documents/Uppsala - år 5/Exjobb/'+str(year)+' 10Hz/' + str(folder_date_str) + '/' + str(file_date_str) +'.csv' #Finns kanske ett bättre sätt att göra detta
    Frequency = np.array(pd.read_csv(Freq_data,sep=',',usecols=[1]))

    ### This part is for identifying indices where data is missing
    time_data = np.array(pd.read_csv(Freq_data,sep=',',usecols=[0])) #necessary duplicate
    time_date = []
    missing_delta_t = []
    missing_index = []
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    for i in range(len(time_data)):
        time_date.append(datetime.datetime.strptime(time_data[i][0], date_format))
    for t in range(len(time_date)-1):
        if time_date[t+1]-time_date[t] != datetime.timedelta(microseconds=100000):
            delta_t = (time_date[t+1]-time_date[t]).seconds*1000 + (time_date[t+1]-time_date[t]).microseconds/1000 # Float, [ms]
            missing_delta_t.append(delta_t)
            missing_index.append(t+1)
            for s in range(int(delta_t/100)-1):
                Frequency = np.insert(Frequency,t+1+s, (Frequency[t+1+s]-Frequency[t+s])/2) #Inserts linear interpolation value at missing index

    data_interval = 0.1 #Time interval between data points [s] Kan vara en inparameter istället(?)
    hourly_data_points = 10*60*60 # Fungerar inte att ha en float i range()
    
    FCR_D_index = []
    FCR_U_index = []
    FCR_N_index = []
    Flex_frac_FCR_D = [0]*time_hours # len() fungerar endast om det är en list
    Flex_frac_FCR_U = [0]*time_hours
    Flex_frac_FCR_N = [0]*time_hours
    if len(Frequency) < hourly_data_points*time_hours: # needed to add this branch in order to handle days where measurements ended early
        FCR_D_count = np.zeros(hourly_data_points*24)
        FCR_U_count = np.zeros(hourly_data_points*24)
        FCR_N_count = np.zeros(hourly_data_points*24)
        for j in range(len(Frequency)):
            if Frequency[j] > 50.1:
                FCR_D_count[j] = 1
                FCR_D_index.append(j)
            elif Frequency[j] < 49.9:
                FCR_U_count[j] = 1
                FCR_U_index.append(j)
            elif 49.9 <= Frequency[j] <= 49.999 or 50.001 <= Frequency[j] <= 50.1:
                FCR_N_count[j] = 1
                #FCR_N_index.append(j)
        for i in range(time_hours):
            Flex_frac_FCR_D[i] = sum(FCR_D_count[i*hourly_data_points:(i+1)*hourly_data_points-1])/(60*60/data_interval)
            Flex_frac_FCR_U[i] = sum(FCR_U_count[i*hourly_data_points:(i+1)*hourly_data_points-1])/(60*60/data_interval)
            Flex_frac_FCR_N[i] = sum(FCR_N_count[i*hourly_data_points:(i+1)*hourly_data_points-1])/(60*60/data_interval)
    else:
        FCR_D_count = int()
        FCR_U_count = int()
        FCR_N_count = int()
        for i in range(time_hours): # -1 då index börjar från 0. Gäller för int(), men behövs det för len(list)?
            for j in range(i*hourly_data_points,(i+1)*hourly_data_points-1): #Behövs -1 i slutet?
                if Frequency[j] > 50.1:
                    FCR_D_count += 1
                    FCR_D_index.append(j)
                elif Frequency[j] < 49.9:
                    FCR_U_count += 1
                    FCR_U_index.append(j)
                elif 49.9 <= Frequency[j] <= 49.999 or 50.001 <= Frequency[j] <= 50.1:
                    FCR_N_count += 1
                    #FCR_N_index.append(j)
            Flex_frac_FCR_D[i] = FCR_D_count/(60*60/data_interval) # Gives fraction of the hour that the flexibility service was operating 
            Flex_frac_FCR_U[i] = FCR_U_count/(60*60/data_interval)
            Flex_frac_FCR_N[i] = FCR_N_count/(60*60/data_interval)
            FCR_D_count = 0
            FCR_U_count = 0 #Resets counters
            FCR_N_count = 0 
    
        
        
    Flex_results = pd.DataFrame({"Flex_frac_FCR_D": Flex_frac_FCR_D,
                                  "Flex_frac_FCR_U": Flex_frac_FCR_U,
                                  "Flex_frac_FCR_N": Flex_frac_FCR_N
                                  })
    return Flex_frac_FCR_D, Flex_frac_FCR_U, Flex_frac_FCR_N, Flex_results, len(Frequency) , len(missing_index)


def Flex_frac_to_excel(hrs, #Number of hours in the current year, specified earlier in simulation.py
                       year):
    import numpy as np
    Flex_frac_FCR_D = np.zeros(hrs)
    Flex_frac_FCR_U = np.zeros(hrs)
    total_freq_len = 0
    total_freq_missing = 0
    for i in range(int(hrs/24)):
        Freq = Frequency(year=year, day=i)
        i1 = i*24
        i2 = i1+24
        Flex_frac_FCR_D[i1:i2] = Freq[0]
        Flex_frac_FCR_U[i1:i2] = Freq[1]
        # Flex_frac_FCR_D[i1:i2] = Frequency.iloc[:,0]
        # Flex_frac_FCR_U[i1:i2] = Frequency.iloc[:,1]
        total_freq_len += Freq[4]
        total_freq_missing += Freq[5]
        print("Calculations: " + str(i+1))
    Flex_to_excel = pd.DataFrame({"Flex_frac_FCR_D": Flex_frac_FCR_D,
                                  "Flex_frac_FCR_U": Flex_frac_FCR_U
                                  })
    Flex_to_excel.to_excel("Flex_frac_"+str(year)+".xlsx")
    total_frac_missing_freq = total_freq_missing / total_freq_len
    return Flex_to_excel, total_freq_len, total_freq_missing, total_frac_missing_freq


    
def styrning3(
        i1: int, # Int, first hour of the day relative to the entire year
        version: str, #Which version of the code is used
        H2_demand: list, #List, the hourly hydrogen demand [kg/h] istället, är det elz_dispatch.iloc[:,6] "Demand"?
        H2_storage: list, #List, the hourly amount of hydrogen in storage [kg], bör vara  h2_storage_list[i1:i2]
        H2_bau: list , #List, hourly hydrogen production from the day-ahead dispatch
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
        FCR_U_price: list, #List, the hourly price/MW for FCR-D up in bidding region SE3
        min_load: float,    #Minimal electrolyzer load in order to avoid cold starts
        activation_duration: int #Worst-case activation duration for the flexibility resources [minutes]
) -> pd.DataFrame:
    # Creating lists of size = number of hours in a day
    P_flex = [0]*24 
    P_activated_max = [0]*24
    P_activated_min = [0]*24
    P_activated = [0]*24
    P_reserved_max = [0]*24
    P_reserved_min = [0]*24
    P_H2_max = [0]*24
    Income_flex = [0]*24 
    Income_FCR_D = [0]*24 
    Income_FCR_U = [0]*24 
    #deltaP_max = [0]*24
    H2_produced = [0]*24
    deltaH2_max = [0]*24
    deltaH2_min = [0]*24
    H2_max = [0]*24
    H2_min = [0]*24
    H2_bau_calc = [0]*24
    H2_flex_FCR_D = [0]*24
    H2_flex_FCR_U = [0]*24
    Branch_min = [0]*24
    Branch_D = [0]*24
    Branch_U = [0]*24
    FCR_D_record = [0]*24
    FCR_U_record = [0]*24
    index_max_list = [0]*24
    index_min_list = [0]*24
    # Creating copies of input booleans
    FCR_D_orginal = FCR_D
    FCR_U_orginal = FCR_U
    FCR_N_orginal = FCR_N
    # Creating necessary floats
    LIMIT = min_load*P_max
    P_flex_FCR_D = float()
    P_flex_FCR_U = float()
    
    for i in range(int(24)): #endast för en dag, så 24
        if i == 0:
            P_flex[i] = P_bau[0]
        else:    
            FCR_D = FCR_D_orginal #Resets boolean value to the imput value
            FCR_U = FCR_U_orginal
            index_max = [j for j,x in enumerate(H2_storage) if x >= h2st_size_vector[0]] #Returns list of indices where the storage is full
            #FCR_D = False if bool(index_max) == True and i <= index_max[-1] else FCR_D # Disables FCR-D down as long as the planned storage is full, enables FCR-D down once the storage won't be full for the rest of the day
            index_min = [k for k,x in enumerate(H2_storage) if x <= 0]
            #Troubleshooting
            index_max_list[i] = index_max
            index_min_list[i] = index_min
            #
            if bool(index_max) == True and i <= index_max[-1]: #Makes sure that operation happens only if index_max[] is non-empty, error otherwise
                FCR_D = False
            else:
                FCR_D = FCR_D_orginal
            if bool(index_min) == True and i <= index_min[-1]:
                FCR_U = False
            else:
                FCR_U = FCR_U_orginal
            #Computations
            #H2_max[i] = h2st_size_vector[0] - H2_storage[i-1] + H2_demand[i] ##Alternativt uttryck från vätgasbalans
            #v7 start
            if version == "v7" or version == "V7":
                deltaH2_max[i] = (h2st_size_vector[0] - max(H2_storage[i:])) / (activation_duration/60) #Prevents division by zero
                H2_max[i] = deltaH2_max[i] + H2_bau[i]
                H2_min[i] = (H2_demand[i] - H2_storage[i-1]) * (activation_duration/60) #Alternativt uttryck från vätgasbalans
                #deltaH2_min[i] = 0 - min(H2_storage[i:]) * (activation_duration/60)
                #H2_min[i] = H2_bau[i] + deltaH2_min[i] 
            elif version == "v8" or version == "V8":
                deltaH2_max[i] = (h2st_size_vector[0] - max(H2_storage[i:])) / (activation_duration/60) #Prevents division by zero
                H2_max[i] = deltaH2_max[i] + H2_bau[i]
                #H2_min[i] = (H2_demand[i] - H2_storage[i-1]) * (activation_duration/60) #Alternativt uttryck från vätgasbalans
                deltaH2_min[i] = 0 - min(H2_storage[i:]) * (activation_duration/60)
                H2_min[i] = H2_bau[i] + deltaH2_min[i] 
            else:
                H2_max[i] = h2st_size_vector[0] - H2_storage[i-1] + H2_demand[i] ##Alternativt uttryck från vätgasbalans
                H2_min[i] = H2_demand[i] - H2_storage[i-1] #Alternativt uttryck från vätgasbalans
            H2_max_closest = take_closest(h2_prod, H2_max[i])
            H2_min_closest = take_closest(h2_prod, H2_min[i])
            H2_max_index = [l for l,x in enumerate(h2_prod) if H2_max_closest == x]
            H2_min_index = [m for m,x in enumerate(h2_prod) if H2_min_closest == x]          
            P_H2_max[i] = P_max * H2_max_index[0] / (len(h2_prod)-1)
            P_H2_min = P_max * H2_min_index[0] / (len(h2_prod)-1)
            #Check which is the highest limit
            if P_H2_min < P_pv[i] and P_pv[i] >= LIMIT:
                deltaP_min = P_pv[i]
                Branch_min[i] = 1
            elif P_H2_min < LIMIT and P_pv[i] < LIMIT:
                deltaP_min = LIMIT
                Branch_min[i] = 2
            else:
                deltaP_min = P_H2_min
                Branch_min[i] = 3
            #Calculations for FCR_D are only possible if increase in power is possible
            
            if P_H2_max[i] >= P_bau[i] and FCR_D == True:
                P_reserved_max[i] = P_H2_max[i] - P_bau[i]
                Income_FCR_D[i] = P_reserved_max[i] * FCR_D_price[i1+i]/1000
                P_activated_max[i] = P_reserved_max[i] * Flex_frac_FCR_D[i]
                P_flex_FCR_D = P_activated_max[i]
                Branch_D[i] = 1
                FCR_D_record[i] = 1
            # elif P_H2_max <= P_bau[i] and FCR_D == True:
            #     if P_H2_max + P_bau[i] > P_max:
            #         P_reserved_max[i] = P_max
            #     else:
            #         P_reserved_max[i] = P_H2_max + P_bau[i]
            #     Income_FCR_D[i] = P_reserved_max[i] * FCR_D_price[i1+i]/1000
            #     P_activated_max[i] = P_reserved_max[i] * Flex_frac_FCR_D[i]
            #     P_flex_FCR_D = P_activated_max[i]
            #     Branch_D[i] = 3
            else:
                P_flex_FCR_D = 0
                P_activated_max[i] = P_flex_FCR_D
                Branch_D[i] = 2
            #Calculations för FCR_U are only possible if decrease in power is possible
            if deltaP_min < P_bau[i] and FCR_U == True:
                P_reserved_min[i] = - (P_bau[i] - deltaP_min)
                Income_FCR_U[i] = abs(P_reserved_min[i]*FCR_U_price[i1+i]/1000)
                P_activated_min[i] = P_reserved_min[i] * Flex_frac_FCR_U[i]
                P_flex_FCR_U = P_activated_min[i]
                Branch_U[i] = 1
                FCR_U_record[i] = 1
            else:
                P_flex_FCR_U = 0 
                P_activated_min[i] = P_flex_FCR_U
                Branch_U[i] = 2
            P_flex[i] = P_bau[i] + P_activated_max[i] + P_activated_min[i]
            Income_flex[i] = Income_FCR_D[i] + Income_FCR_U[i]
            P_activated[i] = P_activated_max[i] + P_activated_min[i]
            #Produced hydrogen
            H2_FCR_D_index = (len(h2_prod)-1)*(P_flex_FCR_D + P_bau[i]) / P_max
            H2_FCR_U_index = (len(h2_prod)-1)*(P_flex_FCR_U + P_bau[i]) / P_max
            H2_flex_FCR_D[i] = h2_prod[round(float(H2_FCR_D_index))]
            H2_flex_FCR_U[i] = h2_prod[round(float(H2_FCR_U_index))]
            H2_bau[i] = h2_prod[round((len(h2_prod)-1)*P_bau[i] / P_max)]
            H2_produced[i] = (H2_flex_FCR_D[i] - H2_bau[i]) - (H2_bau[i] - H2_flex_FCR_U[i])
            # Vad händer om man tar H2_produced från P_activated?
            H2_storage = [x + H2_produced[i] if o >= i else x for o,x in enumerate(H2_storage)]
            
    results = pd.DataFrame({"P_flex":P_flex,
                            "Income_flex": Income_flex,
                            "P_activated": P_activated,
                            "H2_storage": H2_storage,
                            "P_reserved_max": P_reserved_max,
                            "P_reserved_min": P_reserved_min,
                            "Branch_min": Branch_min,
                            "Branch_D": Branch_D,
                            "Branch_U": Branch_U,
                            "FCR_D_record": FCR_D_record,
                            "FCR_U_record": FCR_U_record,
                            "H2_produced": H2_produced,
                            "Index_max_list": index_max_list,
                            "index_min_list": index_min_list,
                            "H2_flex_FCR_D": H2_flex_FCR_D,
                            "H2_flex_FCR_U": H2_flex_FCR_U,
                            "P_activated_max": P_activated_max,
                            "P_activated_min": P_activated_min,
                            "Income_FCR_D": Income_FCR_D,
                            "Income_FCR_U": Income_FCR_U,
                            "DeltaH2_max": deltaH2_max,
                            "H2_max": H2_max,
                            "DeltaH2_min": deltaH2_min,
                            "H2_min": H2_min,
                            "P_H2_max": P_H2_max,
                            "P_H2_min": P_H2_min
                            
                            })
   
    return results

   
    
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
        FCR_U_price: list, #List, the hourly price/MW for FCR-D up in bidding region SE3
        min_load: float    #Minimal electrolyzer load in order to avoid cold starts
) -> pd.DataFrame:
    
    
    P_flex = [0]*24 # Creates list of same size (no of hours in a day)
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
    GRÄNS = min_load*P_max
    Branch = [0]*24
    #i1=0, H2_demand=[50]*24, H2_storage=[150]*24, h2st_size_vector=[300], P_bau=[5]*24, P_pv=[0.5]*24, P_max=10, h2_prod=pem2.h2_prod, FCR_D=True, FCR_U=False, FCR_N=False, Flex_frac_FCR_D = [0.1]*24, Flex_frac_FCR_U=[0]*24, FCR_D_power=FCR_D_power, FCR_D_price=FCR_D_price, FCR_U_power=FCR_U_power, FCR_U_prize=FCR_U_price)
    for i in range(int(24)): #endast för en dag, så 24
        if i == 0:
            P_flex[i] = P_bau[0]
        else:    
            FCR_D = FCR_D_orginal #Resets boolean value to the imput value
            FCR_U = FCR_U_orginal
            index_max = [j for j,x in enumerate(H2_storage) if x >= h2st_size_vector[0]] #Returns list of indices where the storage is full
            #FCR_D = False if bool(index_max) == True and i <= index_max[-1] else FCR_D # Disables FCR-D down as long as the planned storage is full, enables FCR-D down once the storage won't be full for the rest of the day
            index_min = [k for k,x in enumerate(H2_storage) if x <= 0]
            if bool(index_max) == True and i <= index_max[-1]: #Makes sure that operation happens only if index_max[] is non-empty, error otherwise
                FCR_D = False
            else:
                FCR_D = FCR_D_orginal
            if bool(index_min) == True and i <= index_min[-1]:
                FCR_U = False
            else:
                FCR_U = FCR_U_orginal
            if H2_storage[i] >= H2_demand[i]: #Branch 1
                # FCR_D = False if H2_storage[i] == h2st_size_vector[0] else FCR_D == FCR_D_orginal #Borde fungera som en rad, No increased production if the storage is full
                if FCR_D == True and FCR_U == False: #Behöver det finnas en övre gräns för P_ele (finns realistiskt iaf)?
                    H2_max[i] = h2st_size_vector[0] - H2_storage[i-1] + H2_demand[i] ##Alternativt uttryck från vätgasbalans
                    H2_closest = take_closest(h2_prod, H2_max[i])
                    H2_index = [k for k,x in enumerate(h2_prod) if H2_closest == h2_prod[k]] #Finds the index where the produced hydrogen is the same, requires uniqe values(?)
                    P_H2_max = P_max * (H2_index[0]+1) / len(h2_prod) 
                    # P_H2_max = ([k for k,x in enumerate(h2_prod) if H2_max[i] == h2_prod[k]]+1)/len(h2_prod)*P_max #Alternativ omskrivning
                    #Can only increase prod if P_H2_max > P_bau
                    if P_H2_max > P_bau[i]:
                        deltaP_max[i] = P_H2_max - P_bau[i]
                        P_reserved[i] = deltaP_max[i] # Maximum possible power reserved as frequency regulation
                        Income_flex[i] = P_reserved[i] * FCR_D_price[i1+i]/1000
                        P_activated[i] = deltaP_max[i] * Flex_frac_FCR_D[i] 
                        P_flex[i] = P_bau[i] + P_activated[i]
                        if P_bau[i] <= P_max:
                            H2_flex[i] = h2_prod[round((len(h2_prod)-1) * P_flex[i] / P_max)]   #Hydrogen production at given average partial load
                            H2_bau[i] = h2_prod[round((len(h2_prod)-1) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                            Branch[i] = 1
                        else: #Above max power
                            H2_flex[i] = h2_prod[-1]
                            H2_bau[i] = h2_prod[-1] 
                            Branch[i] = 1.5
                        H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                        H2_storage = [x + H2_produced[i] if j >= i else x for j,x in enumerate(H2_storage)] # Increases storage for all indices after the last time the storage is full 
                        
                    else:
                        P_flex[i] = P_bau[i]
        
                elif FCR_U == True and FCR_D == False:         #Gällande ovan rad, P_bau ska kanske vara tidigare stegs effekt P_ele [i-1]?
                    if P_pv[i] >= GRÄNS: # Gränsvärde för att undvika kallstarter om P_pv = 0.
                        P_reserved[i] = - (P_bau[i] - P_pv[i]) #Reglerar ned så mycket som möjligt, med P_pv som undre gräns. Negativ för att indikera minskning
                        Branch[i] = 2.1
                    else: #P_pv[i] < GRÄNS
                        P_reserved[i] = - (P_bau[i] - GRÄNS) #Negative as it regulates down
                        Branch[i] = 2.2
                    if P_reserved[i] > 0: #No FCR_U as P increases
                        P_flex[i] = P_bau[i]
                    else:
                        Income_flex[i] = abs(P_reserved[i] * FCR_U_price[i1+i]/1000) #Income is positive
                        P_activated[i] = P_reserved[i] * Flex_frac_FCR_U[i] # Negative as it's a reduction in used energy
                        P_flex[i] = P_bau[i] + P_activated[i] #Average electrolyzer power usage for the hour
                        if P_bau[i] <= P_max:
                            H2_flex[i] = h2_prod[round((len(h2_prod)-1) * P_flex[i] / P_max)] #Hydrogen production at given average partial load
                            H2_bau[i] = h2_prod[round((len(h2_prod)-1) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                            #Branch[i] = 2
                        else:
                            H2_flex[i] = h2_prod[-1]
                            H2_bau[i] = h2_prod[-1]
                            #Branch[i] = 2.5
                        H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                        H2_storage = [x + H2_produced[i] if j>= i else x for j,x in enumerate(H2_storage)] #Adds to all future storage, not retroactively
                    
                else:
                      P_flex[i] = P_bau[i] #Om inte på stödtjänstmarknaden gäller business as usual
                  
            else: #Branch 2, implicit H2_storage[i] < H2_demand[i]
                if FCR_D == True and FCR_U == False:
                    #Styrningen kommer nog se till att metaniseringsbehovet alltid täcks, så P_bau räcker i basfallet
                    # Ska denna vara nästan samma som FCR_D i första grenen?
                    H2_max[i] = h2st_size_vector[0] + H2_demand[i] - H2_storage[i-1] # Maximum possible hourly increase in hydrogen production [kg]
                    # Behövs if-satsen nedan? Tänker att take_closest bör lösa det ändå. 
                    if H2_max[i] <= max(h2_prod): #Checks if the theoretical maximum hydrogen production is possible in this system
                        H2_closest = take_closest(h2_prod, H2_max[i])
                        H2_index = [k for k,x in enumerate(h2_prod) if H2_closest == h2_prod[k]] #Finds the index where the produced hydrogen is the same, requires uniqe values(?)
                        P_H2_max = H2_index[0]+1 / len(h2_prod) * P_max
                    else:
                        P_H2_max = P_max #Otherwise upper technical limit
                    if P_H2_max > P_bau[i]:
                        deltaP_max[i] = P_H2_max - P_bau[i]
                        P_reserved[i] = deltaP_max[i] # Maximum possible power reserved as frequency regulation
                        Income_flex[i] = P_reserved[i] * FCR_D_price[i1+i]/1000
                        P_activated[i] = deltaP_max[i] * Flex_frac_FCR_D[i] 
                        P_flex[i] = P_bau[i] + P_activated[i] #Average electrolyzer power usage for the hour
                        if P_bau[i] <= P_max:
                            H2_flex[i] = h2_prod[round((len(h2_prod)-1) * P_flex[i] / P_max)]   #Hydrogen production at given average partial load
                            H2_bau[i] = h2_prod[round((len(h2_prod)-1) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                        else: #Above max power
                            H2_flex[i] = h2_prod[-1]
                            H2_bau[i] = h2_prod[-1] 
                        H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                        H2_storage = [x + H2_produced[i] if j >= i else x for j,x in enumerate(H2_storage)] # Increases storage for all indices after the last time the storage is full 
                        Branch[i] = 3
                    else: #If not able to increase P
                        P_flex[i] = P_bau[i]
                        Branch[i] = 3.5
                elif FCR_U == True and FCR_D == False:
                    H2_max[i] = H2_demand[i] - H2_storage[i] #Ska kanske heta H2_min?
                    H2_closest = take_closest(h2_prod, float(H2_max[i]))
                    H2_index = [k for k,x in enumerate(h2_prod) if H2_closest == h2_prod[k]] #Finds the index where the produced hydrogen is the same, requires uniqe values(?)
                    P_H2_max = P_max * (H2_index[0]+1) / len(h2_prod) 
                    if P_H2_max < P_pv[i] and P_pv[i] >= GRÄNS:
                        deltaP_max[i] = P_pv[i]
                        Branch[i] = 4.1
                    elif P_H2_max < GRÄNS and P_pv[i] < GRÄNS:
                        deltaP_max[i] = GRÄNS
                        Branch[i] = 4.2
                    else:
                        deltaP_max[i] = P_H2_max  # The minimal electrolyzer power that leaves h2_storage non-negative (at zero)
                        Branch[i] = 4.3
                    P_reserved[i] = -(P_bau[i] - deltaP_max[i])
                    if P_reserved[i] > 0: #No FCR_U in this case
                        P_flex[i] = P_bau[i]
                    else:
                        Income_flex[i] = abs(P_reserved[i] * FCR_U_price[i1+i]/1000)
                        P_activated[i] = P_reserved[i] * Flex_frac_FCR_U[i]
                        P_flex[i] = P_bau[i] + P_activated[i] #Average electrolyzer power usage for the hour
                        if P_flex[i] <= P_max:
                            H2_flex[i] = h2_prod[round((len(h2_prod)-1) * P_flex[i] / P_max)] #Hydrogen production at given average partial load
                            H2_bau[i] = h2_prod[round((len(h2_prod)-1) * P_bau[i] / P_max)] #Hydrogen production at planned average electrolyzer power
                            #Branch[i] = 4
                        else:
                            H2_flex[i] = h2_prod[-1] #Hydrogen production at given average partial load
                            H2_bau[i] = h2_prod[-1] #Hydrogen production at planned average electrolyzer power
                            #Branch[i] = 4.5
                        H2_produced[i] = H2_flex[i] - H2_bau[i] #Actual additional hydrogen production from flexibility behaviour, this method takes correct measures to partial load efficiencies 
                        H2_storage = [x + H2_produced[i] if j>= i else x for j,x in enumerate(H2_storage)] #Adds to all future storage, not retroactively
                    
                else:
                    P_flex[i] = P_bau[i]
    
    out = pd.DataFrame(
        {"P_flex": P_flex,
         "Income_flex": Income_flex,
         "P_activated": P_activated,
         "H2_storage": H2_storage,
         "P_reserved": P_reserved,  #Last three are for bugfixing
         "H2_produced": H2_produced,
         "Branch" : Branch}
        )
    return out    #Används senare till beräkningar


   