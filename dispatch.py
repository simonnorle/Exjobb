# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:47:57 2023

@author: Linus Engstam
"""

import pandas as pd
import numpy as np
import pulp as plp


"""
Currently, p2g_wwtp3 is used (starting on line 3687).
Gurobi Optimizer is recommended for faster performance, but built-in PuLP optimizer can be enabled within the dispatch strategies.
"""


def p2g_wwtp3(
    h2_dem, #[kg/h]
    heat_demand, #[kWh/h] hourly
    o2_demand, #[mol/h] hourly
    o2_power, #[kWh avoided/kg O2]
    biogas, #[mol/h] CH4 and CO2 flow
    k_values,
    m_values,
    grid,
    wind,
    pv,
    elz_max,  # kW
    elz_min,  # kW
    aux_cons,  # kW
    h2st_max,  # kg?
    h2st_prev,  # storage from previous day
    meth_max,  # maximum H2 use [kg/h]
    meth_min,  # minimum H2 use [kg/h]
    meth_el_factor: float = 0.0, #[kWh/molCO2 converted]
    comp_el_factor: float = 0.0, #[kWh/mol compressed gas]
    meth_el: float = 0.0, #electricity demand for methanation and biogas compression [kWh/molCO2]
    heat_value: list = [0.0,0.0,0.0,0.0], #[€/MWh] for spring, summer, autumn and winter respectively
    startup_cost: float = 0.1,  # fraction of rated power
    standby_cost: float = 0.02,  # fraction of rated power
    elz_startup_time: float = 0.0, # [hour] time until H2 production starts after a cold startup
    elz_heat_time: float = 0.0, # [hour] time until electrolyzer has reached operating temperature
    prev_mode: int = 1,  # on/standby/off in last hour of previous day
    elz_eff: float = 0.7,  # HHV
    wind_cost: float = -9,  # [€/MWh]
    pv_cost: float = -10,  # [€/MWh] (only to prioritize ahead of wind and negative prices)
    bat_cap: float = 0, # [kWh]
    bat_eff: float = 0.95, # battery round trip efficiency
    bat_prev: float = 0, # [kWh] previous day battery charge
    meth_spec_heat: float = 0.0, # [kWh/kgH2] methanation heat generation per kg H2 methanized
    usable_heat: float = 0.8, #usable heat fraction
    h2o_cons: float = 10.0, #water consumption [lH2O/kgH2]
    temp: float = 80.0,
    h2o_temp: float = 15.0,
) -> pd.DataFrame:
    """Returns electrolyzer dispatch based on both renewable and grid electricity
    optimized to minimize the cost of electricity for a fixed H2 demand.
    Also including a linearized part-load efficiency and on/standby/off modes for
    the electrolyzer as well as startup losses for electrolysis and methanation.
    Using linearization parameters as proposed in Wirtz et al. 2021"""

    #Hourly hydrogen demand (from CO2 availability)    
    h2_demand = h2_dem.flatten().tolist()
    
    wind = list(wind)
    pv = list(pv)
    grid = list(grid)
    h2st_prev = h2st_prev * 2.02 / 1000  # mol to kg

    # Define variables
    elz = []
    grid_el = []
    wind_el = []
    pv_el = []
    h2st = []
    h2_prod = []
    h2_prod_start = []
    h2_prod_real = []
    elz_mode = []
    elz_start = []
    elz_stop = []
    elz_off = []
    elz_standby = []
    h2_use = []
    meth_on = []
    o2_prod = []
    o2_use = []
    elz_heat = []
    heat_use = []
    heat_income = []
    elz_heat_demand = []
    ehd = []
    bat = []
    bat_out = []
    standby_el = []
    meth_el1 = []
    comp_el = []
    # Part-load variables
    # Load
    e10 = []
    e20 = []
    e30 = []
    e40 = []
    e50 = []
    e60 = []
    e70 = []
    e80 = []
    e90 = []
    e100 = []
    # On/off
    lambda10 = []
    lambda20 = []
    lambda30 = []
    lambda40 = []
    lambda50 = []
    lambda60 = []
    lambda70 = []
    lambda80 = []
    lambda90 = []
    lambda100 = []

    # Linearization parameters
    # Gradient
    k10 = k_values[0]
    k20 = k_values[1]
    k30 = k_values[2]
    k40 = k_values[3]
    k50 = k_values[4]
    k60 = k_values[5]
    k70 = k_values[6]
    k80 = k_values[7]
    k90 = k_values[8]
    k100 = k_values[9]
    # M-value
    m10 = m_values[0]
    m20 = m_values[1]
    m30 = m_values[2]
    m40 = m_values[3]
    m50 = m_values[4]
    m60 = m_values[5]
    m70 = m_values[6]
    m80 = m_values[7]
    m90 = m_values[8]
    m100 = m_values[9]

    for i in range(len(grid)):
        # Electrolyzer operation
        elz.append(plp.LpVariable("elz_{}".format(i), 0, elz_max))
        elz_mode.append(plp.LpVariable("elz_mode_{}".format(i), 0, 1, plp.LpInteger))
        elz_start.append(plp.LpVariable("elz_start_{}".format(i), 0, 1, plp.LpInteger))
        elz_stop.append(plp.LpVariable("elz_stop_{}".format(i), 0, 1, plp.LpInteger))
        elz_standby.append(plp.LpVariable("elz_standby_{}".format(i), 0, 1, plp.LpInteger))
        elz_off.append(plp.LpVariable("elz_off_{}".format(i), 0, 1, plp.LpInteger))
        # Purchased grid electricity
        grid_el.append(plp.LpVariable("grid_el_{}".format(i), 0, None))
        # Wind-based operation
        wind_el.append(plp.LpVariable("wind_el_{}".format(i), 0, None))
        # PV-based operation
        pv_el.append(plp.LpVariable("pv_el_{}".format(i), 0, None))
        # Hydrogen storage
        h2st.append(plp.LpVariable("h2st_{}".format(i), 0, h2st_max))
        # Hydrogen production and utilization
        h2_prod.append(plp.LpVariable("h2_prod_{}".format(i), 0, None))
        h2_prod_start.append(plp.LpVariable("h2_prod_start_{}".format(i), 0, None))
        h2_prod_real.append(plp.LpVariable("h2_prod_real_{}".format(i), 0, None))
        h2_use.append(plp.LpVariable("h2_use_{}".format(i), 0, meth_max))
        # Methanation
        meth_on.append(plp.LpVariable("meth_on_{}".format(i), 0, 1, plp.LpInteger))
        meth_el1.append(plp.LpVariable("meth_el_{}".format(i), 0, None))
        comp_el.append(plp.LpVariable("comp_el_{}".format(i), 0, None))
        # Oxygen production and utilization
        o2_prod.append(plp.LpVariable("o2_prod_{}".format(i), 0, None))
        o2_use.append(plp.LpVariable("o2_use_{}".format(i), 0, None))
        # Heat production and utilization
        elz_heat.append(plp.LpVariable("elz_heat_{}".format(i), 0, None)) # Heat from electrolyzer
        heat_use.append(plp.LpVariable("heat_use_{}".format(i), 0, None)) # Utilized heat
        heat_income.append(plp.LpVariable("heat_income_{}".format(i), 0, None)) # Income from heat use
        elz_heat_demand.append(plp.LpVariable("elz_heat_demand_{}".format(i), 0, None)) # 
        ehd.append(plp.LpVariable("ehd_{}".format(i), 0, 1, plp.LpInteger))
        #Battery
        bat.append(plp.LpVariable("bat_{}".format(i), 0, bat_cap))
        bat_out.append(plp.LpVariable("bat_out_{}".format(i), 0, bat_cap))
        #Standby electricity consumption
        standby_el.append(plp.LpVariable("sys_el_{}".format(i), 0, None))
        
        # Part-load variables
        # Load
        e10.append(plp.LpVariable("e10_{}".format(i), 0, None)) # Load between 0 and 10 %
        e20.append(plp.LpVariable("e20_{}".format(i), 0, None)) # Load between 10 and 20 %
        e30.append(plp.LpVariable("e30_{}".format(i), 0, None)) # Load between 20 and 30 %
        e40.append(plp.LpVariable("e40_{}".format(i), 0, None)) # Load between 30 and 40 %
        e50.append(plp.LpVariable("e50_{}".format(i), 0, None)) # Load between 40 and 50 %
        e60.append(plp.LpVariable("e60_{}".format(i), 0, None)) # Load between 50 and 60 %
        e70.append(plp.LpVariable("e70_{}".format(i), 0, None)) # Load between 60 and 70 %
        e80.append(plp.LpVariable("e80_{}".format(i), 0, None)) # Load between 70 and 80 %
        e90.append(plp.LpVariable("e90_{}".format(i), 0, None)) # Load between 80 and 90 %
        e100.append(plp.LpVariable("e100_{}".format(i), 0, None)) # Load between 90 and 100 %
        # On/off
        lambda10.append(plp.LpVariable(
            "lambda10_{}".format(i), 0, 1, plp.LpInteger))
        lambda20.append(plp.LpVariable(
            "lambda20_{}".format(i), 0, 1, plp.LpInteger))
        lambda30.append(plp.LpVariable(
            "lambda30_{}".format(i), 0, 1, plp.LpInteger))
        lambda40.append(plp.LpVariable(
            "lambda40_{}".format(i), 0, 1, plp.LpInteger))
        lambda50.append(plp.LpVariable(
            "lambda50_{}".format(i), 0, 1, plp.LpInteger))
        lambda60.append(plp.LpVariable(
            "lambda60_{}".format(i), 0, 1, plp.LpInteger))
        lambda70.append(plp.LpVariable(
            "lambda70_{}".format(i), 0, 1, plp.LpInteger))
        lambda80.append(plp.LpVariable(
            "lambda80_{}".format(i), 0, 1, plp.LpInteger))
        lambda90.append(plp.LpVariable(
            "lambda90_{}".format(i), 0, 1, plp.LpInteger))
        lambda100.append(plp.LpVariable(
            "lambda100_{}".format(i), 0, 1, plp.LpInteger))

    # Problem definition
    prob = plp.LpProblem("Operational_cost_minimization", plp.LpMinimize)

    # Constraints
    for i in range(len(grid)):
        # Defining electricity supply
        prob += wind_el[i] <= wind[i] #wind
        prob += pv_el[i] <= pv[i] #PV
        # prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] #electrolyzer
        prob += standby_el[i] == elz_standby[i] * elz_max * standby_cost
        prob += elz[i] == wind_el[i] + pv_el[i] + grid_el[i] + bat_out[i] - standby_el[i] - meth_el1[i] - comp_el[i] #overall system electricity consumption
        prob += meth_el1[i] == h2_use[i] * meth_el_factor #electricity required for methanation
        if biogas[i,1] > 0:
            prob += comp_el[i] == (((0.25*h2_use[i]*1000/2.02) / biogas[i,1]) * (biogas[i,0] + biogas[i,1])) * comp_el_factor #electricity required for biogas compression
        elif biogas[i,1] == 0:
            prob += comp_el[i] == 0
        # Defining minimum electrolyzer load
        prob += elz[i] >= elz_min * elz_mode[i]
        # Defining maximum electrolyzer load
        prob += elz[i] <= elz_max * elz_mode[i]
        # Defining minimum methanation load
        prob += h2_use[i] >= meth_min * meth_on[i]
        # Defining maximum methanation load
        prob += h2_use[i] <= meth_max * meth_on[i]

        # Part-load variable constraints
        prob += e10[i] <= elz_max * 0.1 * lambda10[i]
        prob += e10[i] >= elz_max * 0 * lambda10[i]
        prob += e20[i] <= elz_max * 0.2 * lambda20[i]
        prob += e20[i] >= elz_max * 0.1 * lambda20[i]
        prob += e30[i] <= elz_max * 0.3 * lambda30[i]
        prob += e30[i] >= elz_max * 0.2 * lambda30[i]
        prob += e40[i] <= elz_max * 0.4 * lambda40[i]
        prob += e40[i] >= elz_max * 0.3 * lambda40[i]
        prob += e50[i] <= elz_max * 0.5 * lambda50[i]
        prob += e50[i] >= elz_max * 0.4 * lambda50[i]
        prob += e60[i] <= elz_max * 0.6 * lambda60[i]
        prob += e60[i] >= elz_max * 0.5 * lambda60[i]
        prob += e70[i] <= elz_max * 0.7 * lambda70[i]
        prob += e70[i] >= elz_max * 0.6 * lambda70[i]
        prob += e80[i] <= elz_max * 0.8 * lambda80[i]
        prob += e80[i] >= elz_max * 0.7 * lambda80[i]
        prob += e90[i] <= elz_max * 0.9 * lambda90[i]
        prob += e90[i] >= elz_max * 0.8 * lambda90[i]
        prob += e100[i] <= elz_max * 1 * lambda100[i]
        prob += e100[i] >= elz_max * 0.9 * lambda100[i]
        # Total electrolyzer load
        prob += elz[i] == e10[i] + e20[i] + e30[i] + e40[i] + \
            e50[i] + e60[i] + e70[i] + e80[i] + e90[i] + e100[i]
        # Only one part of linearization simultaneously
        prob += lambda10[i] == elz_mode[i] - lambda20[i] - lambda30[i] - lambda40[i] - \
            lambda50[i] - lambda60[i] - lambda70[i] - \
            lambda80[i] - lambda90[i] - lambda100[i]

        #Linearized H2 production efficiency
        # prob += h2_prod_real[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
        #                         (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
        #                         (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        prob += h2_prod[i] == ((e10[i]*k10/elz_max) + (lambda10[i]*m10) + (e20[i]*k20/elz_max) + (lambda20[i]*m20) + (e30[i]*k30/elz_max) + (lambda30[i]*m30) +
                                (e40[i]*k40/elz_max) + (lambda40[i]*m40) + (e50[i]*k50/elz_max) + (lambda50[i]*m50) + (e60[i]*k60/elz_max) + (lambda60[i]*m60) +
                                (e70[i]*k70/elz_max) + (lambda70[i]*m70) + (e80[i]*k80/elz_max) + (lambda80[i]*m80) + (e90[i]*k90/elz_max) + (lambda90[i]*m90) + (e100[i]*k100/elz_max) + (lambda100[i]*m100))
        
        # To add a start-up H2 losses, uncomment this and use other option above
        # prob += h2_prod_start[i] == h2_prod_real[i] * (1-elz_startup_time)
        # prob += h2_prod[i] >= h2_prod_start[i] - (100000000*(1-elz_start[i]))
        # prob += h2_prod[i] <= h2_prod_start[i] + (100000000*(1-elz_start[i]))
        # prob += h2_prod[i] >= h2_prod_real[i] - (100000000*elz_start[i])
        # prob += h2_prod[i] <= h2_prod_real[i] + (100000000*elz_start[i])
        
        # Defining start-up
        if i > 0:
            # Electrolyzer
            prob += elz_start[i] >= elz_mode[i] - elz_mode[i-1] - elz_standby[i-1]# + elz_stop[i]
            # Can't start if on or standby during previous hour or on or standby in current hour (not needed)
            # prob += elz_start[i] <= 1 - elz_mode[i-1] - elz_standby[i-1]
            # prob += elz_start[i] <= 1 - elz_off[i] - elz_standby[i]
            # Can't start without being on (not needed)
            # prob += elz_start[i] <= elz_mode[i]
            # Can't go to standby from off or vice versa
            prob += elz_standby[i] <= 1 - elz_off[i-1]
            # prob += elz_off[i] <= 1 - elz_standby[i-1]
        else:  # Considering end of previous day
            if prev_mode == 1:
                prob += elz_start[i] == 0
            elif prev_mode == 0:
                prob += elz_start[i] == elz_mode[i]
                prob += elz_standby[i] == 0

        # Electrolyzer modes (Baumhof)
        prob += elz_mode[i] + elz_standby[i] + elz_off[i] == 1
        # prob += elz_start[i] + elz_stop[i] <= 1
        
        # Can't exceed hourly CO2 availability
        prob += h2_use[i] <= h2_demand[i]

        # Storage charging/discharging
        if i == 0:  # Using previous day storage value for first hour
            # Hydrogen    
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st_prev
            # Battery (currently not using grid electricity)
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat_prev
        else:
            # Hydrogen
            prob += h2st[i] == h2_prod[i] - h2_use[i] + h2st[i-1]
            # Battery
            prob += bat[i] <= ((wind[i] + pv[i] - wind_el[i] - pv_el[i]) * bat_eff) - bat_out[i] + bat[i-1]
        
        # By-product generation
        # Oxygen
        prob += o2_prod[i] == h2_prod[i] * (32/2.02) * 0.5  # kg of o2 produced
        prob += o2_use[i] <= o2_prod[i]
        prob += o2_use[i] <= o2_demand[i]
        
        # Heat
        # prob += elz_heat[i] == elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)
        # Electrolyzer heat demand, remove methanation heat (already known from h2_use) (maximum of this heat and zero)
        prob += 100000 * ehd[i] >= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat)
        prob += 100000 * (1 - ehd[i]) >= (h2_use[i] * meth_spec_heat * usable_heat) - heat_demand[i]
        prob += elz_heat_demand[i] <= heat_demand[i] - (h2_use[i] * meth_spec_heat * usable_heat) + (100000 * (1 - ehd[i])) 
        prob += elz_heat_demand[i] <= 0 + 100000 * ehd[i]
        # No heat during cold start
        prob += elz_heat[i] <= (1-elz_start[i]) * 1000000000
        prob += elz_heat[i] >= 0
        prob += elz_heat[i] <= elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp)) # Input water heat consumption included
        prob += elz_heat[i] >= (elz[i] - (aux_cons*elz_mode[i]) - (h2_prod[i]*39.4)) - (elz_start[i]*1000000000) - ((h2o_cons * h2_prod[i] * 997 / 1000) * (4.18/3600) * (temp - h2o_temp))  
        # Use limited by both production and demand
        prob += heat_use[i] <= usable_heat * elz_heat[i]
        prob += heat_use[i] <= elz_heat_demand[i]
        prob += heat_use[i] >= 0

        # Seasonal district heat cost
        if (i < 1416) or (i >= 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[3] / 1000
        elif (i >= 1416) and (i < 3624):
            prob += heat_income[i] == heat_use[i] * heat_value[0] / 1000
        elif (i >= 3624) and (i < 5832):
            prob += heat_income[i] == heat_use[i] * heat_value[1] / 1000
        elif (i >= 5832) and (i < 8016):
            prob += heat_income[i] == heat_use[i] * heat_value[2] / 1000
    
    
    # Objective (minimize the electricity cost)
    prob += plp.lpSum([grid_el[i] * grid[i] / 1000 for i in range(len(grid))]) + plp.lpSum([wind_el[i] * wind_cost / 1000 for i in range(len(grid))]) + plp.lpSum([pv_el[i] * pv_cost / 1000 for i in range(len(grid))]) + \
        ((plp.lpSum([h2_demand[i] - h2_use[i] for i in range(len(grid))]))*10000000) + (plp.lpSum([elz_start[i] * elz_startup_time*elz_max*grid[i]/1000 for i in range(len(grid))])) - \
            plp.lpSum([o2_use[i] * (o2_power * grid[i] / 1000) for i in range(len(grid))]) - plp.lpSum([heat_income[i] for i in range(len(grid))])# + \
    # electricity costs (grid, wind, pv)
    # oxygen income, and heat income
    
    # Define solver (Gurobi/PuLP)
    #solver = plp.GUROBI_CMD()
    solver = plp.PULP_CBC_CMD() # Uncomment to use built-in PuLP solver
    status = prob.solve(solver)

    e_op = []
    grid_op = []
    wind_op = []
    pv_op = []
    storage = []
    h2_produced = []
    h2_used = []
    e_on = []
    e_standby = []
    e_start = []
    e_off = []
    o_prod = []
    o_use = []
    h_prod = []
    h_use = []
    h_inc = []
    bat_state = []
    bat_discharge = []
    sys_op = []
    sb_el = []
    m_el = []
    c_el = []
    e_h_dem = []
    h2_produced_start = []
    h2_produced_real = []

    # Saving variable solutions
    for i in range(len(grid)):
        e_op.append(elz[i].varValue)
        grid_op.append(grid_el[i].varValue)
        wind_op.append(wind_el[i].varValue)
        pv_op.append(pv_el[i].varValue)
        storage.append(h2st[i].varValue)
        h2_produced.append(h2_prod[i].varValue)
        h2_used.append(h2_use[i].varValue)
        e_standby.append(elz_standby[i].varValue)
        e_start.append(elz_start[i].varValue)
        e_off.append(elz_off[i].varValue)
        e_on.append(elz_mode[i].varValue)
        o_prod.append(o2_prod[i].varValue)
        o_use.append(o2_use[i].varValue)
        h_prod.append(elz_heat[i].varValue)
        h_use.append(heat_use[i].varValue)
        h_inc.append(heat_income[i].varValue)
        bat_state.append(bat[i].varValue)
        bat_discharge.append(bat_out[i].varValue)
        sys_op.append(elz[i].varValue + standby_el[i].varValue + meth_el1[i].varValue + comp_el[i].varValue)
        sb_el.append(standby_el[i].varValue)
        m_el.append(meth_el1[i].varValue)
        c_el.append(comp_el[i].varValue)
        e_h_dem.append(elz_heat_demand[i].varValue)
        h2_produced_start.append(h2_prod_start[i].varValue)
        h2_produced_real.append(h2_prod_real[i].varValue)   

    h2_missed = list(np.array(h2_demand) - np.array(h2_used)) # Unmet H2 demand
    demand_vector = h2_demand
    grid_inc = np.array(grid_op) * grid / 1000
    o2_inc = np.array(o_use) * (o2_power * grid[i] / 1000)

    op = pd.DataFrame(
        {'Electrolyzer operation': e_op,
         'Grid purchases': grid_op,
         'Wind power': wind_op,
         'PV power': pv_op,
         'Electricity price': grid,
         'Storage': storage,
         'Demand': demand_vector,
         'H2 prod': h2_produced,
         'H2 used': h2_used,
         # 'Efficiency': eff,
         'Unmet demand': h2_missed,
         'Status': status,
         'On': e_on,
         'Standby': e_standby,
         'Off': e_off,
         'Cold start': e_start,
          'O2 prod': o_prod,
          'O2 use': o_use,
          'Heat prod': h_prod,
          'Heat use': h_use,
          'Battery state': bat_state,
          'Battery discharging': bat_discharge,
          'Heat income': h_inc,
          'Grid income': grid_inc,
          'Oxygen income': o2_inc,
           'System operation': sys_op,
           'Standby electricity': sb_el,
           'Methanation electricity': m_el,
           'Compression electricity': c_el,
          'Heat demand': e_h_dem,
          'H2 prod start': h2_produced_start,
          'H2 prod real': h2_produced_real,
         })
    return op