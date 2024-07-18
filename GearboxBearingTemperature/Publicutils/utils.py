# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:20:23 2022

@author: xz.fan
"""
import pandas as pd 




def data_process(data,date,gp,gearbox_bearing_temperature1,gearbox_bearing_temperature2,temperature_oil,rs,temperature_cab):
    data = data.loc[:,[date,gp,gearbox_bearing_temperature1,gearbox_bearing_temperature2,temperature_oil,rs,temperature_cab]]
    time = data[date]
#    data = data.loc[ (data[gp] > 0)&(data[gearbox_bearing_temperature2]<70), :]
    data[date] = pd.to_datetime(time)
    data = data.reset_index()
    data = data.sort_values(by = date) 
    data['X1_dr'] = data[gearbox_bearing_temperature1].shift(1)
    data['X1_ndr'] = data[gearbox_bearing_temperature2].shift(1)
    data['x2'] = data[temperature_oil].shift(1)
    data['x3'] = data[temperature_cab].shift(1)
    data['x5'] = data[rs].shift(1)
    data['x6'] = data[gp].shift(1) 
    data['Y_dr'] = data[gearbox_bearing_temperature1]
    data['Y_ndr'] = data[gearbox_bearing_temperature2]
    data = data.reset_index()
    data = data.dropna()
    return data