...#! /usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:47:54 2019

@author: Minxuan
"""

import pandas as pd
import numpy as np
import datetime
import math
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob

# from GearboxBearingTemperature.Publicutils.utils import data_process
 

ts = 'time'
gs = 'generator_rotating_speed'
ws='wind_speed'
gp = 'power'
temperature_cab = 'cabin_temp'
temperature_oil = 'gearbox_oil_temp'

gearbox_bearing_temperature1 = 'f_gearbox_bearing_temp'
gearbox_bearing_temperature2 = 'r_gearbox_bearing_temp'
threshold_rs = 92         #并网转速
turbineName = 'turbineName'




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

def gearbox_bearing_fit_main(data):

    # data pre-processing
#    data_processed = data_pre_processing(data)
    #train model
    data = data.loc[:,[ts, gp, gearbox_bearing_temperature1, gearbox_bearing_temperature2, temperature_oil, gs, temperature_cab]]
    data = data.loc[(data[gs] > threshold_rs) & (data[gp] > 0), :]
    data[ts] = pd.to_datetime(data[ts])
    data_processed = data.dropna(axis=0) 
    if len(data[ts]) != len(data_processed[ts]):
        print('数据中存在缺失值')
    # data_train = data_processed[data_processed[date] < pd.to_datetime('2021-07-01 0:00')]

    data_train=data_process(data, ts, gp, gearbox_bearing_temperature1, gearbox_bearing_temperature2, temperature_oil, gs, temperature_cab)
    X1 = data_train.loc[:,['X1_dr','x2','x3','x5','x6']]
    Y1 = data_train.loc[:,['Y_dr']]
    modeldr = LinearRegression().fit(X1,Y1)#驱动端模型训练
    X2 = data_train.loc[:,['X1_ndr','x2','x3','x5','x6']]
    Y2 = data_train.loc[:,['Y_ndr']]
    modelndr = LinearRegression().fit(X2,Y2)#非驱动端模型训练
    return modeldr,modelndr
    
    
if __name__ == '__main__':

    data_path = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\华润风机数据\\数据\\offline_analysis\\10min\\3.2MW\\'

    fileNames = []
    wind_turbine_IDs = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            wind_turbine_IDs.append(name.split('.')[0])
            fileNames.append(os.path.join(root, name))

    for i in range(len(wind_turbine_IDs)):
        wt_id = wind_turbine_IDs[i]
        path_10min = data_path + wt_id + '.csv'
        all_data_10min = pd.read_csv(path_10min, encoding='gbk')

        all_data_10min[ts] = pd.to_datetime(all_data_10min[ts])  ##插值的时候不能shi datatimeStamp
        all_data_10min = all_data_10min.drop_duplicates([ts])
        vars_10min = all_data_10min.columns.to_list()

        train = all_data_10min[:int(0.4*len(all_data_10min))]

        modeldr, modelndr = gearbox_bearing_fit_main(train)  ##选40%的数据作为训练数据


        outpath = r'../Resource/gearbox_bearing_temp'  # 驱动端模型输出路径
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        model_file = outpath + os.sep + wt_id +'_modeldr.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(modeldr, f)
        f.close()

        model_file = outpath + os.sep + wt_id +'_modelndr.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(modelndr, f)
        f.close()