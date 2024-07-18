#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:46:39 2019

@author: Minxuan
"""

import pandas as pd
import numpy as np
import datetime
import math
from sklearn.linear_model import LinearRegression
from scipy.stats import normaltest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False


ts = 'time'
gs = 'generator_rotating_speed'
gp = 'power'
temp_bear_dr = 'f_gen_bearing_temp'
temp_bear_ndr = 'r_gen_bearing_temp'
temp_cab = 'cabin_temp'
outpath_dr = '../Resource/generator_bearing_temp/dr'
outpath_ndr = '../Resource/generator_bearing_temp/ndr'
turbineName = 'turbineName'


def data_pre_processing(data):
    data = data.loc[:,[ts, gp, temp_bear_dr, temp_bear_ndr, temp_cab, gs]]
    data[ts] = pd.to_datetime(data[ts])
    data[ts] = data[ts].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    # data = data.set_index(date, drop = True).sort_index()

    data.dropna(subset=[gp], inplace=True)

    # data = data[data[gp]>5]  #这一句不能加

    for i in data.columns:
        if i == temp_bear_dr or i == temp_bear_ndr or i == temp_cab or i == gs :
            data[i] = data[i].interpolate(method='linear')
            #如果还有空的，就用后一个填充-20190916
            data[i]=data[i].fillna(method='backfill')
            data[i]=data[i].fillna(method='ffill')
    return data


def Generator_bearing_fsrc_train(data_train, target, outpath, wtid):
    # define features
    data_train = data_train.sort_values(by = ts)
    data_train['speed'] = data_train[gs] * math.pi * 2 / 60
    data_train['espeed'] = (np.abs(data_train[gs]) * 2 * math.pi / 60) ** 1.6
    data_train['power'] = abs(data_train[gp]) * 1000
    data_train['temp_cab_mean'] = 0.5 * (data_train[temp_cab] + data_train[temp_cab].shift(1))

    data_train['Y'] = data_train[target] - 0.5 *(data_train[temp_cab] + data_train[temp_cab].shift(1))
    data_train['x1'] = data_train[target].shift(1) - 0.5 *(data_train[temp_cab] + data_train[temp_cab].shift(1))

    data_train['x2'] = 0.5 * (data_train['speed'] + data_train['speed'].shift(1))
    data_train['x3'] = 0.5 * (data_train['espeed'] + data_train['espeed'].shift(1))
    data_train['x4'] = 0.5 * (data_train['power'] + data_train['power'].shift(1))
    data_train =  data_train.dropna()

    # train model
    X = data_train.loc[:,['x1','x2','x3','x4']]
    Y = data_train.loc[:,['Y']]
    model = LinearRegression().fit(X,Y)
    print( model.coef_, model.intercept_)
    # result output
    # data_train['x1_t']= data_train['x1']*model.coef_[0][0]
    # data_train['x2_t'] = data_train['x2'] * model.coef_[0][1]
    # data_train['x3_t'] = data_train['x3'] * model.coef_[0][2]
    # data_train['x4_t'] = data_train['x4'] * model.coef_[0][3]
    # data_train['y_pre'] = data_train['x1_t']+data_train['x2_t']+data_train['x3_t']+data_train['x4_t']+model.intercept_[0]


    if not os.path.exists(outpath):
        os.makedirs(outpath)
    model_file = outpath + os.sep + '{}'.format(wtid) + '_generatorBearingTemp_model.bin'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    f.close()

def generator_bearing_fsrc_main(data,wtid):

    global ts
    global gs
    global gp
    global temp_cab
    global outpath_dr
    global outpath_ndr

    # data pre-processing
    data_processed = data_pre_processing(data)
    print('data_pre_processing finished')
    # train models
    Generator_bearing_fsrc_train(data_processed,temp_bear_dr,outpath_dr,wtid)
    print(wtid, 'dr_train finished')
    Generator_bearing_fsrc_train(data_processed,temp_bear_ndr,outpath_ndr,wtid)
    print(wtid, 'ndr_train finished')



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
        all_data_10min[turbineName] = wt_id

        all_data_10min[ts] = pd.to_datetime(all_data_10min[ts])
        all_data_10min = all_data_10min.drop_duplicates([ts])
        vars_10min = all_data_10min.columns.to_list()

        train = all_data_10min[:int(len(all_data_10min) * 0.02)]

        generator_bearing_fsrc_main(train, wt_id)














