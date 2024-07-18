#! /usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:34:36 2019

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
# import seaborn as sns
import os
import pickle
import glob

ts = 'time'
ws='wind_speed'
gs = 'generator_rotating_speed'
gp = 'power'
temperature_cab = 'cabin_temp'
temperature_oil = 'gearbox_oil_temp'
gearbox_bearing_temperature1 = 'f_gearbox_bearing_temp'
gearbox_bearing_temperature2 = 'r_gearbox_bearing_temp'
threshold_rs = 92


def data_pre_processing(data):
    data = data.loc[:,[ts, gp, gearbox_bearing_temperature1, gearbox_bearing_temperature2, temperature_oil, gs, temperature_cab]]
#    data = data.loc[:,[date,gp,gearbox_bearing_temperature1,rs,temperature_cab]]
    data = data.loc[(data[temperature_oil] < 60) & (data[gp] > 0), :]

    data[ts] = pd.to_datetime(data[ts])
#    data_processed = data.interpolate(limit_direction='both')
    data_processed = data.dropna(axis=0) 
    if len(data[ts]) != len(data_processed[ts]):
        print('数据中存在缺失值')
    return data_processed

def Gearbox_bearing_train(data_train,target):
    # define features
    data_train = data_train.sort_values(by = ts)
#    data_train['t1'] = threshold_rs - data_train[rs].shift(1)
#    t1 = data_train['t1'].tolist()
#    data_train['t2'] = 0 - data_train['t1']
#    t2 = data_train['t2'].tolist()
    data_train['x1'] = data_train[target].shift(1)
    data_train['x2'] = data_train[gearbox_bearing_temperature1].shift(1)
    data_train['x3'] = data_train[temperature_cab].shift(1)
    data_train['x4'] = data_train[gearbox_bearing_temperature2].shift(1)
#    data_train['x4'] = (data_train[rs].shift(1)**0.5)* np.heaviside(t1,0)
#    data_train['x5'] = data_train[rs].shift(1) * np.heaviside(t2,1)
    data_train['x5'] = data_train[gs].shift(1)
    data_train['x6'] = data_train[gp].shift(1) 
    data_train['Y'] = data_train[target]
    data_train =  data_train.dropna()
    #add label
#    data_train = data_train.reset_index()
#    data_train['label'] = 0
#    for i in range(0,len(data_train[rs])):
#        if data_train.loc[i,gearbox_bearing_temperature1]>70 or data_train.loc[i,gearbox_bearing_temperature2]>70 or data_train.loc[i,temperature_oil]>60 or data_train.loc[i,temperature_cab] >47:
#            data_train.loc[i,'label'] = 1
#        elif (data_train.loc[i,gearbox_bearing_temperature1]<=70 and data_train.loc[i,gearbox_bearing_temperature1]>65) or (data_train.loc[i,gearbox_bearing_temperature2]<=70 and data_train.loc[i,gearbox_bearing_temperature2]>65) or (data_train.loc[i,temperature_oil]<=60 and data_train.loc[i,temperature_oil]>50) or (data_train.loc[i,temperature_cab]<=47 and data_train.loc[i,temperature_cab]>45):
#            if i==0:
#                if data_train.loc[i,gearbox_bearing_temperature1]> data_train.loc[i+1,gearbox_bearing_temperature1] or data_train.loc[i,gearbox_bearing_temperature2]> data_train.loc[i+1,gearbox_bearing_temperature2] or data_train.loc[i,temperature_oil]> data_train.loc[i+1,temperature_oil] or data_train.loc[i,temperature_cab] > data_train.loc[i+1,temperature_cab]:
#                    data_train.loc[i,'label'] = 1 
#            else:
#                if data_train.loc[i-1,'label'] == 1:
#                    data_train.loc[i,'label'] = 1
#    data_train['x7'] = data_train['label'].shift(1)

    X = data_train.loc[:,['x1','x2','x3','x4','x5','x6']]
#    X = data_train.loc[:,['x1','x3','x4','x5','x6']]
    Y = data_train.loc[:,['Y']]
    model = LinearRegression().fit(X,Y)
    # result output
#    if not os.path.exists(outpath):
#        os.makedirs(outpath)
#    model_file = outpath + os.sep + 'WT{}'.format(wtid[2:4]) + '_' + 'model.bin'
#    with open(model_file, 'wb') as f:
#        pickle.dump(model, f)
#    f.close()
    return model
                           

def Gearbox_bearing_predict(data_test,model,target):
    # define features
#    data_test['t1'] = threshold_rs - data_test[rs].shift(1)
#    t1 = data_test['t1'].tolist() 
#    data_test['t2'] = 0 - data_test['t1']
#    t2 = data_test['t2'].tolist() 
    data_test['x1'] = data_test[target].shift(1)
    data_test['x2'] = data_test[gearbox_bearing_temperature1].shift(1)
    data_test['x3'] = data_test[temperature_cab].shift(1)
    data_test['x4'] = data_test[gearbox_bearing_temperature2].shift(1)
    data_test['x5'] = data_test[gs].shift(1)
    data_test['x6'] = data_test[gp].shift(1) 
    data_test['Y'] = data_test[target]
    data_test =  data_test.dropna()
    #add label
#    data_test = data_test.reset_index()
#    data_test['label'] = 0
#    for i in range(0,len(data_test[rs])):
#        if data_test.loc[i,gearbox_bearing_temperature1]>70 or data_test.loc[i,gearbox_bearing_temperature2]>70 or data_test.loc[i,temperature_oil]>60 or data_test.loc[i,temperature_cab] >47:
#            data_test.loc[i,'label'] = 1
#        elif (data_test.loc[i,gearbox_bearing_temperature1]<=70 and data_test.loc[i,gearbox_bearing_temperature1]>65) or (data_test.loc[i,gearbox_bearing_temperature2]<=70 and data_test.loc[i,gearbox_bearing_temperature2]>65) or (data_test.loc[i,temperature_oil]<=60 and data_test.loc[i,temperature_oil]>50) or (data_test.loc[i,temperature_cab]<=47 and data_test.loc[i,temperature_cab]>45)  :
#            if i==0:
#                if data_test.loc[i,gearbox_bearing_temperature1]> data_test.loc[i+1,gearbox_bearing_temperature1] or data_test.loc[i,gearbox_bearing_temperature2]> data_test.loc[i+1,gearbox_bearing_temperature2] or data_test.loc[i,temperature_oil]> data_test.loc[i+1,temperature_oil] or data_test.loc[i,temperature_cab] > data_test.loc[i+1,temperature_cab]:
#                    data_test.loc[i,'label'] = 1 
#            else:
#                if data_test.loc[i-1,'label'] == 1:
#                    data_test.loc[i,'label'] = 1 
#    data_test['x7'] = data_test['label'].shift(1)
    
    # test
    X = data_test.loc[:,['x1','x2','x3','x4','x5','x6']]
    data_test['predict'] = model.predict(X)
    data_test['residual'] = data_test['Y'] - data_test['predict']
#    ax=data_test.plot(x='time',y='predict',label=target+'predict')
##    data_test['residual'] = data_test['Y'] - data_test['predict']
#    data_test.plot(x='time',y='Y',label=target+'actual',ax=ax)
##    data_test.plot(x='time',y='Y_previous',ax=ax)
#    plt.show()
#    data_test['residual'].plot(kind='kde')
#    plt.show()
    # result output
#    residual = np.array(data_test['residual'],dtype='float')
    #残差平均值
#    residual_mean=residual.mean()
    #概率分布图
#    if not os.path.exists(outpath):
#        os.makedirs(outpath)
#    res_file = outpath + os.sep + 'WT{}'.format(wtid[2:4]) + '_' + 'residual.bin'
#    with open(res_file, 'wb') as f:
#        pickle.dump(residual, f)
#    f.close()                    
    return data_test['residual']
    
    
    
def gearbox_oil_main(data):
    '''
    :param data: 训练数据输入
    '''
    # 数据预处理
    data_processed = data_pre_processing(data)
    data_train = data_processed[:int(0.6*len(data_processed))]
    data_test = data_processed[int(0.6*len(data_processed)):]
    # 训练模型并保存
    model_dr = Gearbox_bearing_train(data_train,temperature_oil) #输出模型
    re=Gearbox_bearing_predict(data_test,model_dr,temperature_oil) #输出残差
#    df2=pd.DataFrame()
#    df2['base_plot']=re
#    df2.plot(kind='kde')
#    plt.title('wt19_residual')
#    plt.show()
    
    return model_dr,re    
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

        train = all_data_10min[:int(0.4 * len(all_data_10min))]



        h=gearbox_oil_main(train)

        outpath = r'../Resource/gearbox_oil_temp'  # 齿轮箱油温模型输出路径
        if not os.path.exists(outpath):
            os.makedirs(outpath)

####################单个保存
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        model_file = outpath + os.sep + wt_id + '_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(h[0], f)
        f.close()

        model2_file = outpath + os.sep + wt_id +'_residual.pkl'
        with open(model2_file, 'wb') as f:
            pickle.dump(h[1], f)
        f.close()


