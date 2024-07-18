# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:48:09 2021

@author: Tzzy
"""

import pandas as pd
import numpy as np
from decimal import *
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pickle



level_gw = [0.1946, 0.2922, 0.4381]                   #原算法设计的level
rare = 0.12
threshold = 5

dt = 'time'
ws= 'wind_speed'
nac_position= 'nacelle_position'
wd= 'wind_direction'

def dataQulityCheck(data, status_code):
    '''
    数据质量检查：
    '''
    if len(data) == 0:
        status_code = '200'   #数据为空
    elif data.isnull().all().any():
        status_code = '201'   #某列数据为空
    elif len(data[data[nac_position].abs().fillna(0) > 800]) > 0:
        status_code = '203'   #偏航角度异常
    elif data[nac_position].diff().fillna(0).sum() == 0:
        status_code = '204'   #偏航角度没有变化
    elif len(data[data['wind_speed'].abs().fillna(0) < 2]) >= data.shape[0]/3:
        status_code = '301'   #符合工况的数据太少
    elif len(data[data['wind_speed'].abs().fillna(0) < 2]) == data.shape[0]:
        status_code = '302'   #没有符合工况的数据 
    return data, status_code




def distance_transform(percention):
    '''
    根据百分比计算健康值
    '''
    distance = 100
    correction = (1 - percention + rare) * 100
    
    if correction >= 100:
        distance = 100
    else:
        distance = (correction - 100 * rare) / (1 - rare)
    return distance



def alarm_transform(distance):
    '''
    根据健康值计算报警等级
    '''
    alarm = 0
    if distance >= 80:
        alarm = 0
    elif (distance >= 60) & (distance < 80):
        alarm = 1
    elif (distance >= 40) & (distance < 60):
        alarm = 2
    else:
        alarm = 3
        
    return alarm





def yaw_drive_abnormal_main(data):


    
    data.sort_values(by = dt).reset_index(inplace=True)
    
    
    status_code = '000'
    data, status_code = dataQulityCheck(data, status_code)
    
    data = data.dropna()
    
    if status_code == '000' and len(data) > 60:
        
        
        data['diff_t'] = ((pd.to_datetime(data[dt]) - pd.to_datetime(data[dt]).shift(1))/pd.Timedelta(1, 'S')).fillna(10).astype(int)       #计算两次上送数据的时间差
        data = data[~(data['diff_t'] == 0)]                        #删除重复时间数据          
        
        if len(data) > 60:
            data['yaw_speed'] = (data[nac_position].diff().abs().fillna(0)/data['diff_t']).values         #偏航速度等于偏航角度变化除以所用时间(机舱位置？）

            # plt.plot(data['yaw_speed'])
            # plt.show()

            data.loc[data['yaw_speed'] > 1.5,'yaw_speed'] = 0        #线上数据出现记录异常，硬性处理方法
            data['yaw_acceleration'] = (data['yaw_speed'].diff().abs().fillna(0)/data['diff_t']).values     #偏航加速度
            # plt.plot(data['yaw_acceleration'])
            # plt.show()

            ######这里有两个阈值，
            percention = len(data[(data['yaw_speed'] > 0) & (data['yaw_speed'] < 0.1) | (data['yaw_acceleration'] > 10.3)]) / len(data[data['yaw_speed'] > 0])            #计算边缘数据比例
            distance = distance_transform(percention)
            
    
            fatal = 0
            if len(data[(data['yaw_speed'] > 1.5)]) > 0:           #计算偏航速度过大的数量
                fatal = len(data[(data['yaw_speed'] > 1.5)])
            if len(data[(data['yaw_acceleration'] > 2)]) > 0:      #计算偏航加速度过大的数量
                fatal = fatal + len(data[(data['yaw_acceleration'] > 2)])
                

            
            distance = distance - fatal/5
            if distance < 0:
                distance = 0
            alarm = alarm_transform(distance)

    
    
            raw_data = dict()
            raw_data['nloc'] = [float(Decimal(x).quantize(Decimal('0.0000'))) for x in data[nac_position].values.tolist()]
            raw_data['wd'] = [float(Decimal(x).quantize(Decimal('0.0000'))) for x in data[wd].values.tolist()]
            raw_data['datetime'] = list(map(str, pd.to_datetime(data[dt].values.tolist())))
    
    
            analysis_data = dict()
            analysis_data['yaw_speed'] = [float(Decimal(x).quantize(Decimal('0.0000'))) for x in data['yaw_speed'].values.tolist()]
            analysis_data['threshold'] = [float(Decimal(x).quantize(Decimal('0.0000'))) for x in level_gw]
            analysis_data['datetime'] = list(map(str, pd.to_datetime(data[dt].values.tolist())))
    

        else:
            status_code = '301'
            analysis_data = dict()
            raw_data = dict()
            distance = 100
            alarm = 0
    else:
        status_code = '301'
        analysis_data = dict()
        raw_data = dict()
        distance = 100
        alarm = 0  
            
    result = dict()    
    result['start_time'] = ''
    result['end_time'] = ''
    if len(data) > 0:
        result['start_time'] = data[dt].values[0]
        result['end_time'] = data[dt].values[-1]

        

    result['raw_data'] = raw_data
    result['analysis_data'] = analysis_data
    result['status_code'] = status_code
    result['distance'] = distance
    result['alarm'] = alarm       
    
    start_time,end_time,raw_data,analysis_data,status_code,distance,alarm = result["start_time"],result["end_time"],result["raw_data"],result["analysis_data"],result["status_code"],result["distance"],result["alarm"]


    return start_time,end_time,raw_data,analysis_data,status_code,distance,alarm
    
    

if __name__ == '__main__':
    
    data = pd.read_csv(r'yaw_56_6.csv')
    data['datetime'] = data['dataTimeStamp'].astype('str') 
    
    data = data.dropna()
    result = dict()
    
      
    
    for i in range(28):
        if  i < 7:
            st = '2021-06-0' + str(i+1)
            et = '2021-06-0' + str(i+3)
        elif i < 10 :
            st = '2021-06-0' + str(i+1)
            et = '2021-06-' + str(i+3)
        else:
            st = '2021-06-' + str(i+1)
            et = '2021-06-' + str(i+3)
        temp = data[(data['datetime'] >= st) & (data['datetime'] <= et)]    
        result["start_time"],result["end_time"],result["raw_data"],result["analysis_data"],result["status_code"],result["distance"],result["alarm"] = yaw_drive_abnormal_main(temp) 
        print(result['distance'])
        print(result['status_code'])
    












  




