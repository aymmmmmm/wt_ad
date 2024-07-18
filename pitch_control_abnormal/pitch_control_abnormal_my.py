# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 08:50:25 2022
@author: Tzzy
"""

import pandas as pd
import numpy as np
import decimal
import statsmodels.api as sm
from scipy import stats


t = 'time'
wind_speed = 'wind_speed'               #风速
speed_1 = 'pitch_speed_1'               #变桨速度1
speed_2 = 'pitch_speed_2'               #变桨速度2
speed_3 = 'pitch_speed_3'               #变桨速度3
power = 'power'                  #有功功率


def data_check(data): 
    status_code = '000'
    if len(data) < 100:
        status_code = '103'
    return status_code

def rank_transform(distance):
    if 0 <= distance <= 40:
        rank = 3
    elif 40 < distance <= 60:
        rank = 2
    elif 60 < distance <= 80:
        rank = 1
    elif 80 < distance <= 100:
        rank = 0
    else:
        rank = np.nan
    return rank


def pitch_control_abnormal_main(data):
    
    """
    使用假设检验的方法判断变桨相对速度的均值是否为零，显著性水平选择为0.001，决定了如果相对速度均值不为零的把握有99.9%
    在判断相对速度均值不为零的情况下，如果出现变桨速度超限，则直接会导致报警
    若没有出现变桨速度超限，则以相对速度标准差的偏移来判断是否报警
    """

    
    test_data = data.fillna(method='ffill').fillna(method='bfill').drop_duplicates(subset=[t]).sort_values(by = t)    #删除重复时间数据，对时间重新排序
    status_code = data_check(test_data)
    result = dict()
    if status_code == '103':
        result['start_time'] = ''
        result['end_time'] = ''
        result['raw_data'] = ''
        result['analysis_data'] = ''
        result['status_code'] = '000'
        result['alarm'] = 0
        result['distance'] = 100
    else:
        raw_data = dict()
        raw_data['datetime'] = list(map(str, pd.to_datetime(test_data[t], format='%Y-%m-%d %H:%M:%S').tolist()))
        raw_data['ps1'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), test_data[speed_1].values.tolist()))
        raw_data['ps2'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), test_data[speed_2].values.tolist()))
        raw_data['ps3'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), test_data[speed_3].values.tolist()))
        raw_data['gp'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), test_data[power].values.tolist()))
        raw_data['ws'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), test_data[wind_speed].values.tolist()))
        
        
        
        
        t1,p1 = stats.ttest_1samp((test_data[speed_1]-test_data[speed_2]),0)            #相对速度的单样本假设检验
        t2,p2 = stats.ttest_1samp((test_data[speed_1]-test_data[speed_3]),0)            #相对速度的单样本假设检验
        t3,p3 = stats.ttest_1samp((test_data[speed_3]-test_data[speed_2]),0)            #相对速度的单样本假设检验
        
        distance1 = 100
        if p1 > 0.001:
            distance1 = 80 + 20 * p1                                     #p值越小则，则健康值越接近报警状态
        elif abs(test_data[speed_1].max()) > 5:
            distance1 = 80 - abs(((test_data[speed_1] - test_data[speed_2]).std() - 0.0537)) / (3 * 0.0537)      #只要出现变桨速度超限，就会报警
            if distance1 < 0:
                distance1 = 0
        else:
            distance1 = 100 - abs(((test_data[speed_1] - test_data[speed_2]).std() - 0.0537)) / (3 * 0.0537)     #偏移3倍标准差的量越多，越接近报警
        distance2 = 100    
        if p2 > 0.001:
            distance2 = 80 + 20 * p2   ###这里应该是P2吧 jimmy 20221215
        elif abs(test_data[speed_3].max()) > 5:
            distance2 = 80 - abs(((test_data[speed_1] - test_data[speed_3]).std() - 0.0537)) / (3 * 0.0537)      #只要出现变桨速度超限，就会报警
            if distance2 < 0:
                distance2 = 0
        else:
            distance2 = 100 - abs(((test_data[speed_1] - test_data[speed_3]).std() - 0.0537)) / (3 * 0.0537)     #偏移3倍标准差的量越多，越接近报警        
        distance3 = 100
        if p3 > 0.001:
            distance3 = 80 + 20 * p3  ###这里应该是P3吧 jimmy 20221215
        elif abs(test_data[speed_2].max()) > 5:
            distance3 = 80 - abs(((test_data[speed_3] - test_data[speed_2]).std() - 0.0537)) / (3 * 0.0537)      #只要出现变桨速度超限，就会报警
            if distance3 < 0:
                distance3 = 0
        else:
            distance3 = 100 - abs(((test_data[speed_3] - test_data[speed_2]).std() - 0.0537)) / (3 * 0.0537)     #偏移3倍标准差的量越多，越接近报警

        
        analysis_data = dict()
        kde1 = sm.nonparametric.KDEUnivariate((test_data[speed_1]-test_data[speed_2]).values.astype('float64'))  #相对速度的核密度曲线
        kde1.fit(bw=0.5)
        kde2 = sm.nonparametric.KDEUnivariate((test_data[speed_1]-test_data[speed_3]).values.astype('float64'))  #相对速度的核密度曲线
        kde2.fit(bw=0.5)
        kde3 = sm.nonparametric.KDEUnivariate((test_data[speed_3]-test_data[speed_2]).values.astype('float64'))  #相对速度的核密度曲线
        kde3.fit(bw=0.5)
        analysis_data['density_x1'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde1.support))      
        analysis_data['density_y1'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde1.density))
        analysis_data['density_x2'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde2.support))
        analysis_data['density_y2'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde2.density))
        analysis_data['density_x3'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde3.support))
        analysis_data['density_y3'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde3.density))
        kde = sm.nonparametric.KDEUnivariate(np.random.normal(0,0.5,1000))       #标准相对速度核密度曲线，假设为正态分布，这里为了不保存和导入较大的核密度模型，使用正态分布来逼近
        kde.fit(bw=0.5)
        analysis_data['base_x'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde.support))
        analysis_data['base_y'] = list(map(lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal('0.0000'))), kde.density))
        
        result['start_time'] = str(test_data[t].iloc[0])
        result['end_time'] = str(test_data[t].iloc[-1])
        result['result1'] = distance1
        result['result2'] = distance2
        result['result3'] = distance3
        result['distance'] = round(min(result['result1'], result['result2'], result['result3']), 2)        #健康值选取三支中较小的
        result['alarm1']  = rank_transform(result['result1'])
        result['alarm2']  = rank_transform(result['result2'])
        result['alarm3']  = rank_transform(result['result3'])
        result['alarm'] = max(result['alarm1'], result['alarm2'], result['alarm3'])
        result['status_code'] = status_code
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        

        
        
    start_time = result['start_time']
    end_time = result['end_time']
    raw_data = result['raw_data']
    analysis_data = result['analysis_data']
    status_code = result['status_code']
    alarm = result['alarm']
    distance = result['distance']

    return start_time,end_time,raw_data,analysis_data,status_code,alarm,distance


if __name__ == '__main__':
    df = pd.read_json("D:/DT/PitchControlAbnormal/PitchControlAbnormal-31-data.json")
    start_time, end_time, raw_data, analysis_data, status_code, alarm, distance = pitch_control_abnormal_main(df)
    print(alarm,distance)
