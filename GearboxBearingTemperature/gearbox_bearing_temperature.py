# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:02:26 2019

@author: TZ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from collections import Counter
import pymongo
import yaml
from GearboxBearingTemperature.Publicutils.getwarning import get_warning
# from GearboxBearingTemperature.Publicutils.utils import data_process



date = 'time'
gs = 'generator_rotating_speed'
ws='wind_speed'
gp = 'power'
temperature_cab = 'cabin_temp'
temperature_oil = 'gearbox_oil_temp'
gearbox_bearing_temperature1 = 'f_gearbox_bearing_temp'
gearbox_bearing_temperature2 = 'r_gearbox_bearing_temp'



threshold_01 = 0.8
threshold_00 = 1.6
threshold_1 = 1
n_lookback_alarm = 100
n_lookback_warning = 5
n_continuous = 2
border = 0.05
model_dr_path = 'Resource1/modeldr.pkl'
model_ndr_path = 'Resource1/modelndr.pkl'
#buffer_config_fp = 'Resource/db_config.yml'
def import_data_check(data):
    columns= data.columns
    check_result = dict()
    check_result['response'] = True
    check_result['status_code'] = '000'
    if data.isnull().all().any(): # 判断数据中某列是否全部为空值
        check_result['response'] = False
        check_result['status_code'] = '200'
    elif data.isnull().any().any():
        for i in data.columns: # 判断各列缺失值的比例是否大于5%
            missing_value_counts = len(data[i]) - data[i].count()
            missing_value_rate = (missing_value_counts / len(data[i])) * 100
            if missing_value_rate >= 5:
                check_result['response'] = False
                check_result['status_code'] = '100'
    if (data[columns[2:]].dtypes != 'float').any():
        # raise Exception('the Dtype of Input Data_Raw is not Float')
        check_result['status_code'] = '302'
    #duplicate values
    for i in columns[2:]:
        if sum(data[i].duplicated()) == len(data):
            # raise Exception('Input Data_Raw Contains Too Many Duplicates')
            check_result['status_code'] = '102'
    return check_result

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


def alarm_level(alarm_history, alarm, n_lookback_alarm):
    """
    get_alarm的子函数，根据历史报警积累确定相应的报警等级
    param alarm_series: 历史的报警序列
    param alarm: 当前时刻的报警状态
    return: 报警等级
    """
#    alarm_history_tmp = alarm_history[-n_lookback_alarm:]
    alarm_history.append(alarm)
    alarm_history_tmp=alarm_history[-n_lookback_alarm:]
    distance = 100-min(100 * len([x for x in alarm_history_tmp if x != 0]) / n_lookback_alarm  , 100)
    if alarm == 1:
        if 40 <= distance <= 60:
            alarm_level = 1
        elif 60< distance <= 80:
            alarm_level = 2
        elif 80 < distance:
            alarm_level = 3
    else:
        distance = 100
        alarm_level = 0
    return float(distance), int(alarm_level)
def find_majority(warning):
    """
    get_warning的子函数
    return: 多数类元素
    """
    vote_count = Counter(warning)
    top_two = vote_count.most_common(2)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]: # tie
        return 1
    return top_two[0][0]
def get_alarm(warning_dr_history, warning_ndr_history, alarm_dr_history, alarm_ndr_history, n_lookback_alarm, n_lookback_warning, n_continuous):
    """
    根据规则转化为警报，连续n个点或给定时间窗内超过半数预警则产生警报
    param warning_series: 历史的预警序列
    param n_lookback: 每个点的预警考虑前多少个点
    param n_continuous: 连续n个点连续预警
    return： 是否报警
    """
    alarm_dr = 0
    alarm_ndr = 0
    if all(a == 1 for a in warning_dr_history[-n_continuous:]) or find_majority(warning_dr_history[-n_lookback_warning:]) == 1:
        alarm_dr = 1
    if all(a == 1 for a in warning_ndr_history[-n_continuous:]) or find_majority(warning_ndr_history[-n_lookback_warning:]) == 1:
        alarm_ndr = 1
    final_level_dr = alarm_level(alarm_dr_history, alarm_dr, n_lookback_alarm)
    final_level_ndr = alarm_level(alarm_ndr_history, alarm_ndr, n_lookback_alarm)
    return final_level_dr, final_level_ndr


###main 函数的输入改掉了，resulthistory的离线格式比较麻烦，直接输入各个warninghistory-jimmy 20220706

def gearbox_temp_main(data, model_dr,model_ndr, warning_dr_history, warning_ndr_history, alarm_dr_history, alarm_ndr_history):

    global date
    global gs
    global gp
    global gearbox_bearing_temperature1
    global gearbox_bearing_temperature2
    global temperature_cab
    global temperature_oil
    global threshold_01
    global threshold_00
    global threshold_1
    global n_lookback_alarm
    global n_lookback_warning
    global n_continuous

    global border

    # assetName=str(data['turbineName'].values[0])
    # model_dr=modeldrdict[assetName]
    # model_ndr=modelndrdict[assetName]
    # warning_dr_history = []
    # warning_ndr_history = []
    # alarm_dr_history = []
    # alarm_ndr_history = []
    # try:
    #     resulthistory = resulthistory.loc[(resulthistory['deviceNo'] == data['assetName'].values[0]), :]
    #     for index,h in resulthistory.iterrows():
    #
    #         warning_dr_history.append(h['analysis_data']['warning_dr'])
    #         warning_ndr_history.append(h['analysis_data']['warning_ndr'])
    #         alarm_dr_history.append(h['gbt1_alarm'])
    #         alarm_ndr_history.append(h['gbt2_alarm'])
    # except:
    #     warning_dr_history.append(0)
    #     warning_ndr_history.append(0)
    #     alarm_dr_history.append(0)
    #     alarm_ndr_history.append(0)




    status_code = import_data_check(data)['status_code']
    if status_code == '000':
        time = data[date]
        data[date] = pd.to_datetime(time)
        data = data.reset_index()
        data_new = data_process(data, date, gp, gearbox_bearing_temperature1, gearbox_bearing_temperature2, temperature_oil, gs, temperature_cab)

        dr_hat = model_dr.predict(data_new.loc[:,['X1_dr','x2','x3','x5','x6']])
        res_dr = data_new['Y_dr'].values.reshape(-1, 1) - dr_hat

        ndr_hat = model_ndr.predict(data_new.loc[:,['X1_ndr','x2','x3','x5','x6']])
        
        res_ndr = data_new['Y_ndr'].values.reshape(-1, 1) - ndr_hat
        res_d = res_dr - res_ndr
        warning = get_warning(res_dr, res_ndr, threshold_01, threshold_00, threshold_1, border=0.2)
        warning_dr_history.append(warning[0])
        warning_ndr_history.append(warning[1])
        alarm = get_alarm(warning_dr_history, warning_ndr_history, alarm_dr_history, alarm_ndr_history, n_lookback_alarm, n_lookback_warning, n_continuous)
        alarm_dr, alarm_ndr = alarm[0], alarm[1]
        alarm_dr_history.append(alarm_dr[1])
        alarm_ndr_history.append(alarm_ndr[1])
        alarm_final = 0
        if np.any([alarm_dr[1], alarm_ndr[1]]) != 0:
            alarm_final = 1
        raw_data = {}
        raw_data['datetime'] = list(map(str, data_new[date].tolist()))
        raw_data['gbt1'] = data_new[gearbox_bearing_temperature1].values.tolist()
        raw_data['gbt2'] = data_new[gearbox_bearing_temperature2].values.tolist()
        raw_data['rs'] = data_new[gs].values.tolist()
        raw_data['gp'] = data_new[gp].values.tolist()
        raw_data['gbt1_prediction'] = dr_hat[:,0].tolist()
        raw_data['gbt2_prediction'] = ndr_hat[:,0].tolist()
        raw_data['tamperature_cab'] = data_new[temperature_cab].values.tolist()
        
        analysis_data = {}
        analysis_data['datetime'] = list(map(str, data_new[date].tolist()))
        analysis_data['gbt1_residual'] = res_dr[:,0].tolist()
        analysis_data['gbt2_residual'] = res_ndr[:,0].tolist()
        analysis_data['gbt1_residual_average'] = np.mean(res_dr)
        analysis_data['gbt2_residual_average'] = np.mean(res_ndr)
        analysis_data['res_d'] = res_d[:,0].tolist()
        analysis_data['warning_dr'] = warning[0] # Append this to warning_dr_history
        analysis_data['warning_ndr'] = warning[1] # Append this to warning_ndr_history
        analysis_data['gbt_residual_threshold'] = [threshold_00, threshold_01]
        analysis_data['res_d_threshold'] = threshold_1
    
        result = {}
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['status_code'] = status_code
        result['start_time'] = str(data_new[date].iloc[0])
        result['end_time'] = str(data_new[date].iloc[-1])
        result['gbt1_distance'] = alarm_dr[0]
        result['gbt2_distance'] = alarm_ndr[0]
        result['gbt1_alarm'] = alarm_dr[1] # Append this to alarm_dr_history
        result['gbt2_alarm'] = alarm_ndr[1] # Append this to alarm_ndr_history
        result['distance'] = np.min([alarm_dr[0], alarm_ndr[0]])  ###应该取小的
        result['alarm'] = alarm_final


        ##plot 故障数据显示
        if result['distance']< 90:

            plt.subplot(2, 2, 1)
            data_new['dr_hat'] = model_dr.predict(data_new.loc[:,['X1_dr','x2','x3','x5','x6']])
            data_new['Y_dr'].plot(label='(前轴承)温度_实际')
            data_new['dr_hat'].plot(label='(前轴承)温度_预测')

            res_dr = data_new['dr_hat'] - data_new['Y_dr']
            data_new['res_dr']=res_dr

            # data_new['res_dr'].plot(label='res')
            plt.title(str(data['time'][0])[:-6])
            plt.legend()

            plt.subplot(2, 2, 3)
            data_new['ndr_hat'] = model_ndr.predict(data_new.loc[:,['X1_ndr','x2','x3','x5','x6']])
            data_new['Y_ndr'].plot(label='(后轴承)温度_实际')
            data_new['ndr_hat'].plot(label='(后轴承)温度_预测')
            res_ndr = data_new['ndr_hat'] - data_new['Y_ndr']
            data_new['res_ndr'] =res_ndr
            # data_new['res_ndr'].plot(label='res')
            plt.title(str(data['time'][0])[:-6])
            plt.legend()

            plt.subplot(2, 2, 2)

            plt.hist(res_dr, bins=100, label='res_前轴承')
            plt.title('warning_dr' + str(warning[0]))
            plt.legend()

            plt.subplot(2, 2, 4)

            plt.hist(res_ndr, bins=100, label='res_后轴承')
            plt.title('warning_ndr' + str(warning[1]))
            plt.legend()
            plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.1, wspace=0.3, hspace=0.3)
            import os
            savePath = '../Result/gearbox_bearing_temp/fault/'
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            plt.savefig(savePath + data['turbineName'][0] +  str(data['time'][0])[:-6] + 'gearbox_bearing.jpg', format='jpg', dpi=plt.gcf().dpi,  bbox_inches='tight')
            plt.clf()
    else:
        raw_data = {}
        raw_data['datetime'] = list(map(str, data[date].tolist()))
        raw_data['gbt1'] = data[gearbox_bearing_temperature1].values.tolist()
        raw_data['gbt2'] = data[gearbox_bearing_temperature2].values.tolist()
        raw_data['rs'] = data[gs].values.tolist()
        raw_data['gp'] = data[gp].values.tolist()
        
        analysis_data = None
        
        result = {}
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['status_code'] = status_code
        result['start_time'] = str(data[date].iloc[0])
        result['end_time'] = str(data[date].iloc[-1])
        result['gbt1_distance'] = None
        result['gbt2_distance'] = None
        result['gbt1_alarm'] = None
        result['gbt2_alarm'] = None
        result['distance'] = None
        result['alarm'] = None
    return raw_data,analysis_data,status_code,result['start_time'],result['end_time'],result['gbt1_distance'],result['gbt2_distance'],result['gbt1_alarm'],result['gbt2_alarm'],result['distance'], result['alarm']

if __name__ == '__main__':

    data=pd.read_json('')
    resulthistory=pd.read_json('')
    with open(model_dr_path, 'rb') as fp:
        modeldrdict = pickle.load(fp)
    with open(model_ndr_path, 'rb') as f:
        modelndrdict = pickle.load(f)

    raw_data,analysis_data,status_code,start_time,end_time,gbt1_distance,gbt2_distance,gbt1_alarm,gbt2_alarm,distance,alarm=gearbox_temp_main(data,modeldrdict,modelndrdict,resulthistory)




