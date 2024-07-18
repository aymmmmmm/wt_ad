# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:41:29 2019

@author: TZ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from collections import Counter
import warnings

warnings.filterwarnings("ignore")
import pickle
# import pymongo
# import yaml
# import statsmodels.api as sm
import random
from decimal import *
import sys  # 导入sys模块
sys.setrecursionlimit(30000)


ts = 'time'
gs = 'generator_rotating_speed'
gp = 'power'
temperature_cab = 'cabin_temp'
temperature_oil = 'gearbox_oil_temp'
gearbox_bearing_temperature1 = 'f_gearbox_bearing_temp'
gearbox_bearing_temperature2 = 'r_gearbox_bearing_temp'

turbineName = 'turbineName'

window = 144
step = 12
n_lookback_warning = 10
n_lookback_alarm = 100
n_continuous = 3

threshold_rs=92

def data_preprocess(data):
    data = data.loc[:,
           [ts, gp, gearbox_bearing_temperature1, gearbox_bearing_temperature2, temperature_oil, gs, temperature_cab]]
    time = data[ts]
#    data = data.loc[(data[gp] > 0), :]
    data[ts] = pd.to_datetime(time)
#    data = data.resample('T', on=date).mean()
    data = data.reset_index()
    data = data.sort_values(by=ts)
#    threshold_dr = 92
#    data['t1'] = threshold_dr - data[rs].shift(1)
#    t1 = data['t1'].tolist()
#    data['t2'] = 0 - data['t1']
#    t2 = data['t2'].tolist()
    data = data.sort_values(by=ts)
    data['X1_dr'] = data[temperature_oil].shift(1)
    data['x2'] = data[gearbox_bearing_temperature1].shift(1)
    data['x3'] = data[temperature_cab].shift(1)
    data['x4'] = data[gearbox_bearing_temperature2].shift(1)
#    data['x4'] = (data[rs].shift(1) ** 0.5) * np.heaviside(t1, 0)
#    data['x5'] = data[rs].shift(1) * np.heaviside(t2, 1)
    data['x5'] = data[gs].shift(1)
    data['x6'] = data[gp].shift(1)
    data['Y_dr'] = data[temperature_oil]

#    data = data.reset_index()
#    data['label'] = 0
#    for i in range(0, len(data[rs])):
#        if data.loc[i, gearbox_bearing_temperature1] > 70 or data.loc[i, gearbox_bearing_temperature2] > 70 or data.loc[
#            i, temperature_oil] > 60 or data.loc[i, temperature_cab] > 47:
#            data.loc[i, 'label'] = 1
#        elif (data.loc[i, gearbox_bearing_temperature1] <= 70 and data.loc[i, gearbox_bearing_temperature1] > 65) or (
#                data.loc[i, gearbox_bearing_temperature2] <= 70 and data.loc[i, gearbox_bearing_temperature2] > 65) or (
#                data.loc[i, temperature_oil] <= 60 and data.loc[i, temperature_oil] > 50) or (
#                data.loc[i, temperature_cab] <= 47 and data.loc[i, temperature_cab] > 45):
#            if i == 0:
#                if data.loc[i, gearbox_bearing_temperature1] > data.loc[i + 1, gearbox_bearing_temperature1] or \
#                        data.loc[i, gearbox_bearing_temperature2] > data.loc[i + 1, gearbox_bearing_temperature2] or \
#                        data.loc[i, temperature_oil] > data.loc[i + 1, temperature_oil] or data.loc[
#                    i, temperature_cab] > data.loc[i + 1, temperature_cab]:
#                    data.loc[i, 'label'] = 1
#            else:
#                if data.loc[i - 1, 'label'] == 1:
#                    data.loc[i, 'label'] = 1
#    data['x7'] = data['label'].shift(1)
    data = data.dropna()
    return data


# Function #1
def estimate_sigma(residuals):
    """
    使用概率密度函数的估计计算标准差
    param residuals: 预测残差序列
    return: 计算的标准差
    """
    if len(residuals.shape) != 1:
        try:
            residuals = residuals.reshape(-1)
        except:
            print('Please check the shape of the residuals')
    pdf = norm.pdf(residuals)
    prob = pdf / pdf.sum()  # these are probabilities
    mu = residuals.dot(prob)  # mean value
    mom2 = np.power(residuals, 2).dot(prob)  # 2nd moment
    var = mom2 - mu ** 2  # variance
    return np.sqrt(var)  # standard deviation


# Function #2
def compute_penalty(value, mu, sigma):
    """
    根据正态分布计算各区间对应的惩罚项
    param value: 单个残差值
    param mu: 均值，默认为0
    param sigma: 计算得到的标准差
    return: 该残差值对应的惩罚项
    """
    if mu - sigma < value < mu + sigma:
        return 1
    elif ((mu - 2 * sigma < value <= mu - sigma) | (mu + sigma <= value < mu + 2 * sigma)):
        return 2
    elif ((mu - 3 * sigma < value <= mu - 2 * sigma) | (mu + 2 * sigma <= value < mu + 3 * sigma)):
        return 3
    elif ((value <= mu - 3 * sigma) | (mu + 3 * sigma <= value)):
        return 4


# Function #3
def count_datapoints(residuals, standard_sigma):
    """
    计算落到每个分布区间的样本点个数
    param residuals: 残差序列
    param sigma: 计算得到的标准差
    return: 4个区间样本点个数
    """
    array = np.zeros((len(residuals),))
    for i in range(len(array)):
        array[i] = compute_penalty(residuals.reshape(-1)[i], standard_sigma)
    n_1 = (array == 1).sum()
    n_2 = (array == 2).sum()
    n_3 = (array == 3).sum()
    n_4 = (array == 4).sum()
    return n_1, n_2, n_3, n_4


# Function #4
def compute_HI(residuals, standard_mu, standard_sigma):
    """
    rolling_HI的子函数
    return: 4个HI的值
    """
    array = np.zeros((len(residuals),))
    for i in range(len(array)):
        array[i] = compute_penalty(residuals.reshape(-1)[i], standard_mu, standard_sigma)
    sum_Cn = (array == 1).sum() * 1 + (array == 2).sum() * 2 + (array == 3).sum() * 3 + (array == 4).sum() * 4
    HI_1 = (array == 1).sum() * 1 / sum_Cn
    HI_2 = (array == 2).sum() * 2 / sum_Cn
    HI_3 = (array == 3).sum() * 3 / sum_Cn
    HI_4 = (array == 4).sum() * 4 / sum_Cn
    return HI_1, HI_2, HI_3, HI_4


def rolling_HI(data, model, residual, window, step):
    """
    滑窗计算Health Index
    param data: 待预测的原始数据, data frame
    param model: 训练集得到的预测模型
    param residual: 测试集的预测残差
    param window: 窗宽
    param step: 步长
    return: 4个HI的滑窗序列, matrix(4 * n)
    """
    HI = np.zeros((1, 4)).reshape(-1, 1)
#    data = data_preprocess(data)
    for i in range(window, len(data), step):
        data_tmp = data.iloc[i - window: i, :]
        x_tmp = data_tmp.loc[:, ['X1_dr', 'x2', 'x3','x4', 'x5', 'x6']].values
        y_tmp = data_tmp.loc[:, ['Y_dr']].values.reshape(-1, 1)
        pred_tmp = model.predict(x_tmp)
        diff_tmp = y_tmp - pred_tmp
        HI_tmp = np.array(compute_HI(diff_tmp, 0, estimate_sigma(residual))).reshape(-1, 1)
        HI = np.concatenate((HI, HI_tmp), axis=1)
    HI = HI[:, 1:]
    return HI


# Function #5
def find_majority(warning):
    """
    get_warning的子函数
    return: 多数类元素
    """
    vote_count = Counter(warning)
    top_two = vote_count.most_common(2)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:  # tie
        return 1
    return top_two[0][0]


def get_warning(HI, warning_history):
    """
    根据滑窗计算得到的HI得到预警
    param HI: 计算得到的HI
    param warning_history: 历史的预警序列
    return: 是否预警, 0 or 1
    将本次运行的预警结果添加到历史预警序列库中
    """
    warning = []
    for i in range(HI.shape[1]):
        if HI[:, i][0] + HI[:, i][1] < HI[:, i][2] + HI[:, i][3]:
            warning.append(1)
        else:
            warning.append(0)
    final_warning = find_majority(warning)
    return final_warning


# Function #6
def alarm_level(alarm_history, n_lookback_alarm, alarm):
    """
    get_alarm的子函数，根据历史报警积累确定相应的报警等级
    param alarm_history: 历史的报警序列
    param alarm: 当前时刻的报警状态
    return: 报警等级
    """
    alarm_history_tmp = alarm_history[-n_lookback_alarm:]
    distance = 100 - min(100 * len([x for x in alarm_history_tmp if x != 0]) / n_lookback_alarm, 100)
    alarm_level = 0
    if alarm == 1:
        if 40 < distance < 60:
            alarm_level = 1
        elif 20 < distance <= 40:
            alarm_level = 2
        elif 0 <= distance <= 20:
            alarm_level = 3
    else:
        distance = 100
        alarm_level = 0
    return float(distance), int(alarm_level) , alarm  ##这里是最终的报警等级，并非标记alarm jimmy-20221108


def get_alarm(warning_history, alarm_history, n_lookback_warning, n_lookback_alarm, n_continuous):
    """
    根据规则转化为警报，连续n个点或给定时间窗内超过半数预警则产生警报
    param warning_history: 历史的预警序列
    param n_lookback: 每个点的预警考虑前多少个点
    param n_continuous: 连续n个点连续预警
    return： 是否报警
    """
    alarm = 0
    if warning_history[-n_continuous:] == 1 or find_majority(warning_history[-n_lookback_warning:]) == 1:
        alarm = 1
    final_level = alarm_level(alarm_history, n_lookback_alarm, alarm)
    return final_level


def import_data_check(data):
    check_result = dict()
    check_result['response'] = True
    check_result['status_code'] = '000'
    if data.isnull().all().any():  # 判断数据中某列是否全部为空值
        check_result['response'] = False
        check_result['status_code'] = '200'
    elif data.isnull().any().any():
        for i in data.columns:  # 判断各列缺失值的比例是否大于5%
            missing_value_counts = len(data[i]) - data[i].count()
            missing_value_rate = (missing_value_counts / len(data[i])) * 100
            if missing_value_rate >= 5:
                check_result['response'] = False
                check_result['status_code'] = '100'
    return check_result


def gearbox_oil_main(data, model, base_plot, warning_history, alarm_history):
    

    global gs
    global gp
    global temperature_cab
    global temperature_oil
    global gearbox_bearing_temperature1
    global gearbox_bearing_temperature2
    global window
    global step
    global n_lookback_alarm
    global n_lookback_warning
    global n_continuous
    global model_path
    global buffer_config_fp
    global base_plot_path
    # assetName=str(data[turbineName].values[0])
    # # model=modeldict[assetName]
    # # base_plot= base_plotdict[assetName]
    #######分开打包 -- jimmy --20221016
    # model=modeldict
    # base_plot= base_plotdict


    # alarm_history = []
    # warning_history = []
    # try:
    #     resulthistory = resulthistory.loc[(resulthistory['deviceNo'] == data['assetName'].values[0]), :]
    #     for index,h in resulthistory.iterrows():
    #         warning_history.append(h['analysis_data']['warning'])
    #         alarm_history.append(h['alarm'])
    # except:
    #     warning_history.append(0)
    #     alarm_history.append(0)





    status_code = import_data_check(data)['status_code']
    if status_code == '000':
        data_new = data_preprocess(data)
        y_hat = model.predict(data_new.loc[:, ['X1_dr', 'x2', 'x3','x4',  'x5', 'x6']])
        y_hat = np.array([item[0] for item in y_hat])
        residual = data_new['Y_dr'].values - y_hat
        HI = rolling_HI(data_new, model, base_plot.values, window, step)
        warning = get_warning(HI, warning_history)
        warning_history.append(warning)
        alarm = get_alarm(warning_history, alarm_history, n_lookback_warning, n_lookback_alarm, n_continuous)
        alarm_history.append(alarm[2])
#    status_code = import_data_check(data)['status_code']
#    if status_code == '000':
#        data_new = data_preprocess(data)
#
#    HI = rolling_HI(data_new, model, base_plot, window, step)
#    warning = get_warning(HI, warning_history)
#    warning_history.append(warning)
#    alarm = get_alarm(warning_history, alarm_history, n_lookback_warning, n_lookback_alarm, n_continuous)
#    alarm_history.append(alarm[1])
    
    
    
#    df=pd.DataFrame()
#    df['residual']=residual
#    df['residual'].plot(kind='kde')
#    plt.title('残差分布')
#    plt.show()

#    kde1base = sm.nonparametric.KDEUnivariate(base_plot)
#    kde1base.fit(bw=1)
#    kde1 = sm.nonparametric.KDEUnivariate(residual)
#    kde1.fit(bw=1)
#
#    pred_plot_t = pd.DataFrame()
#    if len(kde1.density) < 2000:
#        sample_size = len(kde1.density)
#    else:
#        sample_size = 2000
#    sample_len = random.sample(list(range(len(kde1.density))), sample_size)
#    pred_plot_t['support'] = kde1.support[sample_len]
#    pred_plot_t['density'] = kde1.density[sample_len]
#
#    ###  sort values
#    pred_plot_t = pred_plot_t.sort_values(by='support')
#    pred_plot = dict()
#    pred_plot['support'] = pred_plot_t['support'].values
#    pred_plot['density'] = pred_plot_t['density'].values

#    base_plot_t = pd.DataFrame()
#    if len(kde1base.density) < 2000:
#        sample_size = len(kde1base.density)
#    else:
#        sample_size = 2000
#    sample_len = random.sample(list(range(len(kde1base.density))), sample_size)
#    base_plot_t['support'] = kde1base.support[sample_len]
#    base_plot_t['density'] = kde1base.density[sample_len]
#
#    ###  sort values
#    base_plot_t = base_plot_t.sort_values(by='support')
#    base_plot = dict()
#    base_plot['support'] = base_plot_t['support'].values
#    base_plot['density'] = base_plot_t['density'].values

    # Construct outputs
        raw_data = {}
        # raw_data['datetime'] = list(map(str, data_new[ts].tolist()))
        raw_data['datetime'] =  data_new[ts].tolist()
        raw_data['oiltemp'] = data_new[temperature_oil].values.tolist()
        raw_data['pred_y'] = y_hat.tolist()
        raw_data['rs'] = list(map(str, data_new[gs].tolist()))
        raw_data['gp'] = list(map(str, data_new[gp].tolist()))
    
        # analysis_data = {}
        analysis_data = dict()
        # analysis_data['online_x'] = list(map(str, data_new[ts].tolist()))
        analysis_data['online_x']=data_new[ts].tolist()
        analysis_data['online_y'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), residual.tolist()))
        # analysis_data['density_x1'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), pred_plot['support']))
        # analysis_data['density_y1'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), pred_plot['density'] ))
        # analysis_data['base_x'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), base_plot['support']))
        # analysis_data['base_y'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), base_plot['density']))
        # analysis_data['pred_y'] = y_hat.tolist()
        analysis_data['warning'] = warning
        analysis_data['alarm_history '] = alarm_history 
    
        result = {}
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['status_code'] = status_code
        result['start_time'] = str(data_new[ts].iloc[0])
        result['end_time'] = str(data_new[ts].iloc[-1])
        result['distance'] = alarm[0]
        result['alarm'] = alarm[1]  # Append this to alarm_history

    else:
        raw_data = {}
        data_new=data
        raw_data['datetime'] = list(map(str, data[ts].tolist()))
        raw_data['oiltemp'] = data[temperature_oil].values.tolist()
        raw_data['pred_y'] = None
        raw_data['rs'] = list(map(str, data[gs].tolist()))
        raw_data['gp'] = list(map(str, data[gp].tolist()))

        analysis_data = None
        alarm=[None,None]
        result = {}
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['status_code'] = status_code
        result['start_time'] = str(data[ts].iloc[0])
        result['end_time'] = str(data[ts].iloc[-1])
        result['distance'] = None
        result['alarm'] = None

    return result['raw_data'], result['analysis_data'], result['status_code'], result['start_time'], result['end_time'], result['distance'], result['alarm']
if __name__ == '__main__':
    #读取训练数据
    import glob
    import pandas
    # df=pd.read_json(r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\GearboxOilTemperature\GearboxOilTemperature-31-data.json')
    # resulthistory=pd.read_json(r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\GearboxOilTemperature\GearboxOilTemperature-31-resulthistory.json')
    df=pd.read_json(r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\GearboxOilTemperature\GearboxOilTemperature-10\GearboxOilTemperature-10-data.json')
    resulthistory=pd.read_json(r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\GearboxOilTemperature\GearboxOilTemperature-10\GearboxOilTemperature-10-resulthistory.json')
#    data=pd.read_csv('D:\风电算法-发电机齿轮箱\Pre_process_1026\W031.csv')
#    data[date] = pd.to_datetime(data[date])
#    data_test1 = data[data[date] >= pd.to_datetime('2021-07-27 0:00')]
#    df = data_test1[data_test1[date] < pd.to_datetime('2021-07-31 23:59')]    
#    data_processed = df.interpolate(limit_direction='both')
#
#    data_processed = data_processed.dropna(axis=0) 
#    data_train = data_processed[data_processed[date] < pd.to_datetime('2021-07-01 0:00')]

    model_path = 'Resource2/10_model_0.24.pkl'
    base_plot_path = 'Resource2/10_residual_0.24.pkl'

    with open(model_path, 'rb') as fp:
        modeldict = pickle.load(fp)

    with open(base_plot_path, 'rb') as f:
        base_plotdict = pickle.load(f)

    raw_data,analysis_data,status_code,start_time,end_time,distance,alarm=gearbox_oil_main(df, modeldict, base_plotdict,resulthistory)

    pass