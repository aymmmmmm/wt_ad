#! /usr/bin/env python 
# -*- coding: utf-8 -*-
"""
date：2022-09-17
@author：Liang XiaoZhi
@describe：机舱控制柜温度异常算法部署
"""

import warnings
import datetime
import pymongo
import re
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import fnmatch
from scipy.interpolate import interp1d
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

t = "time"
cct = "control_carbin_temp"

# def ensemble_predict(data, model):
#     pred_1, pred_2, pred_3 = model[0].predict(data), model[1].predict(data), model[2].predict(data)
#     ensemble = (pred_1 + pred_2 + pred_3) / 3
#     return ensemble


def compute_health(base, pred):
    import math
    from scipy.stats import ks_2samp
    k = ks_2samp(base, pred)[0]
#     return np.exp(-math.e*k)
    return float(np.max([1-(math.e)**(math.e*2*(k-1)), 1+np.log10(1-k**2)]))


def distance_transform(value, x, y):
    lenx = len(x)
    if value < x[0]:
        value_t = 100
    elif (value >= x[0]) & (value < x[lenx - 1]):
        itpl = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        value_t = float(itpl(value))
    else:
        value_t = 60.2
    return value_t


def distance2alarm(distance, data, th=50, rate=0.1):
    if distance >= 80:
        alarm = 0
    elif 60 <= distance < 80:
        alarm = 1
    elif 40 <= distance < 60:
        alarm = 2
    else:
        alarm = 3
    alarm, new_distance = model_post_processing(data, alarm, distance, th=th, rate=rate)
    return alarm, new_distance


def model_post_processing(data, alarm, distance, th=50, rate=0.1):
    data = np.array(data)
    S = len(data)
    A = len(np.where(data >= th)[0])
    new_distance = 0
    if alarm > 0:
        if A/S >= rate:
            alarm = alarm
            new_distance = distance
        else:
            new_distance = distance_transform(A/S, [0, 0.5*rate, rate, 1.5*rate], [100, 90, 79, 61])
            alarm = 0
    else:
        if A / S >= rate:
            alarm = 1
            new_distance = distance_transform(A/S, [0, 0.5*rate, rate, 1.5*rate], [100, 90, 79, 61])
        else:
            alarm = 0
            new_distance = distance

    return int(alarm), new_distance


def ControlCabinetTempAbnormal_main(data, residual_path, model_path, feature_path, th, rate):
    # print(">>> 原始数据维度：", data.shape)
    # df = data[(data['main_status'] == 37) | (data['main_status'] == 38)]

    df = data
    df[t] = pd.to_datetime(df[t])
    print(feature_path)
    # regex = re.compile(r"\[|\]|<", re.IGNORECASE)

    # df = df.dropna()
    # df = df.reset_index(drop=True)
    # df = df.sort_values(t)

    df = df.sort_values(t)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.reset_index(drop=True)
    df = df.sort_values(t)

    result = {}
    # load
    with open(residual_path, 'rb') as file:  # 残差序列（训练阶段保存的）
        residual = pickle.load(file)

    result['analysis_data'] = {}
    result['analysis_data']['benchmark_res'] = residual.tolist()
    plt.clf()
    ax1 = sns.kdeplot(residual)
    result['analysis_data']['benchmark_x'] = list(ax1.lines[0].get_xdata())
    result['analysis_data']['benchmark_y'] = list(ax1.lines[0].get_ydata())
    plt.close()

    with open(model_path, 'rb') as file:
        mod_1 = pickle.load(file)
    # with open(model2_path, 'rb') as file:
    #     mod_2 = pickle.load(file)
    # with open(model3_path, 'rb') as file:
    #     mod_3 = pickle.load(file)
    with open(feature_path, 'rb') as f:
        fea = pickle.load(f)

    #####增加上个时刻温度 jimmy-20221109
    df['cct_shift']= df[cct].shift(1)
    df= df.dropna()

    pred = mod_1.predict(df[fea].values)
    res2 = df[cct] - pred

    if len(df) >= 0.6 * len(data):
        # result['raw_data'] = {'date': [str(x) for x in df[t].values], 'temp': df[cct].values.tolist(),
        #                       'pred': pred.tolist()}

        result['raw_data'] = {'date': df[t], 'temp': df[cct].values.tolist(),
                              'pred': pred.tolist()}
        # plt.clf()
        # plt.plot(res2, label="原始")
        # plt.plot(residual, label="预测")
        # plt.legend()
        # plt.show()

        plt.clf()
        ax2 = sns.kdeplot(res2)
        result['analysis_data']['online_x'] = list(ax2.lines[0].get_xdata())
        result['analysis_data']['online_y'] = list(ax2.lines[0].get_ydata())
        plt.close()

        result['analysis_data']['online_res'] = res2.tolist()

        result['status_code'] = '000'
        result['start_time'] = str(pd.to_datetime(df[t].values)[0])
        result['end_time'] = str(pd.to_datetime(df[t].values)[-1])
        result['distance'] = compute_health(residual, res2) * 100
        ####做了修改
        result['alarm'], result['distance'] = distance2alarm(result['distance'], result['raw_data']['temp'], th, rate)
        # 降采样
        idx = [(i - 1) * 3 for i in range(1, int(len(df[cct].values.tolist()) / 3))]
        idx[-1] = idx[-1] - 1
        result['raw_data']['date'] = list(np.array(result['raw_data']['date'])[idx])

        result['raw_data']['temp'] = list(np.array(result['raw_data']['temp'])[idx])
        result['raw_data']['pred'] = list(np.array(result['raw_data']['pred'])[idx])
        result['analysis_data']['online_res'] = list(np.array(result['analysis_data']['online_res'])[idx])

    elif 0 < len(df) < 0.6 * len(data):
        result['raw_data'] = {'date': [str(x) for x in df[t].values], 'temp': df[cct].values.tolist(),
                              'pred': pred.tolist()}
        plt.clf()
        ax2 = sns.kdeplot(res2)
        result['analysis_data']['online_x'] = ax2.lines[0].get_xdata()
        result['analysis_data']['online_y'] = ax2.lines[0].get_ydata()
        plt.close()

        result['analysis_data']['online_res'] = res2.tolist()

        result['status_code'] = '100'
        result['start_time'] = str(pd.to_datetime(df[t].values)[0])
        result['end_time'] = str(pd.to_datetime(df[t].values)[-1])
        result['distance'] = compute_health(residual, res2) * 100
        result['alarm'], result['distance'] = distance2alarm(result['distance'], result['raw_data']['temp'], th, rate)
        # 降采样
        idx = [(i - 1) * 3 for i in range(1, int(len(df[cct].values.tolist()) / 3))]
        idx[-1] = idx[-1] - 1
        result['raw_data']['date'] = list(np.array(result['raw_data']['date'])[idx])
        result['raw_data']['temp'] = list(np.array(result['raw_data']['temp'])[idx])
        result['raw_data']['pred'] = list(np.array(result['raw_data']['pred'])[idx])
        result['analysis_data']['online_res'] = list(np.array(result['analysis_data']['online_res'])[idx])
    else:
        result['raw_data'] = {'date': [str(x) for x in data[t].values], 'temp': data[cct].values.tolist(),
                              'pred': None}
        result['analysis_data']['online_x'] = None
        result['analysis_data']['online_y'] = None
        result['analysis_data']['online_res'] = None
        result['status_code'] = '200'
        result['start_time'] = str(pd.to_datetime(data[t].values)[0])
        result['end_time'] = str(pd.to_datetime(data[t].values)[-1])
        result['distance'] = None
        result['alarm'] = None

        # 降采样
        idx = [(i - 1) * 3 for i in range(1, int(len(df[cct].values.tolist()) / 3))]
        idx[-1] = idx[-1] - 1
        result['raw_data']['date'] = list(np.array(result['raw_data']['date'])[idx])
        result['raw_data']['temp'] = list(np.array(result['raw_data']['temp'])[idx])

    # plt.clf()
    # # plt.title("{}-{} alarm：{}".format(st, et, result['alarm']))
    # plt.plot(result['raw_data']['temp'], label="truth")
    # plt.plot(result['raw_data']['pred'], label="pred")
    # plt.legend()
    # plt.xlabel('time')
    # plt.ylabel('cct')
    # # plt.savefig("./tmp/{}_{}-{}_测试效果.png".format(data['wt_id'].values[0], st, et))
    # plt.show()


    start_time = result['start_time']
    end_time = result['end_time']
    raw_data = result['raw_data']
    analysis_data = result['analysis_data']
    status_code = result['status_code']
    distance = result['distance']
    alarm = result['alarm']
    return start_time, end_time, raw_data, analysis_data, status_code, distance, alarm


# if __name__ == '__main__':
#     data1 = pd.read_csv('../W010_control_06.csv')
#     data1['dataTimeStamp'] = data1['t'].apply(lambda x: x)
#     residual_path1 = '../Resource/{}/Residual.pkl'.format('W010')
#     model_path1 = '../Resource/{}/Mod1.pkl'.format('W010')
#     feature_path1 = '../Resource/{}/Feature.pkl'.format('W010')
#     th1 = 50
#     rate1 = 0.1
#
#     res = ControlCabinetTempAbnormal_main(data1, residual_path1, model_path1, feature_path1, th1, rate1)
#     print(res[-2:])


