#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time :2021/9/26 10:00
# @author : K Mark, Ziqi, T Bao, R Xie
# @version : 2.0.0
import pandas as pd
import numpy as np
from decimal import *
import xgboost as xgb
from scipy.interpolate import interp1d
import pickle
import math
from collections import Counter
import pymongo
import yaml
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt




ts = 'time'
gs = 'generator_rotating_speed'
gp = 'power'
temp_bear_dr = 'f_i_bearing_temp'
temp_bear_ndr = 'r_i_bearing_temp'
temp_cab = 'cabin_temp'
ws = 'wind_speed'
yaw = 'yaw_error'
ba = 'pitch_1'
has_wtstatus = 'False'
wtstatus = 'main_status'  # 机组状态变量名
wtstatus_n = 0  # 正常发电状态代码
model_type = 'A'



wtid = 'turbineName'
maxgp = 3200  # 额定功率
threshold_01 = 0.6
threshold_00 = 0.9
threshold_1 = 1.0
n_lookback_alarm = 100
n_lookback_warning = 10
n_continuous = 3
border = 0.1



# ws_gp_model_p = 'Resource/W056_ws_gp.model'  # rs_gp_model路径
# model_dr_path = 'Resource/dr/WTW056_model.bin'  #使用发电机轴承温度训练结果
# model_ndr_path = 'Resource/ndr/WTW056_model.bin'
# threshold_path = 'Resource/thresholds.bin'
# train_temp_range_path = 'Resource/${wt_id}_env_range.bin' # 使用风向仪训练结果！
train_temp_range = dict()
train_temp_range['max_train_temp'] = 80
train_temp_range['min_train_temp'] = -10

# bpc_file_path = 'Resource/pc.csv'  #功率曲线，用于前端显示作为对比基线

def get_warning_2(res_dr, res_ndr, threshold_01, threshold_00, threshold_1, border):
    """
    param res_dr: 驱动端温度残差序列
    param res_ndr: 非驱动端温度残差序列
    param threshold1: 温度残差预警阈值
    param threshold2: 残差偏移预警阈值
    param border: 判断为异常的比例边界，默认为0.2
    return: 预警值，1表示触发预警
    """
    # warning_dr = 0
    # warning_ndr = 0
    # res_dp = res_dr - res_ndr
    # res_dn = res_ndr - res_dr
    # index_dr_1 = [i for i, e in enumerate(res_dr) if threshold_01 <= abs(e) < threshold_00]
    # index_dr_2 = [i for i, e in enumerate(res_dp) if abs(e) >= threshold_1]
    # index_ndr_1 = [i for i, e in enumerate(res_ndr) if threshold_01 <= abs(e) < threshold_00]
    # index_ndr_2 = [i for i, e in enumerate(res_dn) if abs(e) >= threshold_1]
    # count_index_dr = [x for x in index_dr_1 if x in index_dr_2]
    # count_index_ndr = [x for x in index_ndr_1 if x in index_ndr_2]
    # if len(count_index_dr) / len(res_dr) > border or np.sum(res_dr >= threshold_00) / len(res_dr) > border:
    #     warning_dr = 1
    # print(np.sum(res_ndr >= threshold_00) / len(res_ndr))
    # if len(count_index_ndr) / len(res_ndr) > border or np.sum(res_ndr >= threshold_00) / len(res_ndr) > border:
    #     print ('ndr_1111')
    #     warning_ndr = 1
    # return warning_dr, warning_ndr

    warning_dr = 0
    warning_ndr = 0
    res_dp = res_dr - res_ndr
    res_dn = res_ndr - res_dr
    index_dr_1 = [i for i, e in enumerate(res_dr) if threshold_01 <= abs(e) < threshold_00]   ###########此处应为绝对值  Jimmy_20211125
    index_ndr_1 = [i for i, e in enumerate(res_ndr) if threshold_01 <= abs(e) < threshold_00]

    index_dr_2 = [i for i, e in enumerate(res_dp) if abs(e) >= threshold_1]
    index_ndr_2 = [i for i, e in enumerate(res_dn) if abs(e) >= threshold_1]

    count_index_dr = [x for x in index_dr_1 if x in index_dr_2]
    count_index_ndr = [x for x in index_ndr_1 if x in index_ndr_2]

    if len(count_index_dr) / len(res_dr) > border or np.sum(abs(res_dr) >= threshold_00) / len(res_dr) > border:  ###【（threshold_01 <= res_dr < threshold_00）且 （res_dp >= threshold_1）】或 【res_dr >= threshold_00】 原来有错误
        warning_dr = 1
    # print(np.sum(abs(res_ndr) >= threshold_00) / len(res_ndr))
    if len(count_index_ndr) / len(res_ndr) > border or np.sum(abs(res_ndr) >= threshold_00) / len(res_ndr) > border:
        # print('ndr_1111')
        warning_ndr = 1
    return warning_dr, warning_ndr


def get_alarm_level(alarm_history, alarm, n_lookback_alarm):
    """
    get_alarm的子函数，根据历史报警积累确定相应的报警等级
    param alarm_series: 历史的报警序列
    param alarm: 当前时刻的报警状态
    return: 报警等级
    """
    alarm_history_tmp = alarm_history[-n_lookback_alarm:]
    # print (alarm_history)
    # distance = min(100 * len([x for x in alarm_history_tmp if x != 0]) / n_lookback_alarm + 40, 100)
    # distance_h = 100 - distance
    # if alarm == 1:  #如果alarm=1才去算距离，如果alarm=0 ，距离就是100， 这个逻辑不太好
    #     if 40 < distance_h <= 60:
    #         alarm_level = 1
    #     elif 20 < distance_h <= 40:
    #         alarm_level = 2
    #     elif distance_h <= 20:
    #         alarm_level = 3

    distance = min(100 * len([x for x in alarm_history_tmp if x != 0]) / n_lookback_alarm, 100)
    distance_h = 100 - distance
    if alarm == 1:  # 如果alarm=1才去算距离，如果alarm=0 ，距离就是100， 这个逻辑不太好
        # alarm_level = 0
        if 80 < distance_h <= 100:
            alarm_level = 1
        elif 60 < distance_h <= 80:
            alarm_level = 2
        elif distance_h <= 60:
            alarm_level = 3
    else:
        distance_h = 100
        alarm_level = 0
    return float(distance_h), int(alarm_level)


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
    final_level_dr = get_alarm_level(alarm_dr_history, alarm_dr, n_lookback_alarm)
    final_level_ndr = get_alarm_level(alarm_ndr_history, alarm_ndr, n_lookback_alarm)
    return final_level_dr, final_level_ndr


def data_preprocess(data):
    data = data.sort_values(by = ts)
    data['speed'] = data[gs] * math.pi * 2 / 60
    data['espeed'] = (data[gs] * 2 * math.pi / 60) ** 1.6
    data['power'] = abs(data[gp]) * 1000
    data['y_dr'] = data[temp_bear_dr] - 0.5 *(data[temp_cab] + data[temp_cab].shift(1))
    data['y_ndr'] = data[temp_bear_ndr] - 0.5 *(data[temp_cab] + data[temp_cab].shift(1))
    data['temp_cab_mean'] = 0.5 *(data[temp_cab] + data[temp_cab].shift(1))
    data['X1_dr'] = data[temp_bear_dr].shift(1) - 0.5 *(data[temp_cab] + data[temp_cab].shift(1))
    data['X1_ndr'] = data[temp_bear_ndr].shift(1) - 0.5 *(data[temp_cab] + data[temp_cab].shift(1))
    data['X2'] = 0.5 * (data['speed'] + data['speed'].shift(1))
    data['X3'] = 0.5 * (data['espeed'] + data['espeed'].shift(1))
    data['X4'] = 0.5 * (data['power'] + data['power'].shift(1))
    data = data.dropna()
    data.reset_index(inplace=True)
    return data


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


def import_data_check(data):
    check_result = dict()
    check_result['response'] = True
    check_result['status_code'] = '000'
    if data.isnull().all().any():  # 判断数据中某列是否全部为空值
        check_result['response'] = False
        check_result['status_code'] = '200'


    elif data.isnull().any().any():
        for i in data.columns:  # 判断各列缺失值的比例是否大于50%
            missing_value_counts = len(data[i]) - data[i].count()
            missing_value_rate = (missing_value_counts / len(data[i])) * 100
            if missing_value_rate >= 50:
                check_result['response'] = False
                check_result['status_code'] = '201'

            elif (missing_value_rate >= 5) & (missing_value_rate < 50) :
                check_result['response'] = False
                check_result['status_code'] = '100'

    return check_result





def gta(data,warning_dr_history,warning_ndr_history,alarm_dr_history,alarm_ndr_history, model_dr, model_ndr):
    # Read in some external files
    # with open(model_dr_path, 'rb') as fp:
    #     model_dr = pickle.load(fp)
    # with open(model_ndr_path, 'rb') as fp:
    #     model_ndr = pickle.load(fp)


    status_code = import_data_check(data)['status_code']
    # raise Exception('status_code is after data  Check  ' + status_code + 'data Length is ' + str(len(data))  + 'afterpreprocess is ' + str(len (data_preprocess(data))))

    data_new = data_preprocess(data)

    # data_new.to_csv('stuck.csv')


    raw_data = {}
    analysis_data = {}
    result = {}
    if len(data_new)>0:
        if status_code == '000' or status_code == '100':

            dr_hat = model_dr.predict(data_new.loc[:, ['X1_dr', 'X2', 'X3', 'X4']])
            ndr_hat = model_ndr.predict(data_new.loc[:, ['X1_ndr', 'X2', 'X3', 'X4']])
            res_dr = data_new['y_dr'].values.reshape(-1, 1) - dr_hat
            res_ndr = data_new['y_ndr'].values.reshape(-1, 1) - ndr_hat
            res_d = res_dr - res_ndr

            # plt.plot(data_new['y_ndr'].values.reshape(-1, 1)  )
            # plt.plot(ndr_hat)
            # plt.show()

            # print('np.mean(res_ndr)',np.mean(res_ndr) )

            warning = get_warning_2(res_dr, res_ndr, threshold_01, threshold_00, threshold_1, border)
            warning_dr_history.append(warning[0])
            warning_ndr_history.append(warning[1])

            alarm = get_alarm(warning_dr_history, warning_ndr_history, alarm_dr_history, alarm_ndr_history,
                              n_lookback_alarm, n_lookback_warning, n_continuous)

            alarm_dr, alarm_ndr = alarm[0], alarm[1]
            alarm_dr_history.append(alarm_dr[1])
            alarm_ndr_history.append(alarm_ndr[1])
            alarm_final = 0
            if np.any([alarm_dr[1], alarm_ndr[1]]) != 0:
                alarm_final = 1

            # Construct outputs
            raw_data['datetime'] = list(map(str, data[ts].tolist()))
            raw_data['gbt1'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[temp_bear_dr].values.tolist()))
            raw_data['gbt2'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[temp_bear_ndr].values.tolist()))
            raw_data['rs'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[gs].values.tolist()))
            raw_data['gp'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[gp].values.tolist()))
            raw_data['gbt1_prediction'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),  [np.nan] + (dr_hat[:, 0] + data_new['temp_cab_mean'].values).tolist()))
            raw_data['gbt2_prediction'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),  [np.nan] + (ndr_hat[:, 0] + data_new['temp_cab_mean'].values).tolist()))

            analysis_data['datetime'] = list(map(str, data[ts].tolist()))
            analysis_data['gbt1_residual'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), [np.nan] + res_dr[:, 0].tolist()))
            analysis_data['gbt2_residual'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), [np.nan] + res_ndr[:, 0].tolist()))
            analysis_data['gbt1_residual_average'] = float(Decimal(np.mean(res_dr)).quantize(Decimal('0.0000')))
            analysis_data['gbt2_residual_average'] = float(Decimal(np.mean(res_ndr)).quantize(Decimal('0.0000')))
            analysis_data['res_d'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), [np.nan] + res_d[:, 0].tolist()))
            analysis_data['warning_dr'] = warning[0]  # Append this to warning_dr_history
            analysis_data['warning_ndr'] = warning[1]  # Append this to warning_ndr_history
            analysis_data['gbt_residual_threshold'] = [threshold_00, threshold_01]
            analysis_data['res_d_threshold'] = threshold_1

            result['raw_data'] = raw_data
            result['analysis_data'] = analysis_data
            result['status_code'] = status_code
            result['start_time'] = str(data_new[ts].iloc[0])
            result['end_time'] = str(data_new[ts].iloc[-1])
            result['gbt1_distance'] = alarm_dr[0]
            result['gbt2_distance'] = alarm_ndr[0]
            result['gbt1_alarm'] = alarm_dr[1]  # Append this to alarm_dr_history
            result['gbt2_alarm'] = alarm_ndr[1]  # Append this to alarm_ndr_history
            result['distance'] = np.max([alarm_dr[0], alarm_ndr[0]])
            result['alarm'] = alarm_final
        else:
            raw_data['datetime'] = data_new[ts].tolist()
            raw_data['gbt1'] = data_new[temp_bear_dr].values.tolist()
            raw_data['gbt2'] = data_new[temp_bear_dr].values.tolist()
            raw_data['rs'] = data_new[gs].values.tolist()
            raw_data['gp'] = data_new[gp].values.tolist()

            result['raw_data'] = raw_data
            result['analysis_data'] = analysis_data
            result['status_code'] = status_code
            result['start_time'] = str(data_new[ts].iloc[0])
            result['end_time'] = str(data_new[ts].iloc[-1])
            result['gbt1_distance'] = None
            result['gbt2_distance'] = None
            result['gbt1_alarm'] = None
            result['gbt2_alarm'] = None
            result['distance'] = None
            result['alarm'] = None
    else:
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['status_code'] = '301'
        result['start_time'] = None
        result['end_time'] = None
        result['gbt1_distance'] = None
        result['gbt2_distance'] = None
        result['gbt1_alarm'] = None
        result['gbt2_alarm'] = None
        result['distance'] = None
        result['alarm'] = None
    return result


def window_mean(data, window, overlap):
    i = 1
    result = []
    while i < len(data):
        sdata = data[i:(i + window)]
        mean_window = np.mean(sdata)
        result.append(mean_window)
        i = i + overlap
    return result


def distance_transform(value, x, y):
    if len(x) == len(y):
        itpl = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        value_t = float(itpl(value))
        if value_t > max(y):
            value_t = max(y)
        elif value_t < min(y):
            value_t = min(y)
        else:
            value_t = value_t
        return value_t
    else:
        raise Exception

##这里改了一下
def rank_transform(distance):
    if 80 < distance:
        rank = 0
    elif 60 < distance <= 80:
        rank = 1
    elif 40 < distance <= 60:
        rank = 2
    elif distance <= 40:
        rank = 3
    else:
        rank = np.nan
    return rank



def generator_bearingstuck(data,result_history, model_dr, model_ndr, ws_gp_model):


    columns = [ts, gs, gp, temp_bear_dr, temp_bear_ndr, temp_cab, ws, ba, wtstatus]

    data = data[columns]

    warning_dr_history = []
    warning_ndr_history = []
    alarm_dr_history = []
    alarm_ndr_history = []

    if len(result_history) < n_lookback_alarm:
        for i in range( n_lookback_alarm - len(result_history) ):
            warning_dr_history.append(0)
            warning_ndr_history.append(0)
            alarm_dr_history.append(0)
            alarm_ndr_history.append(0)

    for i in range(len(result_history)):
        if 'analysis_data' in result_history.columns:
            if (type(result_history.loc[i, 'analysis_data']) == dict):
                warning_dr_history.append(result_history.loc[i, 'analysis_data']['warning_dr'])
                warning_ndr_history.append(result_history.loc[i, 'analysis_data']['warning_ndr'])
            else:
                warning_dr_history.append(float('NaN'))
                warning_ndr_history.append(float('NaN'))
        else:
            warning_dr_history.append(float('NaN'))
            warning_ndr_history.append(float('NaN'))

        if 'gbt1_alarm' in result_history.columns:
            alarm_dr_history.append(result_history.loc[i, 'gbt1_alarm'])
        else:
            alarm_dr_history.append(float('NaN'))

        if 'gbt2_alarm' in result_history.columns:
            alarm_ndr_history.append(result_history.loc[i, 'gbt2_alarm'])
        else:
            alarm_ndr_history.append(float('NaN'))


    if len(data) > 0:
        # data = data.dropna(subset=[ws, gp, rs, temp_bear_dr, temp_bear_ndr, temp_cab, dt, wtstatus])
        # data.index = np.arange(0, len(data), 1)
        if len(data) > 0:
            # with open(threshold_path, 'rb') as f:  #功率预测的标准差，意义不大，写成百分比好了。 Jimmy-20211115
            #     thresholds = pickle.load(f)
            #     f.close()

            result = dict()
            rawd = data.copy()
            gbta_result = gta(data, warning_dr_history, warning_ndr_history, alarm_dr_history, alarm_ndr_history, model_dr, model_ndr)
            # print('GBTA SC:', gbta_result['status_code'])
            raw_data = gbta_result['raw_data']
            analysis_data = gbta_result['analysis_data']
            status_code = gbta_result['status_code']
            gbt1_alarm = gbta_result['gbt1_alarm']
            gbt2_alarm = gbta_result['gbt2_alarm']

#####################################################以上先获得发电机轴承报警结果

            raw_data['bpc_ws'] =  pd.Series(np.arange(3,20,0.5))
            raw_data['bpc_gp'] = pd.Series(ws_gp_model.predict(raw_data['bpc_ws'].values.reshape(-1, 1)))

            # plt.plot(raw_data['bpc_ws'], raw_data['bpc_gp'])
            # plt.show()


            #
            # bpc = pd.read_csv(bpc_file_path)   #用于界面前端显示
            # raw_data['bpc_ws'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.00'))), bpc['WSbin'].values.tolist()))
            # raw_data['bpc_gp'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.00'))), bpc['GP'].values.tolist()))

            if gbta_result['status_code'] == '000' or gbta_result['status_code'] == '100' or gbta_result['status_code'] == '200':
                if has_wtstatus == 'True':
                    data = data.loc[(data[ws] > 0) & (data[wtstatus] == wtstatus_n), :]
                if ba in data.columns:
                    data = data.loc[
                           (data[gs] > 0) & (data[ws] > 0) & (data[gp] > 10) & (
                               ((data[gp] < maxgp * 0.95) & (data[ba] < 3)) | (
                                   (data[gp] > maxgp * 0.95) & (data[ba] < 90))), :]
                else:
                    data = data.loc[(data[gs] > 0) & (data[ws] > 0), :]

                if len(data) > 0:
                    # dropped_index = [v for v in rawd.index if v not in data.index]


                    # with open(ws_gp_model_p, 'rb') as f:    ####直接用结冰算法的功率曲线训练文件好了，特征只有风速 ——Jimmy 20211115
                    #     ws_gp_model = pickle.load(f)
                    #     f.close()

                    ######温度训练结果，就是正常运行的机舱温度范围，目前没有完整的一年运行数据，这一项先不用了,改成固定值 -20210927-Jimmy
                    # with open(train_temp_range_path, 'rb') as f:
                    #     train_temp_range = pickle.load(f)
                    #     f.close()


                    # if model_type == 'A':
                    #     vlist = [ws, yaw, ba]
                    # elif model_type == 'B':
                    #     vlist = [ws, yaw]
                    # else:
                    #     vlist = [ws, ba]

                    vlist = [ws]
                    x_data = data.loc[:, vlist].values.reshape(-1, len(vlist))
                    data['pred_gp'] = ws_gp_model.predict(x_data)   #用model文件预测
                    data['diff_gp'] = data[gp] - data['pred_gp']

                    data['decrease_percent_gp'] = data[gp]/data['pred_gp']-1.0

                    # smooth_diff_gp = window_mean(data['diff_gp'], 100, 50)  #总共144组数据，这里的滑窗没什么意义 20210927-jimmy
                    #
                    # plt.scatter(data[ws], data[gp],label= 'real')
                    # plt.scatter(data[ws], data['pred_gp'],label= 'predict')
                    # plt.legend()
                    # plt.show()
                    # plt.close()


                    # smooth_diff_gp = np.mean(data['diff_gp'])
                    thresholds={}
                    thresholds['label_threshold'] = -0.05

                    if (np.mean(data['diff_gp']) < 0) & (gbta_result['alarm'] >=1) & (len(data)>72):
                        distance = 100 - float(Decimal(distance_transform(abs(np.mean(data['decrease_percent_gp'])),   list(map(abs, [0, -0.1,-0.2,-0.3,-0.5])),   [0, 20, 40, 60, 100])).quantize(  Decimal('0.00')))


                        # distance = 100 - float( Decimal(distance_transform(abs(np.mean(data['decrease_gp'])),
                        #                                list(map(abs, [0, thresholds['mean'] - 2 * thresholds['std'],
                        #                                               thresholds['mean'] - 3 * thresholds['std'],
                        #                                               thresholds['mean'] - 4 * thresholds['std'],
                        #                                               thresholds['mean'] - 10 * thresholds['std']])),
                        #                                [0, 40, 60, 80, 100])).quantize(Decimal('0.00')))


                        # import matplotlib.pyplot as plt
                        #
                        # plt.subplot(2,2,1)
                        # plt.scatter(data[ws], data[gp], s=0.5,label= 'realPower')
                        # plt.scatter(data[ws],  data['pred_gp'], s=0.5, label= 'pred')
                        # plt.scatter(data[ws], data['diff_gp'] ,s=0.5, label= 'diff' )
                        # plt.legend()
                        #
                        # plt.subplot(2, 2, 2)
                        # plt.scatter(data[ws], data[rs], s=0.5, label='rotorSpeed')
                        # plt.legend()
                        #
                        # plt.subplot(2, 2, 3)
                        # plt.scatter(data[ws], data[ba], s=0.5, label= 'pitchAngle')
                        # plt.legend()
                        #
                        # plt.subplot(2, 2, 4)
                        # plt.scatter(data[ws], data[yaw], s=0.5, label= 'yaw')
                        # plt.legend()
                        #
                        # plt.show()
                        #
                        # print (abs(np.mean(data['diff_gp'])), 'distance=', distance)

                    else:
                        distance = 100

                    #机舱温度在正常范围内的比例
                    ratio = len(data.loc[(data[temp_cab] < train_temp_range['max_train_temp'])  & (data[temp_cab] > train_temp_range['min_train_temp'])]) / len(data)

                    rawd['pred_gp'] = np.nan
                    rawd.loc[data.index, 'pred_gp'] = data['pred_gp'].values
                    rawd['diff_gp'] = np.nan
                    rawd.loc[data.index, 'diff_gp'] = data['diff_gp'].values

                    rawd['decrease_percent_gp'] = data['decrease_percent_gp']
                    # rawd['gp_0'] = np.nan
                    # rawd.loc[rawd['diff_gp'] >= thresholds['label_threshold'], 'gp_0'] = rawd.loc[rawd['diff_gp'] >= thresholds['label_threshold'], gp]
                    # rawd['gp_1'] = np.nan
                    # rawd.loc[rawd['diff_gp'] < thresholds['label_threshold'], 'gp_1'] = rawd.loc[  rawd['diff_gp'] < thresholds['label_threshold'], gp]
                    # rawd['diff_gp0'] = np.nan
                    #
                    # rawd.loc[(rawd[temp_cab] >= train_temp_range['min_train_temp'])
                    #          & ( rawd[temp_cab] <= train_temp_range['max_train_temp']), 'diff_gp0'] \
                    #     = rawd.loc[(rawd[temp_cab] >= train_temp_range['min_train_temp'])
                    #                & ( rawd[temp_cab] <= train_temp_range['max_train_temp']), 'diff_gp']
                    # rawd['diff_gp1'] = np.nan
                    #
                    # rawd.loc[(rawd[temp_cab] < train_temp_range['min_train_temp'])
                    #          & (rawd[temp_cab] > train_temp_range['max_train_temp']), 'diff_gp1'] \
                    #     = rawd.loc[(rawd[temp_cab] < train_temp_range['min_train_temp'])
                    #                & ( rawd[temp_cab] > train_temp_range['max_train_temp']), 'diff_gp']
                    # rawd['gp_2'] = np.nan
                    #
                    # rawd.loc[rawd['diff_gp'] < thresholds['label_threshold'], 'gp_2'] \
                    #     = rawd.loc[rawd['diff_gp'] < thresholds['label_threshold'], 'pred_gp']

                    rawd['gp_0'] = np.nan
                    rawd.loc[rawd['decrease_percent_gp'] >= thresholds['label_threshold'], 'gp_0'] = rawd.loc[ rawd['decrease_percent_gp'] >= thresholds['label_threshold'], gp]
                    rawd['gp_1'] = np.nan
                    rawd.loc[rawd['decrease_percent_gp'] < thresholds['label_threshold'], 'gp_1'] = rawd.loc[ rawd['decrease_percent_gp'] < thresholds['label_threshold'], gp]
                    rawd['gp_2'] = np.nan
                    rawd.loc[rawd['decrease_percent_gp'] < thresholds['label_threshold'], 'gp_2'] = rawd.loc[rawd['decrease_percent_gp'] < thresholds['label_threshold'], 'pred_gp']

                    ############diff_gp0, diff_gp1 按温度分
                    rawd['diff_gp0'] = np.nan
                    rawd.loc[(rawd[temp_cab] >= train_temp_range['min_train_temp'])  & (rawd[temp_cab] <= train_temp_range['max_train_temp']), 'diff_gp0'] \
                        = rawd.loc[(rawd[temp_cab] >= train_temp_range['min_train_temp'])  & (rawd[temp_cab] <= train_temp_range['max_train_temp']), 'diff_gp']
                    rawd['diff_gp1'] = np.nan
                    rawd.loc[(rawd[temp_cab] < train_temp_range['min_train_temp'])  & (rawd[temp_cab] > train_temp_range['max_train_temp']), 'diff_gp1'] \
                        = rawd.loc[(rawd[temp_cab] < train_temp_range['min_train_temp']) & (rawd[temp_cab] > train_temp_range['max_train_temp']), 'diff_gp']



                    if ratio < 0.33:
                        status_code = '100'  # 输入数据工况符合训练时工况（#机舱温度在正常范围内）比例小于1/3
                        distance = None
                        alarm = None
                    else:
                        alarm = rank_transform(distance)

                    raw_data['ws'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd[ws].values.tolist()))
                    raw_data['gp_0'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd['gp_0'].values.tolist()))
                    raw_data['gp_1'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd['gp_1'].values.tolist()))
                    raw_data['gp_2'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd['gp_2'].values.tolist()))
                    raw_data['pred_gp'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd['pred_gp'].values.tolist()))
                    analysis_data['gp_residual'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd['diff_gp0'].values.tolist()))
                    analysis_data['gp_residual_out'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd['diff_gp1'].values.tolist()))
                    analysis_data['gp_residual_average'] = float(Decimal(np.nanmean(  [np.nanmean(rawd['diff_gp0'].values), np.nanmean(rawd['diff_gp1'].values)])).quantize( Decimal('0.0000')))
                    analysis_data['gp_threshold'] = [0.5* thresholds['label_threshold'],
                                                     thresholds['label_threshold'],
                                                     2* thresholds['label_threshold']]

                    result['start_time'] = gbta_result['start_time']
                    result['end_time'] = gbta_result['end_time']
                    result['distance'] = distance
                    result['alarm'] = alarm
                    result['raw_data'] = raw_data
                    result['analysis_data'] = analysis_data
                    result['status_code'] = status_code
                    result['gbt1_alarm'] = gbt1_alarm
                    result['gbt2_alarm'] = gbt2_alarm

                else:
                    # raise Exception('the status_code is ' + gbta_result['status_code'])

                    status_code = '101'  # 发电机轴承温度异常检测模块无正常返回
                    result['start_time'] = gbta_result['start_time']
                    result['end_time'] = gbta_result['end_time']
                    result['alarm'] = None
                    result['raw_data'] = None
                    result['analysis_data'] = None
                    result['distance'] = None
                    result['status_code'] = status_code
                    result['gbt1_alarm'] = None
                    result['gbt2_alarm'] = None

            else:
                # raise Exception('the status_code is ' + gbta_result['status_code'] )

                status_code = '101'  # 发电机轴承温度异常检测模块无正常返回
                result['start_time'] = gbta_result['start_time']
                result['end_time'] = gbta_result['end_time']
                result['alarm'] = None
                result['raw_data'] = None
                result['analysis_data'] = None
                result['distance'] = None
                result['status_code'] = status_code
                result['gbt1_alarm'] = None
                result['gbt2_alarm'] = None
        else:
            status_code = '200'
            result = dict()
            result['start_time'] = None
            result['end_time'] = None
            result['alarm'] = None
            result['raw_data'] = None
            result['analysis_data'] = None
            result['distance'] = None
            result['status_code'] = status_code
            result['gbt1_alarm'] = None
            result['gbt2_alarm'] = None
    else:
        status_code = '200'
        result = dict()
        result['start_time'] = None
        result['end_time'] = None
        result['alarm'] = None
        result['raw_data'] = None
        result['analysis_data'] = None
        result['distance'] = None
        result['status_code'] = status_code
        result['gbt1_alarm'] = None
        result['gbt2_alarm'] = None

    return  result['raw_data'],result['analysis_data'] ,result['status_code'],result['start_time'],result['end_time'],  result['alarm'], result['distance'],  result['gbt1_alarm'],  result['gbt2_alarm']



def generator_bs_main(data, result_history, model_dr, model_ndr, ws_gp_model ):
    """
    发电机轴承卡滞异常服务入口
    :param data: 数据集，dataframe
    :return:
    """
    global ws_gp_model_p
    global maxgp
    global temp_cab
    global ts
    global temp_bear_dr
    global temp_bear_ndr
    global ws
    global gp
    global gs
    global ba
    global yaw
    global has_wtstatus
    global wtstatus
    global wtstatus_n
    global model_type
    global train_temp_range_path
    # global bpc_file_path
    global threshold_path
    global threshold_01
    global threshold_00
    global threshold_1
    global n_lookback_alarm
    global n_lookback_warning
    global n_continuous
    global model_dr_path
    global model_ndr_path
    global buffer_config_fp


    return generator_bearingstuck(data,result_history, model_dr, model_ndr, ws_gp_model)


if __name__ == '__main__':
    import os
    path = r'C:\Project\07_DaTang_BladeProtector\Algorithm\05_发电机轴承卡滞\Data\PreP_10min\W056_10min_test.csv'
    data = pd.read_csv(path, engine='python', parse_dates=['time'], infer_datetime_format=True)

    start_time = '2021-11-14 00:00:00'
    # warning_dr_history=[]
    # warning_ndr_history=[]
    # alarm_dr_history=[]
    # alarm_ndr_history=[]

    result_history = pd.DataFrame()

    path_result_history = r'D:\Users\Administrator\Downloads\GeneratorBearingStuck-56_2\GeneratorBearingStuck-56-result_history.json'
    result_history = pd.read_json(path_result_history)
    print(len(result_history))
    print(result_history.columns)
    print(result_history)

    path_data = r'D:\Users\Administrator\Downloads\GeneratorBearingStuck-56_2\GeneratorBearingStuck-56-data.json'
    data = pd.read_json(path_data)

    print(len(data))
    print(data.columns)


    model_dr_path= r'Resource\dr\W056_generatorBearingTemp_model.bin'
    model_ndr_path = r'Resource\ndr\W056_generatorBearingTemp_model.bin'
    ws_gp_model_p =r'Resource\ws_power_model_56.pkl'
    with open(model_dr_path, 'rb') as fp:
        model_dr = pickle.load(fp)
    with open(model_ndr_path, 'rb') as fp:
        model_ndr = pickle.load(fp)
    with open(ws_gp_model_p, 'rb') as f:    ####直接用结冰算法的功率曲线训练文件好了，特征只有风速 ——Jimmy 20211115
        ws_gp_model = pickle.load(f)
        f.close()


    for n in range(30):
        time_range = pd.date_range(start_time, periods=2, freq='24H')
        # st = str(time_range[0])
        # et = str(time_range[1])
        st = time_range[0]
        et = time_range[1]

        # sample_data = data[(data['time'] >= st) & (data['time'] < et)]
        # sample_data[dt]= sample_data['time']
        # sample_data[ws] = sample_data['windSpeed']
        # sample_data[rs] = sample_data['generatorRotatingSpeed']
        # sample_data[temp_bear_dr] = sample_data['frontBearingTemperature']
        # sample_data[temp_bear_ndr] = sample_data['bearing2Temperature']
        # sample_data[ba] = sample_data['turbine_angle1']
        data[ts]=pd.to_datetime(data[ts])
        sample_data = data
        # sample_data = data[(data[dt] >= st) & (data[dt] < et)]
        sample_data = sample_data.dropna()
        sample_data.reset_index(inplace=True)

        result= {}
        result['raw_data'],result['analysis_data'] ,result['status_code'],result['start_time'],result['end_time'],  result['alarm'], result['distance'] = generator_bs_main(sample_data,result_history,model_dr, model_ndr, ws_gp_model )

        result['wtid'] = sample_data['wt_id'][0]
        # print(result)

        csvresult = pd.DataFrame(
            {'start_time': result['start_time'], 'end_time': result['end_time'],
             'distance': result['distance'],   'alarm': result['alarm'],  'status_code': result['status_code']}, index=['0'])
        opf= r'C:\Project\07_DaTang_BladeProtector\Algorithm\05_发电机轴承卡滞\Data\result'
        csvfn = opf + os.sep + str(result['wtid']) + '_pc.csv'

        csvresult.to_csv(csvfn, mode='a', index=False, header=False)
        # print (csvresult)

        start_time = et
