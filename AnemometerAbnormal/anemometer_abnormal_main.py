#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time :2021/08/20
# @author : Zf YU
# version: 3.0.0

"""
风速仪卡滞异常检测模型
model type A
Check if predicted wind speed is much higher than anemometer then it is stuck,
if it is much lower, then it is loosen
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import pickle
import pandas as pd
import xgboost as xgb
import os
import statsmodels.api as sm
from decimal import *
import math
import datetime
from scipy.interpolate import interp1d


maxgp = 2000  # 额定功率
window = 24  # 时间窗口，默认24h

temp = 'environment_temp'  # 机舱外环境温度
ws = 'wind_speed'  # 风速
gp = 'power'  # 有功功率
gs = 'generator_rotating_speed'  # 发动机转速
rs = 'rotor_speed'
ts = 'time'
ye = 'yaw_error'  # 机舱外风向
ba='pitch_1'  # 桨叶角度
turbineName= 'turbineName'
has_wtstatus = 'False'
wtstatus = 'main_status'
wtstatus_n = 14

model_type = 'A'
rs_lower = 5
ws_lower = 3
gp_lower = 10
ba_lower = 10
ba_upper = 90
gp_limit = 0.95


def window_mean(data, window, overlap):  # 滑动窗口取均值
    i = 1
    result = []
    while i < len(data):
        sdata = data[i:(i + window)]
        mean_window = np.mean(sdata)
        result.append(mean_window)
        i = i + overlap
    return result

# def distance_transform(x, max_3sigma, max_4sigma):
#     if x <= max_3sigma:
#         distance = (40 / max_3sigma) * x
#     elif (x > max_3sigma) & (x <= max_4sigma):
#         distance = (40/(max_4sigma-max_3sigma)) * x + (40-(20*max_3sigma/(max_4sigma-max_3sigma)))
#     else:
#         distance = 100
#     return 100-max(0, distance)

def distance_transform(value, x, y):
    #num = abs(math.acos(value))
    itpl = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
    value_t = float(itpl(value))
    return value_t 


def smooth_1min(data):
    # 分解时间特征
    data = data[[temp, ws, gp, gs, dt, ye, ba]]
    data['datetime'] = pd.to_datetime(data[dt], format='%Y-%m-%d %H:%M:%S')
    data['date'] = [i.date() for i in data['datetime']]
    data['hour'] = [i.hour for i in data['datetime']]
    data['minute'] = [i.minute for i in data['datetime']]
    data['second'] = [i.second for i in data['datetime']]
    data['date'] = data['date'].astype(str)
    data['hour'] = data['hour'].astype(str)
    data['minute'] = data['minute'].astype(str)
    data['second'] = data['second'].astype(str)

    # 数据预处理
    data[temp] = data[temp].astype(float)
    data[ws] = data[ws].astype(float)
    data[gp] = data[gp].astype(float)
    data[gs] = data[gs].astype(float)
    data[ba] = data[ba].astype(float)
    data[ye] = data[ye].astype(float)

    # 按分钟分类取平均
    smoothed_data = data[[temp, ws, gp, gs, ye, ba, 'date', 'hour', 'minute']].groupby([data['date'], data['hour'], data['minute']]).mean()
    smoothed_data = smoothed_data.reset_index()
    print("按照分钟分类后数据长度", len(smoothed_data))
    # print('smoothed_data')
    # print(smoothed_data)
    temp_d = data[ws].groupby([data['date'],data['hour'], data['minute']]).std(ddof=1)
    temp_d = temp_d.reset_index()
    # print('temp_d')
    # print(temp_d)
    smoothed_d = pd.merge(smoothed_data, temp_d, on=['date', 'hour', 'minute'], suffixes=('', '_std'))
    smoothed_d['TI'] = smoothed_d[ws+'_std']/smoothed_d[ws]
    smoothed_d = smoothed_d.dropna()
    print('去除TI空值后的数据长度', len(smoothed_d))
    smoothed_d.reset_index()
    # print(len(smoothed_d))
    smoothed_d[dt] = pd.to_datetime(smoothed_d['date'] + ' ' + smoothed_d['hour'] + ':' + smoothed_d['minute'] + ':00', format='%Y/%m/%d %H:%M:%S')
    smoothed_d=smoothed_d.sort_values(by=dt)
    
    return smoothed_d

def count_rule1(data):
    good_state = True
    start_index = 0
    counts = []
    for i in range(len(data)):
        # 计数？
        # if i % 1000 == 0:
        #     print("count1=", i)
        if (data.loc[i, 'ws_diff'] == 0) & (data.loc[i,'wd_diff'] > 1):
            if good_state:
                start_index = i
            good_state = False
        else:
            if not good_state:
                current_index = i
                counts.append(current_index - start_index)
            good_state = True
    return counts

def count_rule2(data):
    good_state = True
    start_index = 0
    counts = []
    for i in range(len(data)):
        # if i % 1000 == 0:
        #     print("count2=", i)
        if (data.loc[i, ws] == 0) & (data.loc[i, gp]>0):
            if good_state:
                start_index = i
            good_state = False
        else:
            if not good_state:
                current_index = i
                counts.append(current_index - start_index)
            good_state = True
    return counts

def count_rule3(data):
    good_state = True
    start_index = 0
    counts = []
    for i in range(len(data)):
        # if i % 1000 == 0:
        #     print("count3=", i)
        if (data.loc[i, ws]<3) & (data.loc[i, gp]>1000):
            if good_state:
                start_index = i
            good_state = False
        else:
            if not good_state:
                current_index = i
                counts.append(current_index - start_index)
            good_state = True
    return counts

def count_to_alarm(count30s,count60s,count90s,count120s):
    if count120s>0:
        alarm = 3
        distance = 0
        fault_type = 'Stuck'
    elif count90s>0:
        alarm = 3
        distance = 40
        fault_type = 'Stuck'
    elif count60s>0:
        alarm = 2
        distance = 60
        fault_type = 'Stuck'
    elif count30s>0:
        alarm = 1
        distance = 80
        fault_type = 'Stuck'
    else:
        alarm = 0
        distance = 100
        fault_type = None
    return alarm, distance, fault_type


def import_data_check(data, columns):
    # None
    #add by Liw
    data=data[columns]
    status_code = '000'
    #
    if data is None: # 判断数据中某列是否全部为空值
        status_code = '300'
        # raise Exception("Input Data_Raw is None")
    # pd.DataFrame()
    elif data.shape[0] == 0:
        status_code = '300'
        # raise Exception('Input Data_Raw is Empty Data_Raw Frame')
    # NAN
    # isnull()返回一个大小相等的全为bool值的数据，如果该处数据为nan，则为true
    # any()一个序列中满足一个True，则返回True；all()一个序列中所有值为True时，返回True，否则为False。
    # 判断是否有一列数据全为nan
    elif data.isnull().all().any():
        kk=data.isnull().all()
        non_col=kk[kk.values==True].index
        status_code = '300'
        # raise Exception('At Least One Column of Input Data_Raw is NAN: '+str(non_col))

    #dimension
    # if type(data.index) != pd.RangeIndex:
    #     print(type(data.index))
    #     print(pd.RangeIndex)
    #     raise Exception('wrongDimension')

    # lack of variable
    # issubset 检查某列是否存在
    if not set(columns).issubset(set(data.columns)):
        status_code = '300'
        # raise Exception('Some Variables are Missing'+str(columns) +'\\'+str(data.columns))
    # data type
    if (data[[temp, ws, gp, gs, ye, ba, rs]].dtypes != 'float').any():
        status_code = '300'
        # raise Exception('the Dtype of Input Data_Raw is not Float')

    #duplicate values
    length_origin = data.shape[0]  # 去除转速为负之后的数据长度
    # data1 = data
    # for i in [temp, ws, gp, rs, wd, ba]:
    #     data1 = data1[(data1[i] - data1[i].shift(1)) != 0.0]
    # length1 = data1.shape[0]  # 去除重复值之后的数据长度
    data2 = data
    # 检测上下两行数据的多个参数是否完全一致
    data2 = data2[(data2[ws] - data2[ws].shift(1)) +
                  (data2[gp] - data2[gp].shift(1)) + (data2[gs] - data2[gs].shift(1)) + (data2[ye] - data2[ye].shift(1)) +
                  (data2[ba] - data2[ba].shift(1)) != 0.0]
    length2 = len(data2)  # 去除重复值之后的数据长度
    #if (length1<0.5*length_origin) | (length2 <0.5*length_origin):
    # if (length2 <0.5*length_origin):
    #     raise Exception('Input Data_Raw Contains Too Many Duplicates')

    return status_code

def anemometer_wsunnormal_main(data,rs_gp_model_p,rs_ws_model_p,train_benchmark_path):
    """
    风速仪卡滞
    :param data: 数据集，dataframe
    :param history_data: 历史数据集，dataframe
    :return:
    """
    #global rs_gp_model_p  # rs_gp_model路径
    #global rs_ws_model_p  # rs_ws_model路径
    global maxgp  # 额定功率
    global window
    global temp
    global ws
    global gp
    global gs
    global rs
    global ye
    global ba
    global dt
    global has_wtstatus
    global wtstatus
    global wtstatus_n
    global model_type
    global rs_lower
    global ws_lower
    global ba_upper
    global ba_lower
    global gp_lower
    global gp_limit

    #global train_temp_range_path
    #global train_benchmark_path

    columns = [ts, gp, temp, ws, ye, rs, ba,gs]
    wt_id = data[turbineName][0]
    # 检查数据
    # print('原始数据长度', len(data))


    status_code = import_data_check(data, columns)

    # 数据的开始与结束时间
    end_time = pd.to_datetime(max(data[ts].values))
    start_time = pd.to_datetime(min(data[ts].values))


    # print("数据检查完毕")
    data = data[[ts, gp, temp, ws, ye, rs, ba,gs]]
    data = data.dropna()
    # print('去除空值后的长度', len(data))

    data = data.reset_index()

    # np.diff 计算数据的离散差值
    data['ws_diff'] = [math.nan] + list(np.diff(data[ws]))
    data['wd_diff'] = [math.nan] + list(np.diff(data[ye]))

    rule1_counts = count_rule1(data)
    rule2_counts = count_rule2(data)
    rule3_counts = count_rule3(data)

    count_30s = len([i for i in rule1_counts if i>8]) + len([i for i in rule2_counts if  i>8]) + len([i for i in rule3_counts if  i>8])
    count_60s = len([i for i in rule1_counts if  i>16]) + len([i for i in rule2_counts if  i>16]) + len([i for i in rule3_counts if  i>16])
    count_90s = len([i for i in rule1_counts if  i>24]) + len([i for i in rule2_counts if  i>24]) + len([i for i in rule3_counts if  i>24])
    count_120s = len([i for i in rule1_counts if  i>32]) + len([i for i in rule2_counts if  i>32]) + len([i for i in rule3_counts if  i>32])


    raw_data_notprocessed = dict()
    raw_data_notprocessed['datetime'] = list(map(str, pd.to_datetime(data[ts].values.tolist())))
    
    
    # Decimal mysql的精准数据类型， quantize函数，设置数据类型
    raw_data_notprocessed['ws'] = list(        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[ws].values.tolist()))
    raw_data_notprocessed['rs'] = list(        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[rs].values.tolist()))
    raw_data_notprocessed['gp'] = list(        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[gp].values.tolist()))
    raw_data_notprocessed['pred_ws'] = None
    raw_data_notprocessed['pred_gp'] = None
    #print("第一次筛选后的数据长度", len(train))
    
    
    # data = smooth_1min(data)  ##不用smooth, jimmy 20221026

    if has_wtstatus == 'True':
        data = data.loc[(data[ws] > ws_lower) & (abs(data[rs]) > rs_lower) & (data[wtstatus] == wtstatus_n), :]
    elif ba in data.columns:
        data = data.loc[ (data[rs] > rs_lower) & (data[ws] > ws_lower) & (data[gp] > gp_lower)
                        &(  ((data[gp] < maxgp * gp_limit) & (data[ba] < ba_lower)) | ((data[gp] > maxgp * gp_limit) & (data[ba] < ba_upper))), :]
    else:
        data = data.loc[(data[rs] > rs_lower) & (data[ws] > ws_lower), :]
    # print("筛选后数据大小为：", len(data))

    # plt.subplot(4, 1, 1)
    # plt.scatter(data[ws], data[ba], s=1)
    # plt.subplot(4, 1, 2)
    # plt.scatter(data[ws], data[gp], s=1)
    # plt.subplot(4, 1, 3)
    # plt.scatter(data[ws], data[rs], s=1)
    # plt.subplot(4, 1, 4)
    # plt.scatter(data[ws], data[wd], s=1)
    #
    # plt.show()

    if len(data) > 0:

        # calculate benchmark from last train

        train_benchark = pd.read_csv(train_benchmark_path, header=None)
        train_benchark.columns = ['left3sigma', 'right3sigma', 'left4sigma', 'right4sigma', 'gp3sigma', 'gp4sigma', 'wt_id', 'date']

        train_bench = train_benchark
        max_3sigma = max(train_bench['right3sigma'])
        max_4sigma = max(train_bench['right4sigma'])
        max_gp3sigma = max(train_bench['gp3sigma'])
        max_gp4sigma = max(train_bench['gp4sigma'])


        # 窗口，按照一分钟的时间分类
        data_adj = data
        if len(data_adj) > 60:
            try:
                # read in model files
                try:
                    with open(rs_gp_model_p, 'rb') as f:
                        rs_gp_model = pickle.load(f)
                        f.close()
                except:
                     raise Exception('load rs_gp_model_p error!')

                try:
                    with open(rs_ws_model_p, 'rb') as f:
                        rs_ws_model = pickle.load(f)
                        f.close()
                except:
                     raise Exception('load rs_ws_model_p error!')

                if model_type == 'A':
                    # 取三个有关联的数据
                    vlist = [rs, ba, ye, gp, temp]  ##改成5个 jimmy 20221019
                    x_data = data_adj.loc[:, vlist].values.reshape(-1, 5)

                    vlist = [rs, ba, ye]
                    x_data2 = data_adj.loc[:, vlist].values.reshape(-1, 3)

                    ws_data = data_adj.loc[:, ws].values.astype('float')
                    gp_data = data_adj.loc[:, gp].values.astype('float')

                    # 预测gp
                    data_adj['pred_gp'] = rs_gp_model.predict(x_data2)
                    # 小于0的全部变为0
                    data_adj.loc[(data_adj['pred_gp']<0), 'pred_gp'] = 0

                    # 预测ws
                    data_adj['pred_ws'] = rs_ws_model.predict(x_data)
                    data_adj['diff_gp'] = data_adj[gp] - data_adj['pred_gp']
                    data_adj['diff_ws'] = data_adj[ws] - data_adj['pred_ws']


                    # 步长5，范围10，取均值
                    smooth_diff_gp = window_mean(data_adj['diff_gp'], 10, 5)
                    smooth_diff_ws = window_mean(data_adj['diff_ws'], 10, 5)
                    if np.isnan(np.mean(smooth_diff_ws)):
                        diff_ws = np.mean(data_adj['diff_ws'])
                    else:
                        diff_ws = np.mean(smooth_diff_ws)
                    if np.isnan(np.mean(smooth_diff_gp)):
                        diff_gp = np.mean(data_adj['diff_gp'])
                    else:
                        diff_gp = np.mean(smooth_diff_gp)


                    if abs(diff_gp) < max_gp4sigma:
                        # 参数需要调整
                        if diff_ws > 0:
                            fault_type = 'Loosen'
                        else:
                            fault_type = 'Stuck'
                        # 计算健康值
                        #distance = int(distance_transform(abs(np.mean(smooth_diff_ws)), max_3sigma, max_4sigma))
                        max_1sigma=max_4sigma-max_3sigma
                        max_5sigma=max_1sigma*5
                        max_6sigma=max_1sigma*6

                        distance = int(distance_transform(abs(np.mean(smooth_diff_ws)),[0,max_3sigma,max_4sigma,max_5sigma, max_6sigma],[100,80,60,40,0] ))
                        if distance>100:
                            distance=100
                        if distance<0:
                            distance=0
                        # print("计算距离:", distance)
                    else:
                        distance = 100
                    if distance <= 40:
                        alarm = 3
                    elif (distance > 40) & (distance <= 60):
                        alarm = 2
                    elif (distance>60) & (distance<=80):
                        alarm = 1
                    elif (distance>80) & (distance<=100):
                        alarm = 0
                    else:
                        alarm = 0
                    # print(data)
                    # print(data_adj)
                    # combined_d = pd.merge(data, data_adj[['date', 'hour', 'minute', 'pred_gp', 'pred_ws', 'diff_gp', 'diff_ws']], how='left', on=['date', 'hour', 'minute'])
                    # print(combined_d)
                    raw_data = dict()

                    #raw_data['datetime'] = list(data_adj[['date', 'hour', 'minute']].values.reshape(-1, 3).tolist())

                    #liw add
                    #raw_data['datetime'] = list(map(str, pd.to_datetime(data_adj[time].values.tolist())))
                    #raw_data['datetime'] = list(map(str,pd.to_datetime(data_adj['date']+' '+data_adj['hour']+':'+data_adj['minute']+':00', format='%Y-%m-%d %H:%M:%S')))
                    raw_data['datetime'] =  list(map(str, pd.to_datetime(data_adj[ts], format='%Y-%m-%d %H:%M:%S').tolist()))
                    raw_data['ws'] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data_adj[ws].values.tolist()))
                    raw_data['pred_ws'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data_adj['pred_ws'].tolist()))
                    raw_data['rs'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data_adj[rs].values.tolist()))
                    raw_data['gp'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data_adj[gp].values.tolist()))
                    raw_data['pred_gp'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),  data_adj['pred_gp'].values.tolist()))


                    analysis_data = dict()

                    #analysis_data['datetime'] = list(data_adj[['date', 'hour', 'minute']].values.reshape(-1, 3).tolist())
                    #liw add
                    #analysis_data['datetime'] = list(map(str, pd.to_datetime(data_adj[time].values.tolist())))
                    #analysis_data['datetime'] = list(map(str,pd.to_datetime(data_adj['date']+' '+data_adj['hour']+':'+data_adj['minute']+':00', format='%Y-%m-%d %H:%M:%S')))
                    analysis_data['datetime'] =  list(map(str, pd.to_datetime(data_adj[ts], format='%Y-%m-%d %H:%M:%S').tolist()))
                    analysis_data['diff_ws'] = list(  map(lambda x: str(Decimal(x).quantize(Decimal('0.0000'))),    data_adj['diff_ws'].values.tolist()))
                    analysis_data['benchmark'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0'))),
                            [-(max_3sigma + (max_4sigma - max_3sigma) / 2), -max_3sigma, (-max_3sigma * 2 / 3),
                             (max_3sigma * 2 / 3), max_3sigma, (max_3sigma + (max_4sigma - max_3sigma) / 2)]))

                else:
                    print('Missing model type or model type is not valid.')

                if alarm==0:
                    result = dict()
                    result['start_time'] = str(start_time)
                    result['end_time'] = str(end_time)
                    result['distance'] = distance
                    result['fault_type'] = None
                    result['alarm'] = alarm
                    result['raw_data'] = raw_data
                    result['analysis_data'] = analysis_data
                    # print('sucess')
                    result['status_code'] = status_code
                else:
                    # print('alarm not 0')
                    result = dict()
                    result['start_time'] = str(start_time)
                    result['end_time'] = str(end_time)
                    result['distance'] = distance
                    result['fault_type'] = fault_type
                    result['alarm'] = alarm
                    result['raw_data'] = raw_data
                    result['analysis_data'] = analysis_data
                    result['status_code'] = status_code

            except:
                alarm, distance, fault_type = count_to_alarm(count_30s,count_60s,count_90s,count_120s)
                # print('no model file')
                status_code = '301'  # no model file
                result = dict()
                result['start_time'] = str(start_time)
                result['end_time'] = str(end_time)
                result['alarm'] = alarm
                result['raw_data'] = raw_data_notprocessed
                result['analysis_data'] = None
                result['distance'] = distance
                result['fault_type'] = fault_type
                result['status_code'] = status_code
        else:
            # print('TI and gp 不满足条件')
            alarm, distance, fault_type = count_to_alarm(count_30s,count_60s,count_90s,count_120s)
            status_code = '302'  # TI and gp 不满足条件
            result = dict()
            result['start_time'] = str(start_time)
            result['end_time'] = str(end_time)
            result['alarm'] = alarm
            result['raw_data'] = raw_data_notprocessed
            result['analysis_data'] = None
            result['distance'] = distance
            result['fault_type'] = fault_type
            result['status_code'] = status_code

    else:
        # print("data的长度为0")
        alarm, distance, fault_type = count_to_alarm(count_30s,count_60s,count_90s,count_120s)
        status_code = '300'  # status code etc filter
        result = dict()
        result['start_time'] = str(start_time)
        result['end_time'] = str(end_time)
        result['alarm'] = alarm
        result['raw_data'] = raw_data_notprocessed
        result['analysis_data'] = None
        result['distance'] = distance
        result['fault_type'] = fault_type
        result['status_code'] = status_code

    if result['status_code']=='000':
        if result['distance'] < 80:
            plt.plot(result['raw_data']['ws'], label='wind_speed')
            plt.plot(result['raw_data']['pred_ws'], label ='predict_wind_speed')
            plt.title('distance=' + str(result['distance']))
            plt.legend()

            savePath = '../Result/anemometer_abnormal/fault/'
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(savePath + wt_id + '_' + str(start_time)[:10] + '.jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
            plt.clf()
    return result['start_time'],result['end_time'],result['raw_data'],result['analysis_data'],result['status_code'],result['alarm'],result['distance']



if __name__ == "__main__":
    model_gp_path = r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\风速仪异常\01_Git\ModelTrain_SourceCode\result\W031_rs_gp_model.model'
    model_ws_path = r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\风速仪异常\01_Git\ModelTrain_SourceCode\result\W031_rs_ws_model.model'
    benchmark_path = r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\风速仪异常\01_Git\ModelTrain_SourceCode\result\train_benchmark-MY.csv'
    with open(model_gp_path, 'rb') as fp:
        model_gp = pickle.load(fp)
    with open(model_ws_path, 'rb') as fp:
        model_ws = pickle.load(fp)


    path2 = r'D:\Users\Administrator\Desktop\AnemometerAbnormal-31\AnemometerAbnormal-31-data.json'
    input_data = pd.read_json(path2, encoding='utf-8')
    # input_data['wind_cabin_angle'] = input_data['wind_direction']
    train = input_data[[dt, gp, temp, ws, ye, gs, ba]]

    # plt.subplot(4, 1, 1)
    # plt.scatter(train[ws], train[ba], s=1)
    # plt.subplot(4, 1, 2)
    # plt.scatter(train[ws], train[gp], s=1)
    # plt.subplot(4, 1, 3)
    # plt.scatter(train[ws], train[rs], s=1)
    # plt.subplot(4, 1, 4)
    # plt.scatter(train[ws], train[wd], s=1)
    #
    # plt.show()

    result={}
    result['start_time'],result['end_time'],result['raw_data'],\
    result['analysis_data'],result['status_code'],result['alarm'],\
    result['distance'] = anemometer_wsunnormal_main(data=train, rs_gp_model_p=model_gp_path ,rs_ws_model_p=model_ws_path,train_benchmark_path=benchmark_path )
    pass