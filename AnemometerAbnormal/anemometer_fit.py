#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time :2018/10/31
# @author : T Bao
# version: 2.0.0

"""
风速仪异常识别模型训练

"""
import numpy as np
import sys
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import statsmodels.api as sm
import random
import glob
import datetime
import csv

maxgp = 3200  # 额定功率


temp = 'environment_temp'  # 机舱外环境温度
ba='pitch_1'  # 桨叶角度
window = 24  # 时间窗口，默认24h
ws = 'wind_speed'  # 风速
gp = 'power'  # 有功功率
gs = 'rotor_speed'  # 发动机转速
ts = 'time'
ye = 'yaw_error'  # 机舱外风向
gs= 'generator_rotating_speed'
turbineName= 'turbineName'

has_wtstatus = 'False'
wtstatus = 'main_status'
wtstatus_n = 10  ###用不到
model_type = 'A'
rs_lower = 5
ws_lower = 3
gp_lower = 10
ba_lower = 10
ba_upper = 90
gp_limit = 0.95
TI_upper = 0.4
ws_upper = 25
outpath = '..\\Resource\\anemometer_abnormal'
if not os.path.exists(outpath):
    os.makedirs(outpath)


def anemometer_wsunnormal_fit(data):

    # train_x = data.loc[:, [rs, ba, wd]].values  # 列选择

    train_x = data.loc[:, [gs, ba, ye, gp, temp]].values  # 风速预测中加入功率  jimmy 20221019 效果不明显  ##再加温度
    train_x2 = data.loc[:, [gs, ba, ye]].values

    train_y = data.loc[:, ws].values.reshape(-1, 1)  # 转变为1列
    train_y2 = data.loc[:, gp].values.reshape(-1, 1)
    # 将特征集和样本结果进行划分
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    rs_ws_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=1000, objective='reg:squarederror', silent=True)
    rs_ws_model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse', verbose=True)

    x_train, x_test, y_train, y_test = train_test_split(train_x2, train_y2, test_size=0.2, random_state=42)
    rs_gp_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500,  bjective='reg:squarederror', silent=True)
    rs_gp_model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='rmse', verbose=True)

    # 计算左右sigma边界
    ws_diff = data.loc[:, ws].values - rs_ws_model.predict(train_x)
    ws_diff_mean = np.mean(ws_diff)
    ws_diff_std = np.std(ws_diff)
    left3sigma = ws_diff_mean-3*ws_diff_std
    right3sigma = ws_diff_mean+3*ws_diff_std
    left4sigma = ws_diff_mean-4*ws_diff_std
    right4sigma = ws_diff_mean+4*ws_diff_std
    print('ws diff 3 sigma left ' + str(left3sigma))
    print('ws diff 3 sigma right ' + str(right3sigma))
    print('ws diff 4 sigma left' + str(left4sigma))
    print('ws diff 4 sigma right' + str(right4sigma))

    gp_diff = data.loc[:, gp].values - rs_gp_model.predict(train_x2)
    gp_diff_mean = np.mean(gp_diff)
    gp_diff_std = np.std(gp_diff)
    gp3sigma = gp_diff_mean+3*gp_diff_std
    gp4sigma = gp_diff_mean+4*gp_diff_std
    print('gp diff 3 sigma '+ str(gp3sigma))
    print('gp diff 4 sigma '+ str(gp4sigma))

    benchmark = [left3sigma, right3sigma, left4sigma, right4sigma, gp3sigma, gp4sigma]

    return rs_ws_model, rs_gp_model, benchmark


def anemometer_wsunnormal_fit_main(train):
    # global outpath
    global maxgp  # 额定功率
    global window
    global temp
    global ws
    global gp
    global gs
    global ye
    global ba
    global ts
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
    global TI_upper
    global ws_upper
    # 数据筛选部分设定
    wt_id= train[turbineName][0]
    temp_data = train[[temp]]
    train_bench = dict()
    train_bench['min_train_temp'] = np.nanpercentile(temp_data, 0.01)
    train_bench['max_train_temp'] = np.nanpercentile(temp_data, 99.99)

    train = train[[ts, gp, rs, ye, ws, ba, temp]]  ##加温度


    if len(train) > 0:

        train.loc[ (train[rs] > rs_lower) & (train[ws] > ws_lower) & (train[gp] > gp_lower)
               & (  ((train[gp] < maxgp * gp_limit) & (train[ba] < ba_lower)) | ((train[gp] > maxgp * gp_limit) & (train[ba] < ba_upper))), 'label'] = 1
                 # & ((train[gp] < maxgp * gp_limit) & (train[ba] < ba_lower)) , 'label'] = 1   ##保留额定风速以后的部分

        useless = train.loc[train['label'] == 0, :]
        train = train.loc[train['label'] == 1, :]

        print('Length of data after filter:', len(train))

        if len(train) > 1000:
            rs_ws_model, rs_gp_model, benchmark = anemometer_wsunnormal_fit(train)
            # print(1111111111111111111111111111111111)
            rs_ws_model_file = outpath + os.sep + str(wt_id) + '_rs_ws_model.model'
            with open(rs_ws_model_file, 'wb') as f:
                pickle.dump(rs_ws_model, f)
                # print(222222222222)
                f.close()
            #
            rs_gp_model_file = outpath + os.sep + str(wt_id) + '_rs_gp_model.model'
            with open(rs_gp_model_file, 'wb') as f:
                pickle.dump(rs_gp_model, f)
                # print(33333333333333)
                f.close()

            # sigma 值
            train_bench_file = outpath + os.sep + str(wt_id) + 'train_benchmark.csv'
            benchmark.append(wt_id)
            benchmark.append(datetime.datetime.now())
            with open(train_bench_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(benchmark)

            status_code = '000'
            return status_code, benchmark, rs_ws_model, rs_gp_model

        else:
            rs_ws_model, rs_gp_model = None, None
            benchmark = None
            status_code = '100'
            return status_code, benchmark, rs_ws_model, rs_gp_model
    else:
        status_code = '200'
        benchmark = None
        return status_code, benchmark



if __name__ == '__main__':
    import time as t

    data_path = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\华润风机数据\\数据\\offline_analysis\\1min\\3.2MW\\'


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
        # all_data_10min[gs] = all_data_10min[rs] *100

        all_data_10min[ts] = pd.to_datetime(all_data_10min[ts])
        all_data_10min = all_data_10min.drop_duplicates([ts])
        vars_10min = all_data_10min.columns.to_list()

        train = all_data_10min[:int(len(all_data_10min) * 0.4)]


        train = train.set_index(ts, drop=True).sort_index()
        train.dropna(subset=[ws, gp], inplace=True)
        for i in train.columns:
            if i == gs or i == ba:
                train[i] = train[i].interpolate(method='linear')
                train[i] = train[i].fillna(method='backfill')
                train[i] = train[i].fillna(method='ffill')
        train=train.reset_index(drop= False)

        status_code = anemometer_wsunnormal_fit_main(train)

        print (wt_id, 'Done')

