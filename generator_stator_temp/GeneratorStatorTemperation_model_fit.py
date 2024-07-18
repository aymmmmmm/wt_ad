#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GeneratorStatorTemperation_model_fit.py
@Time    :   2021/09/28 10:42:47
@Author  :   Xu Zelin   updated from jing.fan
@Version :   1.0
'''

from functools import total_ordering
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os
import datetime
import pymongo
from lightgbm import LGBMRegressor
from scipy.stats import norm


generatorStatorTemp=['generator_winding_temp_U']
bearingtemp='f_i_bearing_temp'
bearingtemp_F ='r_i_bearing_temp'
rspeed = 'generator_rotating_speed'
ec='grid_current_A'
t='time'
p='power'
cab_temp='cabin_temp'
turbineName= 'turbineName'




def data_process(data):
    """
    特征：绕组温度、机舱温度、轴承温度、电流平方、转速平方、前一时刻功率、x7、前一时刻绕组温度和机舱温度组合是否满足某个条件、
    后一时刻时间是否连续(实际没用到)
    y:前1分钟的温度
    """
    print(data.info())
    data.interpolate(method='linear', limit=25 * 60, axis=0, inplace=True)  # TODO 增加插值 25分钟*每分钟30条 by XuZelin

    data["gen_temp_median"] = data[generatorStatorTemp].max(axis=1)  # TODO 尝试max
    data = data.loc[:, ["gen_temp_median", bearingtemp, bearingtemp_F, rspeed, ec, t, p, cab_temp]]
    data[t] = pd.to_datetime(data[t])
    data = data.resample('10T', on=t).mean()
    data = data.reset_index()
    data = data.sort_values(by=t)
    data['x1'] = data["gen_temp_median"].shift(1)  # TODO x1和y替换了,by Xuzelin
    data['x2'] = data[cab_temp]
    data['x3'] = data[bearingtemp]
    data['x4'] = (data[ec]) ** 2
    data['x5'] = (data[rspeed]) ** 2
    data['x6'] = data[p].shift(1)
    data['x7'] = data[bearingtemp_F]  # np.ones(np.size(data['x6']))# TODO x7:所有数值是1？,是的,改成了前轴承温度
    data['y'] = data["gen_temp_median"]
    data = data.reset_index()
    data['x8'] = 0

    # TODO 优化这里的代码,加快运行速度 by XuZelin
    data["x8"] = data[["gen_temp_median", cab_temp]].apply(lambda x: int(x["gen_temp_median"] > 50 or x[cab_temp] > 47),
                                                           axis=1)
    if 45 < data.loc[1, "gen_temp_median"] < 50 or 45 < data.loc[1, cab_temp] < 47:
        if data.loc[2, "gen_temp_median"] < data.loc[1, "gen_temp_median"] or data.loc[2, cab_temp] < data.loc[
            1, cab_temp]:
            data.loc[1, 'x8'] = 1
        else:
            if data.loc[0, 'x8'] == 1:
                data.loc[1, 'x8'] = 1

    '''
    # 原始代码
    for i in range(len(data[generatorStatorTemp])):# i=1处，为什么这么处理？
        if data.loc[i,generatorStatorTemp] > 50 or data.loc[i,cab_temp] > 47:
            data.loc[i,'x8'] = 1
        elif 45 < data.loc[i,generatorStatorTemp] < 50 or 45 < data.loc[i,cab_temp] < 47:
            if i == 1:
                if data.loc[i+1,generatorStatorTemp] < data.loc[i,generatorStatorTemp] or data.loc[i+1,cab_temp] < data.loc[i,cab_temp]:
                    data.loc[i,'x8'] = 1
                else:
                    if data.loc[i-1,'x8'] == 1:
                        data.loc[i,'x8'] = 1
    '''
    data['x8'] = data['x8'].shift(1)
    # data = time_continuity(data, sLength=60)# TODO 特征没用到，先注释by XuZelin
    data = data.dropna(subset=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y'])  # TODO 增加了subset参数
    return data


def time_continuity(data, sLength=60):
    interval = datetime.timedelta(seconds=sLength)  # second length
    data = data.reset_index()
    data['time_continuity_flag'] = False
    for i in range(0, len(data[t]) - 1):
        x = data.loc[i, t].strftime('%Y-%m-%d-%H-%M')
        x_ = data.loc[i + 1, t].strftime('%Y-%m-%d-%H-%M')
        if datetime.datetime.strptime(x, '%Y-%m-%d-%H-%M') + interval == datetime.datetime.strptime(x_,
                                                                                                    '%Y-%m-%d-%H-%M'):
            data.loc[i + 1, 'time_continuity_flag'] = True
    train_data = data[data['time_continuity_flag'] == True]
    return train_data

def estimate_sigma(residuals):
    pdf = norm.pdf(residuals)
    prob = pdf / pdf.sum()  # these are probabilities
    mu = residuals.dot(prob)  # mean value
    # print("#### mu:",mu)
    mom2 = np.power(residuals, 2).dot(prob)  # 2nd moment
    var = mom2 - mu ** 2  # variance
    return np.sqrt(var), mu  # TODO 返回mu by XuZelin  standard deviation


def stator_temp_fit_physical(data):  # data has been processed by data_process function
    physical_model = LinearRegression()  # # LGBMRegressor()
    feature_column = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']  #
    physical_model.fit(data.loc[:, feature_column], data.loc[:, ['y']])
    print('fit score:', str(physical_model.score(data.loc[:, feature_column], data.loc[:, ['y']])))
    return physical_model


def stator_temp_predict_physical(physical_model, data):  # data has been processed by data_process function
    feature_column = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']  # 'x1',
    data['predict'] = physical_model.predict(data.loc[:, feature_column])
    print('test score:', str(physical_model.score(data.loc[:, feature_column], data.loc[:, ['y']])))
    data['residual'] = data['predict'] - data['y']
    return data['residual'].values


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


        # all_data_10min[t] = pd.to_datetime(all_data_10min[t])  ##插值的时候不能shi datatimeStamp
        all_data_10min = all_data_10min.drop_duplicates([t])
        vars_10min = all_data_10min.columns.to_list()

        # train = all_data_10min[:int(len(all_data_10min) * 0.4)]

        columns= [bearingtemp, bearingtemp_F, rspeed, ec, t, p, cab_temp] + generatorStatorTemp

        ndata = data_process(all_data_10min[columns])  # 构造特征

        train_data = ndata[:int(len(ndata) * 0.4)]
        test_data = ndata[int(len(ndata) * 0.4) : int(len(ndata) * 0.5)]
        print('len of train data:', train_data.shape)
        print('len of test data:', test_data.shape)

        physical_model = stator_temp_fit_physical(train_data)

        data_residual = stator_temp_predict_physical(physical_model, test_data)
        # print("系数：",physical_model.coef_)
        print("均值：", data_residual.mean())

        outpath_physical = '../Resource/generator_stator_temp/'
        if not os.path.exists(outpath_physical):
            os.makedirs(outpath_physical)

        model_file=os.path.join(outpath_physical,(wt_id+'_' + 'model_max.pkl'))
        with open(model_file, 'wb') as f:
            pickle.dump(physical_model, f)

        data_file=os.path.join(outpath_physical,(wt_id+'_' + 'data_max.pkl'))
        with open(data_file, 'wb') as f:
            pickle.dump(data_residual, f)

        base_sigma, base_mu = estimate_sigma(data_residual)
        print("base_sigma:", base_sigma)
