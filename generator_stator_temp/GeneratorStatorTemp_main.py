#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GeneratorStatorTemp_main.py
@Time    :   2021/09/28 10:42:40
@Author  :   Xu Zelin   updated from TZ18100801
@Version :   1.0
'''

from logging import raiseExceptions
import logging
from matplotlib import markers
import numpy as np
from numpy.lib.index_tricks import _fill_diagonal_dispatcher
import pandas as pd
from pandas.tseries.offsets import Hour
from scipy.stats import norm
from collections import Counter
import warnings
# from PublicUtils.utils import estimate_sigma
warnings.filterwarnings("ignore")
import datetime
import pymongo
import yaml
import random
from decimal import *
import statsmodels.api as sm
import pickle
import os
import matplotlib.pyplot as plt



online = True
isJFWT = False

window = 18  # TODO 原来重合过多，144和12
step = 18
n_lookback_warning = 10
n_lookback_alarm = 100
n_continuous = 3
base_sigma = 2  ##默认是2



generatorStatorTemp = ['generator_winding_temp_U']
bearingtemp = 'f_i_bearing_temp'
bearingtemp_F = 'r_i_bearing_temp'
rspeed = 'generator_rotating_speed'
ec = 'grid_current_A'
t = 'time'
p = 'power'
cab_temp = 'cabin_temp'




columns = [t, bearingtemp, bearingtemp_F, rspeed, ec, p, cab_temp] + [col for col in generatorStatorTemp]

def estimate_sigma(residuals):
    pdf = norm.pdf(residuals)
    prob = pdf / pdf.sum()  # these are probabilities
    mu = residuals.dot(prob)  # mean value
    # print("#### mu:",mu)
    mom2 = np.power(residuals, 2).dot(prob)  # 2nd moment
    var = mom2 - mu ** 2  # variance
    return np.sqrt(var),mu  #TODO 返回mu by XuZelin  standard deviation

def data_process(data):
    """
    特征：绕组温度、机舱温度、轴承温度、电流平方、转速平方、前一时刻功率、x7、前一时刻绕组温度和机舱温度组合是否满足某个条件、
    后一时刻时间是否连续(实际没用到)
    y:前1分钟的温度
    """
    #print(data.info())
    data.loc[:,([bearingtemp, bearingtemp_F, rspeed, ec, p, cab_temp] + [col for col in generatorStatorTemp])]=data.loc[:,([bearingtemp, bearingtemp_F, rspeed, ec, p, cab_temp] + [col for col in generatorStatorTemp])].astype('float64')
    data.loc[:,([bearingtemp, bearingtemp_F, rspeed, ec, p, cab_temp] + [col for col in generatorStatorTemp])]=data.loc[:,([bearingtemp, bearingtemp_F, rspeed, ec, p, cab_temp] + [col for col in generatorStatorTemp])].interpolate(method='linear', limit=3, axis=0, inplace=False)  # TODO 增加插值 25分钟*每分钟30条 by XuZelin
    #data[t]=pd.to_datetime(data[t])
    # print('interpolate completed.')
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


def stator_temp_predict_physical(physical_model, data):
    # data = data_process(data)
    data['predict'] = physical_model.predict(data.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']])  #
    data['residual'] = data['predict'] - data['y']
    return data


def find_majority(warning):
    vote_count = Counter(warning)
    top_two = vote_count.most_common(2)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        return 1

    return top_two[0][0]


def rolling_HI(data, model, residual, base_sigma, window, step):
    HI = np.zeros((4, 1))  # .reshape(-1, 1)#4行1列
    base_sigma_discard, base_mu = estimate_sigma(residual)  # TODO 返回mu
    # print("Base_sigma",base_sigma)
    for i in range(window, len(data), step):
        # data_tmp = data.iloc[i - window: i, :]
        diff_tmp = data.iloc[(i - window): i, :]['residual'].values
        # stator_temp_predict_physical(model, data_tmp)['residual'].values.astype(np.float)
        # TODO 实际上standard_mu并不是0，修改为base_mu, by xuzelin
        HI_tmp = np.array(compute_HI(diff_tmp, base_mu, base_sigma)).reshape(-1, 1)   ###4个区间内各自的点数百分比，总和为1
        HI = np.concatenate((HI, HI_tmp), axis=1)
    HI = HI[:, 1:]  # 4行n列的数据，n对应多少个window，4个数分别是统计residual在哪个sigma范围的比例
    return HI

def get_warning(HI, warning_history):
    warning = []
    for i in range(HI.shape[1]):
        if HI[:, i][0] + HI[:, i][1] < HI[:, i][2] + HI[:, i][3]:  # 3,4比1,2大
            warning.append(1)
        else:
            warning.append(0)
    final_warning = find_majority(warning)
    return final_warning


def alarm_level(alarm_history, n_lookback_alarm, alarm):
    alarm_history_tmp = alarm_history[-n_lookback_alarm:]
    distance = 100 - min(100 * len([x for x in alarm_history_tmp if x != 0]) / n_lookback_alarm + 20,
                         100)  # TODO min第2项从0改为100,+40改为+20
    if alarm == 1:
        if 60 < distance <= 80:  # TODO 由60关注改为80关注
            alarm_level = 1
        elif 40 < distance <= 60:
            alarm_level = 2
        elif distance <= 40:
            alarm_level = 3
    else:
        distance = max(80 + distance / 4, 80.1)  # TODO 原来是100，改成根据原distance调整
        alarm_level = 0
    return float(distance), int(alarm_level)


def get_alarm(warning_history, alarm_history, n_lookback_warning, n_lookback_alarm, n_continuous):
    alarm = 0
    #
    if len(warning_history) > 1:
        # print("warning history greater than 1")
        pass

    if len(warning_history) >= 10:
        if all(a == 1 for a in warning_history[-n_continuous:]) or find_majority(
                warning_history[-n_lookback_warning:]) == 1:
            alarm = 1
    else:
        if all(a == 1 for a in warning_history[-n_continuous:]):  # warning_history连续1个warning也会使alarm=1
            alarm = 1
    final_level = alarm_level(alarm_history, n_lookback_alarm, alarm)
    return final_level


def compute_penalty(value, sigma):
    # 统计偏差在各个sigma区间，返回1，2，3，4
    k = [1, 2, 3]  # TODO 原来[2,4,6]
    if value < k[0] * sigma:
        return 1
    elif k[0] * sigma <= value <  k[1] * sigma:
        return 2
    elif  k[1] * sigma <= value <  k[2] * sigma:
        return 3
    elif k[2] * sigma <= value:
        return 4


def compute_HI(residuals, standard_mu, standard_sigma):
    # 生成4个统计1，2，3，4变量的统计量,1,2,3,4当前权重分别是1,2,3,4得到sum_Cn,然后计算1,2,3,4的占比
    array = np.zeros((len(residuals),))
    for i in range(len(array)): ##每一个残差位于 1σ，2σ， 3σ 哪个区间
        array[i] = compute_penalty(np.abs(residuals.reshape(-1)[i] - standard_mu), standard_sigma)  # TODO 修改为np.abs()
    #sum_Cn = (array == 1).sum() * 1 + (array == 2).sum() * 2 + (array == 3).sum() * 3 + (array == 4).sum() * 4
    HI_1 = (array == 1).sum() / array.size# * 1 / sum_Cn
    HI_2 = (array == 2).sum() / array.size# * 2 / sum_Cn
    HI_3 = (array == 3).sum() / array.size# * 3 / sum_Cn
    HI_4 = (array == 4).sum() / array.size# * 4 / sum_Cn
    return HI_1, HI_2, HI_3, HI_4


def import_data_check(data):
    if data is None:
        raise Exception("input data is None")
    # pd.DataFrame()
    if data.shape[0] == 0:
        raise Exception("input data is empty DataFrame()")
    # drop variable
    # if (set(data.columns[1:-1]).issubset(set(columns))) and (not set(columns).issubset(set(data.columns[1:-1]))):  #注意要1：-1，因为有_id
    #    raise Exception("the variable columns is missing")
    # input NAN
    if data.isnull().all().all():
        raise Exception("input data is NAN")
    # dimension
    # if type(data.index) != pd.RangeIndex: # TODO 暂时注释
    #     raise Exception('wrongDimension')
    if data[columns].isnull().all().any():  # 判断数据中某列是否全部为空值
        tmp = data[columns].notna().sum()
        cols = list(tmp[tmp == 0].index)
        raise Exception(f'data column {cols} is NAN')

    # data_tmp = data.copy()
    # window_5s = window * 12
    # length1 = data.shape[0]  # 原始数据长度
    # data = data.dropna(subset=columns)
    # length2 = data.shape[0]  # 去除缺失值之后长度
    # data = data[data[rspeed] >= 0]
    # length3 = data.shape[0]  # 去除转速为负之后的数据长度
    if (data[columns].std() == 0).any():
        raise BaseException('column of input data has repeated value')  # 判断数据某列数值是否全部重复
    # print('data preposessing completed.')
    # TODO 判断是否局部全部不变，如果是，去掉这部分数据
    if online:
        data[t] = pd.to_datetime(data[t])
        data.set_index(t, inplace=True, drop=False)
    # print('reset index completed.')
    # tmp = (data[columns].resample("H").std() == 0).apply(lambda x: x.all(), axis=1)  # 某个小时所有数值都不变
    # if tmp.any():
    #     ind_list = [ind.strftime("%Y-%m-%d %H") for ind in tmp[tmp == True].index]
    #     for ind in ind_list:
    #         data[ind] = None
    #     data = data.dropna(how="all")
    # input data
    if data[columns].notna().sum().min() < window * 12:  # TODO 略微简化代码
        raise Exception('the length of input data is not enough')              
    # input includes NAN
    # elif length1 > length2 and length2 < window_5s:
    #     raise Exception('the length of input data is not enough because of columnContainsNAN')
    # negative rs
    elif len(data[data[rspeed] >= 0][rspeed].dropna()) < window * 12:
        raise Exception('the length of input data is not enough because of the negative generatorRotatingSpeed')
    # print('window checked.')
    # elif length3 > length4 and length4 < window_5s:
    # print(window_5s)
    # print(length4)
    # print(length1)
    #    raise Exception('the length of input data is not enough because of the duplicated values in certain columns')

    check_result = dict()
    check_result['response'] = True
    check_result['status_code'] = '000'

    # TODO updated by xzl
    if data[columns].notna().sum().min() < 0.05 * len(data):
        check_result['response'] = False
        check_result['status_code'] = '100'

    ''' #原来代码
    if data_tmp.isnull().any().any():
        for i in data_tmp.columns: # 判断各列缺失值的比例是否大于5%
            missing_value_counts = len(data_tmp[i]) - data_tmp[i].count()
            missing_value_rate = (missing_value_counts / len(data_tmp[i])) * 100
            if missing_value_rate >= 5:
                check_result['response'] = False
                check_result['status_code'] = '100'
                break
    '''
    return check_result, data


def mongo_check(wtid, collection, warning_history, alarm_history):
    try:
        wtid = wtid[-2:]  # TODO 数据库只存两位数字,sort字段修改
        for i in collection.find({'deviceNo': wtid}).sort([('invoke_time', -1)])[:n_lookback_warning]:
            warning_history.append(i['analysis_data']['warning'])
        for i in collection.find({'deviceNo': wtid}).sort([('invoke_time', -1)])[:n_lookback_alarm]:
            alarm_history.append(i['alarm'])
        mongo_status_code = '000'
    except:
        warning_history.append(0)
        alarm_history.append(0)
        mongo_status_code = '020'
    return warning_history, alarm_history, mongo_status_code


def generator_stator_temp_main(data, model, base_plot, warning_history, alarm_history):
    global generatorStatorTemp
    global bearingtemp
    global bearingtemp_F
    global rspeed
    global ec
    global t
    global p
    global cab_temp
    # global model_dr
    global n_lookback_warning
    global n_lookback_alarm
    global n_continuous
    global base_sigma
    global step
    global window
    global columns
    # global buffer_config_fp
    # global base_plot_path
    if not online:
        global i

    columns = [t, bearingtemp, bearingtemp_F, rspeed, ec, p, cab_temp] + [col for col in generatorStatorTemp]
    #data=pd.DataFrame(data)
    data = data.loc[:,columns]  # 选取需要的数据点位

    check_result, data = import_data_check(data)  #
    status_code = check_result['status_code']

    # print('data checked.')


    base_residual = base_plot

    data_test = data_process(data)
    # print("测试数据大小:",j,len(data_test))
    data_test_pred = stator_temp_predict_physical(model, data_test)
    residual = data_test_pred['residual'].values.astype(np.float)
    HI = rolling_HI(data_test, model, base_residual, base_sigma, window, step)
    warning = get_warning(HI, warning_history)  ##这里不需要warning_history
    warning_history.append(warning)
    # print(len(warning_history),warning_history)
    alarm = get_alarm(warning_history, alarm_history, n_lookback_warning, n_lookback_alarm, n_continuous)  # alarm[0]是distance， alarm[1]是真正的alarm
    alarm_history.append(alarm[1])

    kde1base = sm.nonparametric.KDEUnivariate(base_residual)
    kde1base.fit(bw=1)
    kde1 = sm.nonparametric.KDEUnivariate(residual)
    kde1.fit(bw=1)

    pred_plot_t = pd.DataFrame()
    if len(kde1.density) < 2000:
        sample_size = len(kde1.density)
    else:
        sample_size = 2000
    sample_len = random.sample(list(range(len(kde1.density))), sample_size)
    pred_plot_t['support'] = kde1.support[sample_len]
    pred_plot_t['density'] = kde1.density[sample_len]

    ###  sort values
    pred_plot_t = pred_plot_t.sort_values(by='support')
    pred_plot = dict()
    pred_plot['support'] = pred_plot_t['support'].values
    pred_plot['density'] = pred_plot_t['density'].values

    base_plot_t = pd.DataFrame()
    if len(kde1base.density) < 2000:
        sample_size = len(kde1base.density)
    else:
        sample_size = 2000
    sample_len = random.sample(list(range(len(kde1base.density))), sample_size)
    base_plot_t['support'] = kde1base.support[sample_len]
    base_plot_t['density'] = kde1base.density[sample_len]

    ###  sort values
    base_plot_t = base_plot_t.sort_values(by='support')
    base_plot = dict()
    base_plot['support'] = base_plot_t['support'].values
    base_plot['density'] = base_plot_t['density'].values

    # print('residual completed.')

    # construct outputs
    raw_data = {}
    raw_data['datetime'] = list(map(str, data_test_pred[t].tolist()))
    raw_data['stator_temp'] = data_test_pred["gen_temp_median"].values.tolist()
    raw_data['stator_temp_pred'] = data_test_pred['predict'].values.tolist()

    analysis_data = {}
    analysis_data['density_x1'] = list(        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), pred_plot['support']))
    analysis_data['density_y1'] = list(        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), pred_plot['density']))
    analysis_data['base_x'] = list(        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), base_plot['support']))
    analysis_data['base_y'] = list(        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), base_plot['density']))
    analysis_data['warning'] = warning

    result = {}
    result['start_time'] = str(min(data_test[t]))
    result['end_time'] = str(max(data_test[t]))
    result['raw_data'] = raw_data
    result['analysis_data'] = analysis_data
    result['distance'] = alarm[0]
    result['alarm'] = alarm[1]

    # print('alarm completed.')

    # if mongo_status_code == '020' and status_code == '000':  # TODO 暂时注释2行
    #     status_code = '020'
    result['status_code'] = status_code

    # print(result["distance"], result["alarm"])
    start_time=result["start_time"]
    end_time=result["end_time"]
    raw_data=result["raw_data"]
    analysis_data=result["analysis_data"]
    status_code=result["status_code"]
    distance=result["distance"]
    alarm=result["alarm"]


    return (start_time, end_time, raw_data, analysis_data, status_code, distance, alarm)


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def result_plot(result, res_index):
    plt.rcParams['font.sans-serif'] = ['simhei']  # 解决中文显示问题-设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax0 = fig.add_subplot(gs[:, 0])

    raw_time = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), result["raw_data"]["datetime"]))
    ax1.plot(raw_time, result["raw_data"]['stator_temp'], label="True", c="b")
    ax1.plot(raw_time, result["raw_data"]['stator_temp_pred'], '--r', label="Pred")
    ax1.set_title("发电机绕组温度")
    ax1.set_xlabel(result['status_code'])
    ax1.grid()
    ax1.legend()

    if result['analysis_data'] and result['analysis_data']["density_x1"]:
        ax2.plot(result["analysis_data"]["base_x"], result["analysis_data"]["base_y"], c="b", label="base")
        ax2.plot(result["analysis_data"]["density_x1"], result["analysis_data"]["density_y1"], "--r", label="test")
        ax2.set_title(f"健康值: {res_index['distances'][-1]},   预警等级: {result['alarm']}")
        # ax2.set_ylabel("温差特征值")
        ax2.legend()
    ax0.plot(res_index["index"], np.array(res_index["distances"]), marker="D", markerfacecolor="cyan", label="健康值")
    ax0.set_ylabel("偏差角度")
    ax0.set_ylim([0, 110])
    ax0.legend(loc="upper left")
    ax0.grid()
    plt.show()

# if __name__ == '__main__':
#     online = False

#     data_path = "../../data/" if isJFWT else "../../data/MY/"
#     with open("./model/GeneratorStatorTemp_max.pkl","rb") as f:#TODO 取max
#         model_all = pickle.load(f)
#     for i in ["031"]:  #循环每个风机"070","085","056",  MY: "019","031","010"
#         res_index = dict(distances = [],index = [])

#         print("*"*50,"W"+i,"*"*50)

#         data = pd.read_csv(data_path + "W" +  i +".csv",index_col=0)
#         data["time"] = pd.to_datetime(data["time"])+datetime.timedelta(hours=8)#时区
#         data.set_index("time",drop=False,inplace=True)
#         data = data["2021-6-1":]#["2021-6-25":"2021-6-28"]
#         ts = pd.date_range(data.index[0],data.index[-1],freq="d",normalize=True)
#         for j in range(23,24):# len(ts)-3  16,17  len(ts)-28  22,24
#             tmp = data.loc[:, columns][ts[j]:ts[j+3]]
#             if len(tmp.resample("10T").mean().dropna()) > window:#防止数据过少,无法计算
#                 res = generator_stator_temp_main(tmp,model_all,None)
#                 res_index["distances"].append(res['distance'])
#                 res_index["index"].append(ts[j+3])
#                 print("***",ts[j+3],end="    ")

#         if not online:
#             result_plot(res,res_index)    
#         print("min_distance:",np.min(res_index["distances"]))


#         print(res.keys(),len(res["raw_data"]["datetime"]),len(res["analysis_data"]["density_x1"]))


# 在线数据测试
import json

# with open("./model/GeneratorStatorTemp" + ".pkl", "rb") as f:
#     model_all = pickle.load(f)
# # model = model_all["W085"]
# with open("../data/GeneratorStatorTemp-31-data.json", "r") as f:
#     data = json.load(f)
# res = pd.DataFrame(np.array(data), columns=["info"])
# for col in ["winding_temp_1", "winding_temp_2", "winding_temp_3", "winding_temp_4", "winding_temp_5", "winding_temp_6",
#             'rear_generator_temp', 'front_generator_temp', 'grid_curr_L1', 'generator_rotating_speed',
#             'dataTimeStamp', 'active_power', 'cabinTemperature', 'assetName']:
#     # ['winding_temp_max','r_i_bearing_temp','f_i_bearing_temp', 'generator_rotating_speed','converter_generatorside_curr_1',
#     #         'dataTimeStamp','active_power','cabinTemperature','assetName']:
#     res[col] = res["info"].apply(lambda x: x.get(col, None))
# generator_stator_temp_main(res, model_all, buffer_config_fp)

