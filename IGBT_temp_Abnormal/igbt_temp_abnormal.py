#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @datetime :2021/11/26 13:29
# @author : R XIE
# @update : after Erzhong.chang by Xu Zelin
# @version : 1.0.3

import numpy as np
import pandas as pd
from decimal import *
import statsmodels.api as sm
from scipy.interpolate import interp1d
import datetime
import time
import os
import matplotlib.pyplot as plt

online = True


# 在线
dt = 'time'
gp = 'power'
temp1 = ['converter_temp_genside'] # 网侧逆变器温度
temp2 = ['converter_temp_gridside'] # 机侧逆变器温度
wtidvar = 'turbineName'

columns = [dt, gp, wtidvar]+[col for col in temp1]+[col for col in temp2]
k = [0, 3, 4, 5, 6]
maxgp = 3300
h_temp_limit_gridside = 100 #高温阈值
h_temp_limit_turbineside = 110 #高温阈值
wt_ids = ["56","70","85"] #集群内所有风机
output_wtids = ["56","70","85"] #需要输出结果的风机

# fleet_dt = 'fleet_' + str(dt)
# fleet_temp1 = 'fleet_' + str(temp1)
# fleet_temp2 = 'fleet_' + str(temp2)

def data_process(data,temp1,temp2):#TODO 缺失值处理,取最大值
    """
    """
    
    for wt in wt_ids:
        data.loc[data[wtidvar]==wt,[gp]+[col for col in temp1]+[col for col in temp2]] = data.loc[data[wtidvar]==wt,[gp]+[col for col in temp1]+[col for col in temp2]].interpolate(method='linear',limit=25*60,axis=0)#TODO 增加插值
    data["gridSideTemp"] = np.max(data[temp1],axis=1)# TODO 比.apply(lambda x:x.max()快很多
    data["turbineSideTemp"] = np.max(data[temp2],axis=1)
    temp1 = "gridSideTemp"
    temp2 = "turbineSideTemp"
    return data,temp1,temp2

def result_format(result,res,wt):
    if result["start_time"]:
        result["start_time"] = min(time.strptime(result["start_time"],"%Y-%m-%d %H:%M:%S"),time.strptime(res["start_time"],"%Y-%m-%d %H:%M:%S"))
        result["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S",result["start_time"])
        result["end_time"] = min(time.strptime(result["end_time"],"%Y-%m-%d %H:%M:%S"),time.strptime(res["end_time"],"%Y-%m-%d %H:%M:%S"))
        result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S",result["end_time"])
    else:
        result["start_time"] = res["start_time"]
        result["end_time"] = res["end_time"]
    for k in ["raw_data","analysis_data","status_code","distance","distance1","distance2","alarm"]:#
        result[k][wt]  = res[k]  
    return result



def distance_transform(value, x, y):
    if len(x) == len(y):
        lenx = len(x)
        if value < x[0]:
            value_t = 0
        elif (value >= x[0]) & (value < x[lenx - 1]):
            itpl = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            value_t = float(itpl(value))
        else:
            value_t = 100
        return 100 - value_t
    else:
        raise Exception

def rank_transform(distance):# TODO 健康值和rank对应修改
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

def resample_bytime_mean(data, minnum, dt):
    """
    resample by time window
    return mean value of resampled data
    :param data: 输入数据，dataframe
    :param minnum: 按几分钟resample，float
    :param dt: 日期时间变量名，str
    :return: 输出结果，dataframe
    """
    data.index = data[dt].values
    scale = str(minnum) + 'T'
    r = data.resample(scale).mean()
    r[dt] = r.index
    return r

def igbt_temp_abnormal(data, dt, gp, temp1, temp2, wtidvar, k, wtid, maxgp):
    """
    两温度测点的IGBT温度异常集群对标模型
    :param data: 输入数据，同时间内同风场所有机组数据，dataframe
    :param dt: 时间变量，str
    :param gp: 有功功率，str
    :param temp1: IGBT温度1，str
    :param temp2: IGBT温度2，str
    :param wtidvar: 机组号变量，str
    :param k: 阈值系数，str
    :param wtid: 当前机组号，str
    :return: dict
    """
    status_code = '000'
    data = data.loc[:, [dt, gp, temp1, temp2, wtidvar]]
    data[dt] = pd.to_datetime(data[dt])
    end_time = str(max(data[dt]))
    start_time = str(min(data[dt]))
    # gp_lower = np.percentile(data[gp].values, 90)
    # gp_upper = np.percentile(data[gp].values, 100)
    gp_lower = maxgp * 0.8
    gp_upper = maxgp * 1.2

    raw_data = dict()
    raw_data["fleet_t"] = []
    raw_data["fleet_gridSideTemp"] = []
    raw_data["fleet_turbineSideTemp"] = []
    raw_data["t"] = []
    raw_data["gridSideTemp"] = []
    raw_data["turbineSideTemp"] = []
    analysis_data = dict()
    analysis_data['fleet_density1_x'] = []
    analysis_data['fleet_density2_x'] = []
    analysis_data['fleet_density1_y'] = []
    analysis_data['fleet_density2_y'] = []
    analysis_data['density1_x'] = []
    analysis_data['density1_y'] = []
    analysis_data['density2_x'] = []
    analysis_data['density2_y'] = []
    result = dict()

    rawd_grouped = resample_bytime_mean(data, 10, dt)
    rawd_grouped = rawd_grouped.dropna()

    # fleet_dt = 'fleet_' + str(dt)
    # fleet_temp1 = 'fleet_' + str(temp1)
    # fleet_temp2 = 'fleet_' + str(temp2)

    raw_data["fleet_t"] = list(map(str, pd.to_datetime(rawd_grouped[dt].values).tolist()))
    raw_data["fleet_gridSideTemp"] = list(
        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd_grouped[temp1].values.tolist()))
    raw_data["fleet_turbineSideTemp"] = list(
        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd_grouped[temp2].values.tolist()))

    if len(data.loc[data[wtidvar] == wtid, :]) > 0:
        rawd_wt = resample_bytime_mean(data.loc[data[wtidvar] == wtid, :], 10, dt)
        rawd_wt = rawd_wt.dropna()
        raw_data["t"] = list(map(str, pd.to_datetime(rawd_wt[dt].values).tolist()))
        raw_data["gridSideTemp"] = list(
            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd_wt[temp1].values.tolist()))
        raw_data["turbineSideTemp"] = list(
            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd_wt[temp2].values.tolist()))
    else:
        status_code = '201'  # 输入数据不为空，但该机组数据为空
        # raw_data[dt] = []
        # raw_data[temp1] = []
        # raw_data[temp2] = []

    if (np.nanstd(data[temp1]) != 0) & (np.nanmean(data[temp1]) > 0) & (np.nanstd(data[temp2]) != 0) & (
        np.nanmean(data[temp2]) > 0):
        if (np.nanmean(data[temp1]) > 10) & (np.nanmean(data[temp2]) > 10):
            temp1_mean_all = np.mean(data.loc[:, temp1])
            temp2_mean_all = np.mean(data.loc[:, temp2])

            temp1_std_all = np.nanstd(data.loc[:, temp1])
            temp2_std_all = np.nanstd(data.loc[:, temp2])

            thresholds1_inner_all = []
            thresholds2_inner_all = []
            for x in k:
                thresholds1_inner_all.append(temp1_mean_all + x * temp1_std_all)
                thresholds2_inner_all.append(temp2_mean_all + x * temp2_std_all)

            if len(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), :]) > 0:
                temp1_mean = np.mean(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), temp1])
                temp2_mean = np.mean(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), temp2])

                temp1_std = np.std(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), temp1])
                temp2_std = np.std(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), temp2])

                kde1 = sm.nonparametric.KDEUnivariate( data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), temp1].astype('float').values)
                kde1.fit(bw=1)

                kde2 = sm.nonparametric.KDEUnivariate(  data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), temp2].astype('float').values)
                kde2.fit(bw=1)

                analysis_data['fleet_density1_x'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde1.support))
                analysis_data['fleet_density1_y'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde1.density))
                analysis_data['fleet_density2_x'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde2.support))
                analysis_data['fleet_density2_y'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde2.density))

                sdata = data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper) & (data[wtidvar] == wtid), :]
                if len(sdata) > 0:
                    temp1_mean_wt = np.mean(   sdata.loc[:, temp1])
                    temp2_mean_wt = np.mean(   sdata.loc[:, temp2])

                    thresholds1_inner = []
                    thresholds2_inner = []
                    for x in k:
                        thresholds1_inner.append(temp1_mean + x * temp1_std)
                        thresholds2_inner.append(temp2_mean + x * temp2_std)

                    distance1 = float(   Decimal(distance_transform(temp1_mean_wt, thresholds1_inner, [0, 20, 40, 60, 100])).quantize(                            Decimal('0.0000')))
                    distance2 = float(   Decimal(distance_transform(temp2_mean_wt, thresholds2_inner, [0, 20, 40, 60, 100])).quantize(                            Decimal('0.0000')))
                    if (distance1 > 80) and (len(sdata.loc[sdata[temp1] > h_temp_limit_gridside])/len(sdata) > 0.6):
                        distance1 = 80 #TODO 强制改distance
                    if (distance2 > 80) and (len(sdata.loc[sdata[temp2] > h_temp_limit_turbineside])/len(sdata) > 0.6):
                        distance2 = 80 #TODO 强制改distance    
                    distance = min(distance1, distance2)
                    alarm = rank_transform(distance)

                    kde3 = sm.nonparametric.KDEUnivariate(sdata.loc[:, temp1].astype('float').values)
                    kde3.fit(bw=1)

                    kde4 = sm.nonparametric.KDEUnivariate(sdata.loc[:, temp2].astype('float').values)
                    kde4.fit(bw=1)

                    analysis_data['density1_x'] = list(                        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde3.support))
                    analysis_data['density1_y'] = list(                        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde3.density))
                    analysis_data['density2_x'] = list(                        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde4.support))
                    analysis_data['density2_y'] = list(                        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde4.density))

                else:
                    status_code = '300'  # 该机组符合工况数据为空
                    # analysis_data['density1_x'] = []
                    # analysis_data['density1_y'] = []
                    # analysis_data['density2_x'] = []
                    # analysis_data['density2_y'] = []

                    if len(data.loc[data[wtidvar] == wtid, :]) > 0:
                        distance1 = float(
                            Decimal(distance_transform(np.mean(data.loc[data[wtidvar] == wtid, temp1]),
                                                       thresholds1_inner_all,
                                                       [0, 20, 40, 60, 100])).quantize(
                                Decimal('0.0000')))
                        distance2 = float(
                            Decimal(distance_transform(np.mean(data.loc[data[wtidvar] == wtid, temp2]),
                                                       thresholds2_inner_all,
                                                       [0, 20, 40, 60, 100])).quantize(
                                Decimal('0.0000')))
                        distance = min(distance1, distance2)#TODO max改成min
                        alarm = rank_transform(distance)
                    else:
                        status_code = '201'
                        distance1 = None
                        distance2 = None
                        distance = None
                        alarm = None
            else:
                status_code = '301'  # 全场符合工况数据为空
                # analysis_data['fleet_density1_x'] = []
                # analysis_data['fleet_density2_x'] = []
                # analysis_data['fleet_density1_y'] = []
                # analysis_data['fleet_density2_y'] = []
                # analysis_data['density1_x'] = []
                # analysis_data['density1_y'] = []
                # analysis_data['density2_x'] = []
                # analysis_data['density2_y'] = []

                if len(data.loc[data[wtidvar] == wtid, :]) > 0:
                    distance1 = float(  Decimal(   distance_transform(np.mean(data.loc[data[wtidvar] == wtid, temp1]), thresholds1_inner_all,
                                               [0, 20, 40, 60, 100])).quantize(   Decimal('0.0000')))
                    distance2 = float(  Decimal(   distance_transform(np.mean(data.loc[data[wtidvar] == wtid, temp2]), thresholds2_inner_all,
                                               [0, 20, 40, 60, 100])).quantize(   Decimal('0.0000')))
                    distance = min(distance1, distance2)#TODO
                    alarm = rank_transform(distance)
                    
                else:
                    status_code = '201'
                    distance1 = None
                    distance2 = None
                    distance = None
                    alarm = None
        else:
            status_code = '204'  # 数据质量异常：温度数据可能不正常
            # analysis_data['fleet_density1_x'] = []
            # analysis_data['fleet_density2_x'] = []
            # analysis_data['fleet_density1_y'] = []
            # analysis_data['fleet_density2_y'] = []
            # analysis_data['density1_x'] = []
            # analysis_data['density1_y'] = []
            # analysis_data['density2_x'] = []
            # analysis_data['density2_y'] = []
            distance1 = None
            distance2 = None
            distance = None
            alarm = None
    else:
        status_code = '203'  # 数据质量异常：温度数据均值为0或没有变化
        # analysis_data['fleet_density1_x'] = []
        # analysis_data['fleet_density2_x'] = []
        # analysis_data['fleet_density1_y'] = []
        # analysis_data['fleet_density2_y'] = []
        # analysis_data['density1_x'] = []
        # analysis_data['density1_y'] = []
        # analysis_data['density2_x'] = []
        # analysis_data['density2_y'] = []
        distance1 = None
        distance2 = None
        distance = None
        alarm = None

    result['distance'] = distance
    result['distance1'] = distance1
    result['distance2'] = distance2
    result['raw_data'] = raw_data
    result['analysis_data'] = analysis_data
    result['status_code'] = status_code
    result['start_time'] = start_time
    result['end_time'] = end_time
    result['alarm'] = alarm
    return result

def igbt_temp_abnormal_wrapper(data, wtid, dt, gp, temp1, temp2, wtidvar, k, maxgp):
       
    data,temp1,temp2 = data_process(data,temp1,temp2)#数据处理
    
    raw_data = dict()
    raw_data["fleet_t"] = []
    raw_data["fleet_gridSideTemp"] = []
    raw_data["fleet_turbineSideTemp"] = []
    raw_data["t"] = []
    raw_data["gridSideTemp"] = []
    raw_data["turbineSideTemp"] = []
    analysis_data = dict()
    analysis_data['fleet_density1_x'] = []
    analysis_data['fleet_density2_x'] = []
    analysis_data['fleet_density1_y'] = []
    analysis_data['fleet_density2_y'] = []
    analysis_data['density1_x'] = []
    analysis_data['density1_y'] = []
    analysis_data['density2_x'] = []
    analysis_data['density2_y'] = []
    

    if len(data)>0:
        data = data.dropna(subset=[gp, temp1, temp2])

        if isinstance(k, str):
            k = eval(k)

        if len(data) > 0:
            result = igbt_temp_abnormal(data, dt, gp, temp1, temp2, wtidvar, k, wtid, maxgp)
        else:
            status_code = '200'
            result = dict()

            result['distance'] = None
            result['distance1'] = None
            result['distance2'] = None
            result['raw_data'] = raw_data
            result['analysis_data'] = analysis_data
            result['start_time'] = None
            result['end_time'] = None
            result['status_code'] = status_code
            result['alarm'] = None
    else:
        status_code = '200'
        result = dict()

        result['distance'] = None
        result['distance1'] = None
        result['distance2'] = None
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['start_time'] = None
        result['end_time'] = None
        result['status_code'] = status_code
        result['alarm'] = None

    ##############报警结果plot
    if (result['distance'] is not None) & (result['status_code'] == '000'):
        if result['distance'] < 90:

            # ################Plot 最终结果
            plt.subplot(2,1,1)
            plt.plot(result['analysis_data']['fleet_density1_x'], result['analysis_data']['fleet_density1_y'],  label='集群')
            plt.plot(result['analysis_data']['density1_x'], result['analysis_data']['density1_y'], color='black', label='本机组')
            plt.xlabel('温度1')
            plt.ylabel('概率密度')
            plt.legend()
            plt.title(wtid + '_' + str(result['start_time'])[0:10] + '_distance' + str(result['distance']))

            plt.subplot(2,1,2)
            plt.plot(result['analysis_data']['fleet_density2_x'], result['analysis_data']['fleet_density2_y'],  label='集群')
            plt.plot(result['analysis_data']['density2_x'], result['analysis_data']['density2_y'], color='black', label='本机组')
            plt.xlabel('温度2')
            plt.ylabel('概率密度')
            plt.legend()

            # plt.show()
            savePath = '../Result/igbt_temp_abnormal/fault/'

            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(savePath + wtid + '_' + str(result['start_time'])[0:10] + 'igbt_temp.jpg', format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
            plt.clf()

    return result


def igbt_abnormal_main(data, _wt_ids, _output_wtids):
    global dt
    global gp
    global temp1
    global temp2
    global wtidvar
    global k
    global maxgp
    # global wtid
    global wt_ids
    global output_wtids
    global h_temp_limit_gridside
    global h_temp_limit_turbineside
    global columns
    
    
    ##为了跑离线新增的 Jimmy 20221107
    wt_ids=_wt_ids
    output_wtids = _output_wtids


    columns = [dt, gp, wtidvar]+[col for col in temp1]+[col for col in temp2]
    wt_ids = [str(c) for c in wt_ids] # TODO 在线转换为str
    output_wtids = [str(c) for c in output_wtids]

    result ={
        "start_time":None,
        "end_time":None,
        "raw_data":{},
        "analysis_data":{},
        "status_code":{},
        "distance":{},
        "distance1":{},
        "distance2":{},
        "alarm":{}
    }
    
    for wt in output_wtids:

        res = igbt_temp_abnormal_wrapper(data, wt, dt, gp, temp1, temp2, wtidvar, k, maxgp)
        result = result_format(result,res,wt)



    return (result["start_time"],result["end_time"],result["raw_data"],result["analysis_data"],
            result["status_code"],result["distance"],result["distance1"],result["distance2"],result["alarm"])

    
if __name__ == "__main__":
#     online = False
    
#     folder = "../../data"
#     data = pd.DataFrame()
#     for i in ["W056","W070","W085"]:
#         print("*"*50,i,"*"*50)
#         tmp = pd.read_csv(os.path.join(folder,i+".csv"))[columns]#,nrows = 100000
#         data = pd.concat([data,tmp])
#     data["time"] = pd.to_datetime(data["time"]) + datetime.timedelta(hours=8)
#     data.set_index("time",drop=False,inplace=True)
#     data[wtidvar] = data[wtidvar].apply(lambda x:x[-2:])
#     data.sort_index(inplace=True)
    
#     periods = pd.date_range(start = data["time"].min(),end=data.time.max(),freq = '1D',normalize=True)#
    
#     res_index ={"time":[],"result":{"56":[],"70":[],"85":[]},"alarm":{"56":[],"70":[],"85":[]}}
#     for j in range(len(periods)-3):#  len(periods)
#         tmp = data[periods[j]:periods[j+3]]
#         if  len(tmp.resample("10T").mean().dropna()) > 10 :#防止数据过少,无法计算
#             res = igbt_abnormal_main(tmp)
#             print(periods[j+3],res['distance'],res['alarm'])
#             res_index["time"].append(periods[j+3].date())
#             for wt in output_wtids:
#                 res_index["result"][wt].append(res["distance"][wt])
#                 res_index["alarm"][wt].append(res["alarm"][wt])
#                 # print(periods[j+3],res['distance'],res['alarm'])
#     plt.rcParams['font.sans-serif'] = ['simhei']  # 解决中文显示问题-设置字体为黑体
#     plt.rcParams['axes.unicode_minus'] = False
#     fig,ax=plt.subplots(2,1,figsize=(16,8))
#     for wt in output_wtids:
#         ax[0].plot(res_index["time"],res_index["result"][wt],label="W0"+wt,marker=".")
#         ax[1].plot(res_index["time"],res_index["alarm"][wt],label="W0"+wt,marker=".")
#         ax[0].set_title("健康值")
#         ax[0].set_ylim([0,110])
#         ax[0].legend()
#         ax[1].set_title("alarm")
#         ax[1].set_ylim([-1,5])
#         ax[1].legend()
#         ax[0].grid()
#         ax[1].grid()
#     plt.show()
    
    #  # 在线数据测试
    import json
    
    with open(r"D:\Users\Administrator\Desktop\IgbtTempAbnormal-GW\IgbtTempAbnormal-GW-data.json","r") as f:
        data  = json.load(f)
    res = pd.DataFrame(np.array(data),columns=["info"])
    for col in [dt, gp, wtidvar]+[col for col in temp1]+[col for col in temp2]:
        res[col] = res["info"].apply(lambda x:x.get(col,None))
    result = igbt_abnormal_main(res)
    print(result[4],result[5],result[6])