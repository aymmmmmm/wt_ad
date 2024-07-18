#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @datetime :2021/11/24 15:28
# @author : R XIE 
# @updated after Erzhong.chang by Xuzelin
# @version : 1.0.3

import numpy as np
import pandas as pd
import os
import pickle
from decimal import *
import statsmodels.api as sm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import datetime
import time



ts = 'time'
gp = 'power'
temp1 = 'converter_temp_gridside'
turbineName = 'turbineName'
wt_ids = [ ] #集群内所有风机




# ts = 'dataTimeStamp'
# gp = 'active_power'
# temp1 = 'IGBT_temp'
# turbineName = 'assetName'
# wt_ids = ["10","19","31"] #集群内所有风机

k = '[0, 3, 4, 5, 6]'
maxgp = 2000
columns = [ts, gp, temp1, turbineName]
h_temp_limit = 120 #高温阈值

output_wtids = wt_ids #需要输出结果的风机  "10","19",

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

def converter_temp_abnormal_os(data, dt, gp, temp1, wtidvar, k, wtid, maxgp):
    """
    单测点的变流器温度异常集群对标模型
    :param data: 输入数据，dataframe
    :param dt: 时间变量名，str
    :param gp: 有功功率变量名，str
    :param temp1: 变流器温度变量名，str
    :param wtidvar: 机组号变量名，str
    :param k: 阈值系数，str
    :param wtid: 当前机组号，需与数据中采用的值一致，str
    :return: dict
    """
    status_code = '000'
    data = data.loc[:, [dt, gp, temp1, wtidvar]]
    data[dt] = pd.to_datetime(data[dt])
    end_time = str(max(data[dt]))
    start_time = str(min(data[dt]))
    gp_lower = maxgp*0.8
    gp_upper = maxgp*1.2

    result = dict()

    raw_data = dict()
    raw_data["fleet_dt"] = None
    raw_data["fleet_cit"] = None
    raw_data["t"] = None
    raw_data["cit"] = None

    analysis_data = dict()
    analysis_data['fleet_density1_x'] = None
    analysis_data['fleet_density1_y'] = None
    analysis_data['density1_x'] = None
    analysis_data['density1_y'] = None

    rawd_grouped = resample_bytime_mean(data, 10, dt)
    rawd_grouped = rawd_grouped.dropna()

    # fleet_dt = 'fleet_' + str(dt)
    # fleet_temp1 = 'fleet_' + str(temp1)

    raw_data["fleet_dt"] = list(map(str, pd.to_datetime(rawd_grouped[dt].values).tolist()))
    raw_data["fleet_cit"] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd_grouped[temp1].values.tolist()))

    if len(data.loc[data[wtidvar] == wtid, :]) > 0:
        rawd_wt = resample_bytime_mean(data.loc[data[wtidvar] == wtid, :], 10, dt)
        rawd_wt = rawd_wt.dropna()
        raw_data["t"] = list(map(str, pd.to_datetime(rawd_wt[dt].values).tolist()))
        raw_data["cit"] = list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), rawd_wt[temp1].values.tolist()))
    else:

        # raise Exception('WT is NULL while Unit is not NUll')
        status_code = '201'
        raw_data["t"] = []
        raw_data["cit"] = []

    if (np.nanstd(data[temp1]) != 0) & (np.nanmean(data[temp1]) > 0):
        if np.nanmean(data[temp1]) > 10:
            if len(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper), :]) > 10:
                temp1_mean = np.mean(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper) , temp1])# TODO 不包括自己,注意wtid加了引号 & (data[wtidvar] != 'wtid')
                temp1_std = np.std(data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper) , temp1])#TODO 不包括自己,注意wtid加了引号 & (data[wtidvar] != 'wtid')

                kde1 = sm.nonparametric.KDEUnivariate( data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper) , temp1].astype('float').values)#TODO 不包括自己,注意wtid加了引号 & (data[wtidvar] != 'wtid')
                kde1.fit(bw=1)

                analysis_data['fleet_density1_x'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde1.support))
                analysis_data['fleet_density1_y'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde1.density))
                # plt.plot(analysis_data['fleet_density1_x'],analysis_data['fleet_density1_y'],'b') # 

                sdata = data.loc[(data[gp] > gp_lower) & (data[gp] <= gp_upper) & (data[wtidvar] == wtid), :]
                if len(sdata) > 10:
                    temp1_mean_wt = np.mean(   sdata.loc[:, temp1])

                    thresholds1_inner = []
                    for x in k:
                        thresholds1_inner.append(temp1_mean + x * temp1_std)
                    # print(temp1_mean_wt)
                    # print(thresholds1_inner)

                    distance = distance_transform(temp1_mean_wt, thresholds1_inner, [0, 20, 40, 60, 100])
                    alarm = rank_transform(distance)
                    if (alarm == 0) & (len(sdata.loc[sdata[temp1]> h_temp_limit])/len(sdata) > 0.6):
                        alarm = 1 #TODO alram
                        distance = 80 #TODO 强制改distance
                    #     print("---------")
                    # else:
                    #     print("**********")

                    kde3 = sm.nonparametric.KDEUnivariate(sdata.loc[:, temp1].astype('float').values)
                    kde3.fit(bw=1)

                    analysis_data['density1_x'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde3.support))
                    analysis_data['density1_y'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), kde3.density))
                    # plt.plot(analysis_data['density1_x'],analysis_data['density1_y'],'g')
                    #plt.xlabel('support set')
                    #plt.ylabel('probability density')
                    #plt.title('Temperature probability density of No.31 every three days')
                    
                else:
                    status_code = '302'  # 本机无符合工况的数据
                    # analysis_data['density1_x'] = []
                    # analysis_data['density1_y'] = []
                    # if len(data.loc[data[wtidvar] == wtid, :]) > 0:
                    #     distance = float(
                    #         Decimal(
                    #             distance_transform(np.mean(data.loc[data[wtidvar] == wtid, temp1]), [0, 30, 40, 60, 80],
                    #                                [0, 40, 60, 80, 100])).quantize(
                    #             Decimal('0.0000')))
                    #     alarm = rank_transform(distance)
                    # else:
                        # status_code = '201'
                    analysis_data['density1_x'] = None
                    analysis_data['density1_y'] = None
                    distance = None
                    alarm = None

            else:
                status_code = '301'  # 所有机组无符合工况数据
                # if len(data.loc[data[wtidvar] == wtid, :]) > 0:
                #     distance = float(
                #         Decimal(distance_transform(np.mean(data.loc[data[wtidvar] == wtid, temp1]), [0, 30, 40, 60, 80],
                #                                    [0, 40, 60, 80, 100])).quantize(
                #             Decimal('0.0000')))
                #     alarm = rank_transform(distance)
                # else:
                #     status_code = '201'
                #     distance = None
                #     alarm = None
                analysis_data = None
                # analysis_data['fleet_density1_x'] = []
                # analysis_data['fleet_density1_y'] = []
                # analysis_data['density1_x'] = []
                # analysis_data['density1_y'] = []
                distance = None
                alarm = None

        else:
            status_code = '304'  # 数据质量异常：温度数据可能不正常
            analysis_data = None
            # analysis_data['fleet_density1_x'] = []
            # analysis_data['fleet_density1_y'] = []
            # analysis_data['density1_x'] = []
            # analysis_data['density1_y'] = []
            distance = None
            alarm = None
    else:
        status_code = '303'  # 数据质量异常：温度数据均值为0或没有变化
        analysis_data = None
        # analysis_data['fleet_density1_x'] = []
        # analysis_data['fleet_density1_y'] = []
        # analysis_data['density1_x'] = []
        # analysis_data['density1_y'] = []
        distance = None
        alarm = None
    

    
    
    result['distance'] = distance
    result['raw_data'] = raw_data
    result['analysis_data'] = analysis_data
    result['status_code'] = status_code
    result['start_time'] = start_time
    result['end_time'] = end_time
    result['alarm'] = alarm
    return result

def data_process(data):#TODO 缺失值处理
    """
    """
    for wt in wt_ids:
        data.loc[data[turbineName] == wt, [gp, temp1]] = data.loc[data[turbineName] == wt, [gp, temp1]].interpolate(method='linear', limit=25 * 60, axis=0)#TODO 增加插值
    return data

def converter_temp_abnormal_wrapper(data, dt, gp, temp1, wtidvar, k, wtid, maxgp):
    # None
    status_code = '000'
    result = dict()
    if data is None: # 判断数据中某列是否全部为空值
        status_code = '300'
        # raise Exception("input data is None")
    #pd.DataFrame()
    elif data.shape[0] == 0:
        status_code = '300'
        # raise Exception('input data is pd.DataFrame()')

    if data[columns].isnull().all().any(): # 判断数据中某列是否全部为空值
        tmp = data[columns].notna().sum()
        cols = list(tmp[tmp == 0].index)
        status_code = '300'
        # raise Exception(f'data column {cols} is NAN')

    #drop variable
    if not set(columns).issubset(set(data.columns)):
        status_code = '300'
        # raise Exception('Variable Column is Missing')

    result['distance'] = None
    result['raw_data'] = None
    result['analysis_data'] = None
    result['status_code'] = status_code
    result['start_time'] = None
    result['end_time'] = None
    result['alarm'] = None

    data = data_process(data)#数据处理

    if len(data)>0:
        data = data.dropna(subset=[gp, temp1])
        if isinstance(k, str):
            k = eval(k)

        if len(data) > 0:
            result = converter_temp_abnormal_os(data, dt, gp, temp1, wtidvar, k, wtid, maxgp)
        else:
            status_code = '300'
            # raise Exception('the more nan value')



##############报警结果plot
    if result['distance'] is not None:
        if result['distance'] < 90:

            # ################Plot 最终结果
            plt.plot(result['analysis_data']['fleet_density1_x'], result['analysis_data']['fleet_density1_y'],  label='集群')
            plt.plot(result['analysis_data']['density1_x'], result['analysis_data']['density1_y'], color='black', label='本机组')

            plt.xlabel('温度')
            plt.ylabel('概率密度')
            plt.legend()
            plt.title(wtid + '_' + str(result['start_time'])[0:10]+ '_distance'+ str(result['distance']))

            # plt.show()
            savePath = '../Result/converter_temp_abnormal/fault/'

            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(savePath + wtid + '_' + str(result['start_time'])[0:10] + 'converter_temp.jpg', format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
            plt.clf()

    return result

def result_format(result,res,wt):
    if result["start_time"]:
        result["start_time"] = min(time.strptime(result["start_time"],"%Y-%m-%d %H:%M:%S"),time.strptime(res["start_time"],"%Y-%m-%d %H:%M:%S"))
        result["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S",result["start_time"])
        result["end_time"] = min(time.strptime(result["end_time"],"%Y-%m-%d %H:%M:%S"),time.strptime(res["end_time"],"%Y-%m-%d %H:%M:%S"))
        result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S",result["end_time"])
    else:
        result["start_time"] = res["start_time"]
        result["end_time"] = res["end_time"]
    for k in ["raw_data","analysis_data","status_code","distance","alarm"]:
        result[k][wt]  = res[k]  
    return result

def converter_temp_abnormal_main(data, _wt_ids, _output_wtids):
    global ts
    global gp
    global temp1
    global turbineName
    global k
    global wt_ids
    global output_wtids

    ##为了跑离线新增的 Jimmy 20221107
    wt_ids=_wt_ids
    output_wtids = _output_wtids


    wt_ids = [str(c) for c in wt_ids] # TODO 在线转换为str
    output_wtids = [str(c) for c in output_wtids]
    
    result ={
        "start_time":None,
        "end_time":None,
        "raw_data":{},
        "analysis_data":{},
        "status_code":{},
        "distance":{},
        "alarm":{}
    }
    
    data = data[columns]
    for wt in output_wtids:
        # print("*"*50,wt,"*"*50)
        res = converter_temp_abnormal_wrapper(data, ts, gp, temp1, turbineName, k, wt, maxgp)
        print (wt, res['start_time'], res['status_code'], res['alarm'], res['distance'])
        result = result_format(result,res,wt)

    return (result["start_time"],result["end_time"],result["raw_data"],result["analysis_data"],
            result["status_code"],result["distance"],result["alarm"])

    
if __name__ == '__main__':
    
    # """
    # import pymongo
    # conn = pymongo.MongoClient(host="192.168.199.243", port=27017)
    # db = conn['farm']
    # collection = db['eastelec_data_process']
    # print("*************Connected*****************")
    # ts_range = pd.date_range('2019-05-23 12:40:00', '2019-05-23 20:40:00', periods=2)
    # data = pd.DataFrame()
    # for ts in range(len(ts_range) - 1):
    #     # print(ts_range[ts])
    #     st = str(ts_range[ts])
    #     et = str(ts_range[ts + 1])

    #     for i in collection.find(
    #             {'wt_id': 'WT5', 't': {'$gt': st, '$lt': et}},
    #             {'active_power': 1, 'cit': 1,
    #              't': 1, 'wt_id': 1}):
    #         d = pd.DataFrame.from_dict(i, orient='index').T
    #         data = pd.concat([data, d])

    #     data = data.reset_index(drop=True)
    # """
    
    # online = False
    # folder = "../../data/MY"
    # data = pd.DataFrame()
    # for i in ["W019","W031","W010"]:#"W056","W070","W085"  MY: "W019","W031","W010"
    #     print("*"*50,i,"*"*50)
    #     tmp = pd.read_csv(os.path.join(folder,i+".csv"))[columns]#,nrows = 400000
    #     data = pd.concat([data,tmp])
    # data["time"] = pd.to_datetime(data["time"]) + datetime.timedelta(hours=8)
    # data.set_index("time",drop=False,inplace=True)
    # data["wt_id"] = data["wt_id"].apply(lambda x:x[-2:])
    # data.sort_index(inplace=True)
    
    # periods = pd.date_range(start = data["time"].min(),end=data.time.max(),freq = '1D',normalize=True)#
    
    # res_index ={"time":[],"result":{"10":[],"19":[],"31":[]},"alarm":{"10":[],"19":[],"31":[]}}
    # for j in range(len(periods)-3):#  len(periods)  len(periods)-3
    #     temp = data[periods[j]:periods[j+3]]
    #     if  len(temp.resample("10T").mean().dropna()) > 10 :#防止数据过少,无法计算
    #         res = converter_temp_abnormal_main(temp)
    #         print(periods[j+3],res['distance'],res['alarm'])
    #         res_index["time"].append(periods[j+3].date())
    #         for wt in output_wtids:
    #             res_index["result"][wt].append(res["distance"][wt])
    #             res_index["alarm"][wt].append(res["alarm"][wt])
    #             # print(periods[j+3],res['distance'],res['alarm'])
    # plt.rcParams['font.sans-serif'] = ['simhei']  # 解决中文显示问题-设置字体为黑体
    # plt.rcParams['axes.unicode_minus'] = False
    # fig,ax=plt.subplots(2,1,figsize=(16,8))
    # for wt in output_wtids:
    #     ax[0].plot(res_index["time"],res_index["result"][wt],label="W0"+wt,marker=".")
    #     ax[1].plot(res_index["time"],res_index["alarm"][wt],label="W0"+wt,marker=".")
    #     ax[0].set_title("健康值")
    #     ax[0].set_ylim([0,110])
    #     ax[0].legend()
    #     ax[1].set_title("alarm")
    #     ax[1].set_ylim([-1,5])
    #     ax[1].legend()
    #     ax[0].grid()
    #     ax[1].grid()
    # plt.show()
    
    # data = pd.read_csv('../data/MY/converter.csv')
    # columns = ['t', 'active_power', 'cit', 'wt_id']
    # for i in columns[1:-1]:
    #     data[i] = data[i].astype('str').astype('float')
        
    
    # data.drop('Unnamed: 0',axis=1, inplace=True)
    # #s_date = datetime.datetime.strptime('20210701', '%Y%m%d').date()
    # #print(s_date)
    # #e_date = datetime.datetime.strptime('20210703', '%Y%m%d').date()
    
    # for i in range(28):
    #     if i < 7:
    #         st = '2021-06-0' + str(i+1)
    #         et = '2021-06-0' + str(i+3)
    #     elif i < 10 :
    #         st = '2021-06-0' + str(i+1)
    #         et = '2021-06-' + str(i+3)
    #     else:
    #         st = '2021-06-' + str(i+1)
    #         et = '2021-06-' + str(i+3)
    #     temp = data[(data['t'] >= st) & (data['t'] <= et)]    
    #     res = converter_temp_abnormal_main(temp)
    #     print(res['distance'])
    #     print(i)


    # 在线数据测试
    import json

    with open(r"C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\Converter_Temp_Abnormal\03_ConverterTempAbnormal\data\ConverterTempAbnormal-MY-data.json","r") as f:
        data  = json.load(f)
    res = pd.DataFrame(np.array(data),columns=["info"])
    for col in ['dataTimeStamp','active_power','IGBT_temp','assetName']:
        res[col] = res["info"].apply(lambda x:x.get(col,None))
    result = converter_temp_abnormal_main(res)
    print(result[4],result[5],result[6])