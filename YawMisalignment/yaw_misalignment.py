#!/usr/bin/env python
"""
@File    :   yaw_misalignment.py
@Time    :   2021/10/18
@Author  :   R XIE
@Version :   3.2.0
@License :   (C)Copyright 2019-2020, CyberInsight
@Desc    :  偏航对风不正识别
@Update  :   Doufengfeng，2022/10/24

"""



from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import gc
from decimal import *
from sklearn.ensemble import IsolationForest as iforest
import os
import datetime
import warnings
import datetime
warnings.filterwarnings('ignore')



# 上线
ws = "wind_speed"  #"风速实时值"
gp =  "power" #发电机功率实时值
yaw = "yaw_error"   #对风角，原来叫风向1秒平均值
ba =  "pitch_1"
ts = "time"  #时间
wt_status = 'main_status' # 风机运行状态
turbineName = 'turbineName'

yaw_left_edge = -180
yaw_right_edge = 180
min_data_need = 400 # 
alarm_threshold = '[6,8,10]'
wtstatus_n = [5]
has_wtstatus = "False"
cut_speed = 2.5
rated_speed = 10

plotflag = False

def distance_transform(value, x, y):
    if len(x) == len(y):
        lenx = len(x)
        if value < x[0]:
            value_t = 100
        elif (value >= x[0]) & (value < x[lenx - 1]):#
            itpl = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            value_t = float(itpl(value))
        else:
            value_t = 0
        return value_t 
    else:
        raise Exception

def to_alarm(value, alarm_threshold):
    alarm = None
    L = [0]+eval(alarm_threshold) + [eval(alarm_threshold)[2]+3]
    distance = distance_transform(abs(value),L,[100,80,60,40,0])#TODO 健康值对应进行了修改
    if abs(value) < eval(alarm_threshold)[0]:
        alarm = 0
        # distance = 100 - 20/eval(alarm_threshold)[0]* abs(value)# 
    elif (abs(value) >= eval(alarm_threshold)[0]) & (abs(value) < eval(alarm_threshold)[1]):
        alarm = 1
        # distance = 60 - 10*(abs(value)-eval(alarm_threshold)[0])
    elif (abs(value) >= eval(alarm_threshold)[1]) & (abs(value) < eval(alarm_threshold)[2]):
        alarm = 2
        # distance = 40 - 20/3*(abs(value)-eval(alarm_threshold)[1])
    elif abs(value) >= eval(alarm_threshold)[2]:
        alarm = 3
        # distance = np.max([0, 20 - 20/3*(abs(value)-eval(alarm_threshold)[2])])

    return alarm, distance


def slice_by_perc(data, var, upper, lower):
    """选择20%-80%分位数"""
    sdata = data.loc[(data[var] >= np.percentile(data[var], lower)) & (data[var] <= np.percentile(data[var], upper)), :]
    return sdata


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
    r = data.resample(scale).apply(np.nanmean)# TODO mean改为nanmean
    r[dt] = r.index
    return r


def loss_calc_precent(x):
    return (1 - (math.cos(abs(x * math.pi / 180))) ** 2) * 100


def smooth_data(data, var, yaw, var_u=5, var_l=0, yaw_w=0.2, bin_minnum_upper=50, bin_minnum_lower=10, std_limit=1, upper=90,
                lower=10, range_limit=1.5, method='iForest', contamination=0.01):
    '''
    
    '''
    data = data.loc[(data[var] > var_l) & (data[var] < var_u), :]
    breaks = np.arange(yaw_left_edge, yaw_right_edge, yaw_w)
    data['yaw_group'] = pd.cut(data[yaw], breaks)
    yawlist = pd.unique(data['yaw_group'].values)
    new_data = pd.DataFrame()

    if method == 'iForest':
        for i in yawlist:
            sdata = data.loc[data.yaw_group == i, :]
            if len(sdata) >= bin_minnum_upper:
                new_data = new_data.append(sdata)
        data = new_data
        # sns.scatterplot()
        if len(data)>0:
            x = data.loc[:, [yaw, var]].values
            model = iforest(max_samples=100, contamination=contamination,random_state=53)#TODO 增加了随机种子
            model.fit(x)
            data['label'] = model.predict(x)
            new_data = data.loc[data['label'] == 1, :]#将异常的数据去除，降噪
        else:
            new_data = data
    else:
        for i in yawlist:
            sdata = data.loc[data.yaw_group == i, :]
            if len(sdata) >= bin_minnum_upper:
                if method == 'A':
                    while (np.std(sdata[var].values) > std_limit) & (len(sdata) > bin_minnum_lower):
                        former_std = np.std(sdata[var].values)
                        sdata = slice_by_perc(sdata, var, upper, lower)
                        if np.std(sdata[var].values) >= former_std:
                            break
                    new_data = new_data.append(sdata)
                elif method == 'B':
                    window_width = max(sdata[var].values) - min(sdata[var].values)
                    window_center = np.median(sdata[var].values)
                    sdata['distance'] = sdata[var] - window_center
                    while (window_width >= range_limit) & (len(sdata) > bin_minnum_lower):
                        if sdata.loc[sdata['distance'] == max(sdata['distance']), var].values[0] > min(
                                sdata.loc[sdata['distance'] >= np.median(sdata['distance']), var].values):
                            sdata = slice_by_perc(sdata, var, upper, 0)
                        else:
                            sdata = slice_by_perc(sdata, var, 0, lower)
                        if len(sdata) < bin_minnum_lower:
                            sdata = pd.DataFrame()
                        else:
                            window_width = max(sdata[var].values) - min(sdata[var].values)
                            window_center = np.median(sdata[var].values)
                            sdata['distance'] = sdata[var] - window_center

                    new_data = new_data.append(sdata)
                else:
                    if len(sdata) <= bin_minnum_lower:
                        sdata = pd.DataFrame()
                    else:
                        sdata = sdata
                    new_data = new_data.append(sdata)
    data = new_data
    return data


def yaw_misalignment(data, ws=ws, gp=gp, yaw=yaw, time=ts, yaw_left_edge=-30, yaw_right_edge=30, min_data_need=1000,
                     smooth_mode='True', contamination=0.01, method='iForest'):
    """
    对风偏航不正识别模型
    :param data: 输入数据，dataframe
    :param ws: 风速变量名
    :param gp: 功率变量名
    :param yaw: 对风角变量名
    :param time: 实际变量名
    :param yaw_left_edge: 对风角度左边界，一般对风角数据为-180~180度数值，部分型号可能不同，
    通过此参数及以下右边界参数指定对风角度中值左右范围，此间的数据纳入计算。
    :param yaw_right_edge: 对风角度右边界
    :param min_data_need: 窗口最小数据量要求
    :param smooth_mode: 是否对数据进行预处理
    :param contamination: 异常点比例
    :param method: 预处理算法，默认Isolation Forest
    :return:
    """
    status_code = '000'
    data[ws] = data[ws].values.astype('float')
    data[gp] = data[gp].values.astype('float')
    data[yaw] = data[yaw].values.astype('float')
    data[time] = pd.to_datetime(data[time].values)
    start_time = datetime.datetime.strftime(min(data[time]),"%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strftime(max(data[time]),"%Y-%m-%d %H:%M:%S")

    # resampled = resample_bytime_mean(data, 1, time)

    resampled = data
    raw_data = dict()
    resampled = resampled.dropna()

    # if min(data[time]) > datetime.datetime.strptime('2022-3-30 0:00:00', '%Y-%m-%d %H:%M:%S'):
    #     a=1
    raw_data['time'] = list(map(str, pd.to_datetime(resampled[time].values.tolist())))
    raw_data['yaw'] = list(
        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled[yaw].values.tolist()))
    raw_data['windspeed'] = list(
        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled[ws].values.tolist()))
    raw_data['power'] = list(
        map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled[gp].values.tolist()))

    # 原始数据筛选
    data = data.loc[(data[ws] > cut_speed) & (data[ws] <= rated_speed) & (data[yaw] > yaw_left_edge) &
                    (data[yaw] < yaw_right_edge) & (data[gp] > 0), :]
    data['gpindex'] = data[gp].values / (data[ws].values ** 3)


    resampled['gpindex'] = resampled[gp].values / (resampled[ws].values ** 3)

    analysis_data = dict()
    analysis_data['online_x'] = []
    analysis_data['online_x2'] = []
    analysis_data['online_y'] = []
    analysis_data['curve_x'] = []
    analysis_data['curve_y'] = []
    analysis_data['vline'] = None
    analysis_data['threshold'] = []


    if len(data) > min_data_need:
        if smooth_mode == 'True':
            data = smooth_data(data, 'gpindex', yaw, var_u=10, var_l=0, yaw_w=1, bin_minnum_upper=50,
                               bin_minnum_lower=5,
                               std_limit=1,
                               upper=80, lower=20, range_limit=2, method=method,
                               contamination=contamination)#正常增加了yaw_group和iforest的判断label
        else:
            data = data

        # print(data.shape)
        # print(data.head())

        if len(data) > 0:
            wsbreaks = np.arange(np.ceil(cut_speed), np.floor(rated_speed), 0.5)#TODO 为什么是[4,8]?这里比较复杂
            data['ws_group'] = pd.cut(data[ws], wsbreaks)
            wslist = pd.unique(data['ws_group'].values)

            curve_collector, maxwdir_collector = dict(), dict()
            yawData2 = np.arange(min(data[yaw]), max(data[yaw]), 0.5)

            for wsg in wslist:#对风角和风速
                sdata = data.loc[data['ws_group'] == wsg, :] # 没有值 ==None
                if len(sdata) > 0:
                    max_gpindex_collector = []
                    yaw_collector = []
                    for yaw_value in yawData2:
                        sdata2 = sdata.loc[(sdata[yaw] >= yaw_value) & (sdata[yaw] < yaw_value + 0.5), :]
                        if len(sdata2) > 0:
                            max_gpindex = np.median(
                                sdata2.loc[sdata2['gpindex'] >= np.percentile(sdata2['gpindex'], 99), 'gpindex'].values)
                            #大于99分位数的风能利用系数中位数
                            max_gpindex_collector.append(max_gpindex)
                            yaw_collector.append(yaw_value)
                        else:
                            max_gpindex_collector.append(-1)
                            yaw_collector.append(yaw_value)

                    yaw_gp_d = pd.DataFrame({'yaw': yaw_collector, 'gpindex': max_gpindex_collector})
                    yaw_gp_d = yaw_gp_d.loc[yaw_gp_d['gpindex'] != -1, :]
                    
                    if (len(yaw_collector) > 0) & (np.std(
                            [maxgp for maxgp in max_gpindex_collector if (maxgp != -1) & (~np.isnan(maxgp))]) >= 0.2):
                        lowess = sm.nonparametric.lowess
                        model = pd.DataFrame(
                            lowess(np.array(yaw_gp_d['gpindex'].values),
                                   np.array(yaw_gp_d['yaw'].values))).drop_duplicates()
                        itpl = interp1d(model.iloc[:, 0], model.iloc[:, 1], bounds_error=False,
                                        fill_value='extrapolate')#extrapolate,外延
                        itpl_data = itpl(yawData2)
                        
                        itpld = pd.DataFrame({'Wdir': yawData2, 'PowerIndex': itpl_data})
                        maxpred = max(itpl_data)
                        
                        maxwdir = itpld.loc[itpld.PowerIndex == maxpred, 'Wdir'].values[0]

                        curve_collector[wsg] = itpl_data
                        maxwdir_collector[wsg] = maxwdir

            maxwdir_list = list(maxwdir_collector.values())
            key_list = list(maxwdir_collector.keys())
            while np.std(maxwdir_list) > 5:
                #不同风速bin选出的maxwdir标准差大，把距离中位数大的maxwdir舍弃
                distance = [x - np.median(maxwdir_list) for x in maxwdir_list]

                if abs(min(distance)) <= abs(max(distance)):
                    for i in range(len(maxwdir_list)):
                        if maxwdir_list[i] == max(maxwdir_list):
                            maxwdir_collector.pop(key_list[i])
                            curve_collector.pop(key_list[i])
                else:
                    for i in range(len(maxwdir_list)):
                        if maxwdir_list[i] == min(maxwdir_list):
                            maxwdir_collector.pop(key_list[i])
                            curve_collector.pop(key_list[i])
                maxwdir_list = list(maxwdir_collector.values())
                key_list = list(maxwdir_collector.keys())
                
            
            if len(maxwdir_list) > 1:#评估可靠性等级，并把数据赋值给analysis_data
                plot_y = np.median(np.array(list(curve_collector.values())), axis=0)#取不同风速下gpindex的中位数
                itpld = pd.DataFrame({'Wdir': yawData2, 'PowerIndex': plot_y})

                # ################Plot 最终结果
                # plt.plot(itpld['Wdir'], itpld['PowerIndex'], color='black', linestyle='--', label='拟合曲线')
                # plt.xlabel('偏航角')
                # plt.ylabel('Cp_特征值')
                # plt.legend()
                # plt.title(data[turbineName][0] + '_' + start_time[0:10])
                #
                # # plt.show()
                # savePath = '../Result/yaw_misalignment/{}/'.format(data[turbineName][0])
                #
                # if not os.path.exists(savePath):
                #     os.makedirs(savePath)
                # plt.savefig(savePath + start_time[0:10] + 'yaw.jpg', format='jpg', dpi=plt.gcf().dpi,    bbox_inches='tight')
                # plt.clf()


                if len(maxwdir_list) >= 2:
                    maxwdir2 = float(Decimal(np.median(maxwdir_list)).quantize(Decimal('0.0000')))

                    if ((np.std(maxwdir_list) < 1) & (len(maxwdir_list) > 0.3 * len(wsbreaks)) & ((max(maxwdir_list) != max(yawData2)) & (min(maxwdir_list) != min(yawData2)))):
                        if (max(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'Wdir']) - \
                            min(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'Wdir'])  \
                            <= max(int(len(itpld) * 0.1), 3) * 0.5):
                            if (max(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1),3)]), 'PowerIndex']) - 
                                min(itpld.loc[itpld[ 'PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'PowerIndex']) >  
                                0.002 * max(int(len(itpld) * 0.1), 3)):
                                
                                maxwdir = float(Decimal( max(itpld.loc[itpld['PowerIndex'] == max(itpld['PowerIndex']), 'Wdir'])).quantize(
                                    Decimal('0.0000')))

                                if abs(maxwdir2) > abs(maxwdir):
                                    maxwdir = maxwdir
                                else:
                                    maxwdir = maxwdir2
                                r_rank = 'A'
                            else:
                                maxwdir = float(Decimal(np.median(itpld.loc[itpld['PowerIndex'] > max(
                                    itpld['PowerIndex'].values) * 0.99, 'Wdir'])).quantize(Decimal('0.0000')))
                                if abs(maxwdir2) > abs(maxwdir):
                                    maxwdir = maxwdir
                                else:
                                    maxwdir = maxwdir2
                                r_rank = 'B'
                        else:
                            maxwdir = float(Decimal(np.median(itpld.loc[itpld['PowerIndex'] > max(
                                itpld['PowerIndex'].values) * 0.99, 'Wdir'])).quantize(Decimal('0.0000')))
                            r_rank = 'D'
                            status_code = '200' # 100修改为200
                    elif (((np.std(maxwdir_list) >= 1) & (np.std(maxwdir_list) < 3) & (len(maxwdir_list) > 0.1 * len(wsbreaks))) | (
                                (np.std(maxwdir_list) < 1.5) & (len(maxwdir_list) <= 0.3 * len(wsbreaks))) & (
                                (max(maxwdir_list) != max(yawData2)) & (min(maxwdir_list) != min(yawData2)))):
                        if (max(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'Wdir']) 
                                - min(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'Wdir']) 
                                <= max(int(len(itpld) * 0.1), 3) * 0.5):
                            if (max(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1),3)]), 'PowerIndex']) 
                                - min(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'PowerIndex']) 
                                > 0.002 * max(int(len(itpld) * 0.1), 3)):
                                
                                maxwdir = float(Decimal(max(itpld.loc[itpld['PowerIndex'] == max(itpld['PowerIndex']), 'Wdir'])).quantize(
                                    Decimal('0.0000')))
                                if abs(maxwdir2) > abs(maxwdir):
                                    maxwdir = maxwdir
                                else:
                                    maxwdir = maxwdir2
                                r_rank = 'B'
                            elif ((max(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'Wdir']) - 
                                   min(itpld.loc[itpld['PowerIndex'] > min(itpld['PowerIndex'].sort_values(ascending=False)[0:max(int(len(itpld) * 0.1), 3)]), 'Wdir']) 
                                   <= max(int(len(itpld) * 0.1), 12) * 0.5) & (np.std(maxwdir_list) < 2.5)):
                                maxwdir = float(Decimal(np.median(itpld.loc[itpld['PowerIndex'] > max(itpld['PowerIndex'].values) * 0.99, 'Wdir'])).quantize(Decimal('0.0000')))
                                if abs(maxwdir2) > abs(maxwdir):
                                    maxwdir = maxwdir
                                else:
                                    maxwdir = maxwdir2
                                r_rank = 'B'
                            else:
                                maxwdir = float(Decimal(np.median(itpld.loc[itpld['PowerIndex'] > max(
                                    itpld['PowerIndex'].values) * 0.99, 'Wdir'])).quantize(Decimal('0.0000')))
                                if abs(maxwdir2) > abs(maxwdir):
                                    maxwdir = maxwdir
                                else:
                                    maxwdir = maxwdir2
                                r_rank = 'D'
                                status_code = '200' # 100修改为200
                        else:
                            maxwdir = float(Decimal(np.median(itpld.loc[itpld['PowerIndex'] > max(
                                itpld['PowerIndex'].values) * 0.99, 'Wdir'])).quantize(Decimal('0.0000')))
                            if abs(maxwdir2) > abs(maxwdir):
                                maxwdir = maxwdir
                            else:
                                maxwdir = maxwdir2
                            r_rank = 'D'
                            status_code = '200' # 100修改为200
                    else:
                        maxwdir = 0
                        r_rank = 'D'
                        status_code = '200' # 100修改为200
                else:
                    status_code = '200' # 100修改为200
                    maxwdir = float(Decimal(np.median(itpld.loc[itpld['PowerIndex'] > max(
                        itpld['PowerIndex'].values) * 0.99, 'Wdir'])).quantize(Decimal('0.0000')))
                    r_rank = 'D'

                if status_code == '000':
                    analysis_data = dict()
                    analysis_data['online_x'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled[yaw].values.tolist()))
                    analysis_data['online_x2'] = list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), list(map(loss_calc_precent, analysis_data['online_x']))))
                    try:
                        analysis_data['online_y'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled['gpindex'].values.tolist()))
                    except:
                        # for i in range(len(resampled['gpindex'].values.tolist())):
                        #     if i == 3133:
                        #         a=1
                        #     print (i, resampled['gpindex'].values.tolist()[i], float(Decimal(resampled['gpindex'].values.tolist()[i]).quantize(Decimal('0.0000'))))
                        # plt.scatter(resampled[yaw].values.tolist(), resampled['gpindex'].values.tolist())
                        # plt.show()
                        a=1


                    analysis_data['curve_x'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), yawData2.tolist()))
                    analysis_data['curve_y'] = list( map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), plot_y))
                    analysis_data['vline'] = float(Decimal(maxwdir).quantize(Decimal('0.0000')))
                    analysis_data['threshold'] = sorted(list(map(lambda x: -x, eval(alarm_threshold)))) + eval( alarm_threshold)
                else:
                    analysis_data = None

            else:
                #结果数据筛选不满足条件
                status_code = '302'
                maxwdir = None
                r_rank = None

        else:
            #平滑后数据为空
            status_code = '301'
            maxwdir = None
            r_rank = None


        result = dict()
        result['start_time'] = start_time
        result['end_time'] = end_time
        result['max_wdir'] = maxwdir
        result['rank'] = r_rank
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['status_code'] = status_code
    else:
        #数据筛选后数量不足
        status_code = '300'
        result = dict()
        result['start_time'] = start_time
        result['end_time'] = end_time
        result['max_wdir'] = None
        result['status_code'] = status_code
        result['rank'] = ''
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
    
      
    gc.collect()
    return result


def yaw_result_mixer(data, ws=ws, gp=gp,
                     yaw=yaw, time=ts,
                     yaw_left_edge=yaw_left_edge,
                     yaw_right_edge=yaw_right_edge,
                     min_data_need=min_data_need, smooth_mode='True', method='iForest'):

    result = yaw_misalignment(data, ws=ws, gp=gp,
                              yaw=yaw, time=time,
                              yaw_left_edge=yaw_left_edge,
                              yaw_right_edge=yaw_right_edge,
                              min_data_need=min_data_need, smooth_mode=smooth_mode, method=method)

    if result['status_code'] == '000':
        result['start_time'] = result['start_time']
        result['end_time'] = result['end_time']
        result['max_wdir'] = result['max_wdir']
        result['raw_data'] = result['raw_data']
        result['analysis_data'] = result['analysis_data']
    else:
        result['status_code'] = result['status_code']
        result['rank'] = None
        result['start_time'] = result['start_time']
        result['end_time'] = result['end_time']
        result['max_wdir'] = 0
        result['raw_data'] = result['raw_data']
        result['analysis_data'] = result['analysis_data']

    return result


# def yaw_misalignment_wrapper(data, ws, gp, yaw, time, yaw_left_edge, yaw_right_edge, min_data_need, ba,
#                              wtstatus_n, has_wtstatus, alarm_threshold):
# 直接把这个改成main函数
def yaw_misalignment_main(data):
    '''
    status_code: 200-输入data文件为空
    '''
    data = data_preprocess(data) # TODO 新增数据处理函数
    # print(data.shape)
    if len(data)>0:
        data = data.dropna(subset=[ws, gp, yaw, ts])
        if len(data) > 0:
            data[ts] = pd.to_datetime(data[ts].values)
            start_time = str(min(data[ts]))
            end_time = str(max(data[ts]))

            if has_wtstatus == 'True':
                data = data[data[wt_status].isin(wtstatus_n)]
            else:
                data = data.loc[(data[ba] < 5) & (data[ws]>0), :]   ###增加一个风速>0
            # print(data.head())
            # print(data.shape)
            # print(6666)
            if len(data) > 0:
                if np.nanmax(data[yaw]) > 180:
                    data.loc[data[yaw] >= 180, yaw] = data.loc[data[yaw] >= 180, yaw] - 360
                result = yaw_result_mixer(data, ws=ws, gp=gp,
                                          yaw=yaw, time=ts,
                                          yaw_left_edge=yaw_left_edge,
                                          yaw_right_edge=yaw_right_edge,
                                          min_data_need=min_data_need, smooth_mode='False', method='iForest')


                if result['max_wdir'] is not None:
                    alarm, distance = to_alarm(result['max_wdir'], alarm_threshold)
                    result['alarm'] = alarm
                    result['distance'] = distance
                else:
                    result['alarm'] = None
                    result['distance'] = None
            else:
                #筛选后数据为空
                status_code = '300'
                resampled = resample_bytime_mean(data, 1, ts)
                raw_data = dict()
                resampled = resampled.dropna()
                raw_data['time'] = list(map(str, pd.to_datetime(resampled[ts].values.tolist())))
                raw_data['yaw'] = list(
                    map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled[yaw].values.tolist()))
                raw_data['windspeed'] = list(
                    map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled[ws].values.tolist()))
                raw_data['power'] = list(
                    map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), resampled[gp].values.tolist()))

                result = dict()
                result['start_time'] = start_time
                result['end_time'] = end_time
                result['max_wdir'] = None
                result['status_code'] = status_code
                result['rank'] = None
                result['raw_data'] = raw_data
                result['analysis_data'] = None
                result['alarm'] = None
                result['distance'] = None
        else:
            #输入数据指定变量为空
            status_code = '200' #

            result = dict()
            result['start_time'] = None
            result['end_time'] = None
            result['max_wdir'] = None
            result['alarm'] = None
            result['distance'] = None
            result['status_code'] = status_code
            result['rank'] = None
            result['raw_data'] = None
            result['analysis_data'] = None
    else:
        #输入数据为空
        status_code = '200' #

        result = dict()
        result['start_time'] = None
        result['end_time'] = None
        result['max_wdir'] = None
        result['alarm'] = None
        result['distance'] = None
        result['status_code'] = status_code
        result['rank'] = None
        result['raw_data'] = None
        result['analysis_data'] = None

    
    print(result['max_wdir'], result['distance'], result['alarm'], result['status_code'],result["rank"])
    if (result['status_code']=='000') & (abs(float(result['max_wdir']))>3):
        # ################Plot 最终结果
        plt.scatter(result['analysis_data']['online_x'],result['analysis_data']['online_y'] , s=1)
        plt.plot(result['analysis_data']['curve_x'], result['analysis_data']['curve_y'], color='black')
        plt.axvline(result['max_wdir'], color='pink', label='max_wdir='+ str(result['max_wdir']))
        plt.ylim(0, 10)
        plt.xlim(-30,30)
        plt.xlabel('偏航角')
        plt.ylabel('Cp_特征值')
        plt.legend()
        plt.title(data[turbineName][0] + '_' + start_time[0:10])

        # plt.show()
        savePath = '../Result/yaw_misalignment/'

        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(savePath + data[turbineName][0] + '_'+start_time[0:10] + 'yaw.jpg', format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
        plt.clf()


        pass

    

    return (result["start_time"],result["end_time"],result["raw_data"],result["analysis_data"], result["status_code"],result["max_wdir"],result["alarm"],result["distance"],result["rank"])

def data_preprocess(data):
    columns = [ts, turbineName, ws, yaw, ba, gp]
    data = data[columns]

    data[ba].fillna(method = "ffill",inplace=True) 
    data[ba].fillna(method = "bfill",inplace=True)
        
    return data

def result_plot(result,res_index):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    
    plt.rcParams['font.sans-serif'] = ['simhei']  # 解决中文显示问题-设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(constrained_layout=True,figsize=(16,8))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    ax2_2 = ax2.twinx()
    
    raw_time = list(map(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"),result["raw_data"]["time"]))
    ax1.plot(raw_time,result["raw_data"]["yaw"])
    ax1.set_title("raw data:对风角-时间")
    ax1.set_xlabel(result['status_code'])
    ax1.grid()
    ax2.plot(raw_time,result["raw_data"]["windspeed"],label="风速")
    ax2_2.plot(raw_time,result["raw_data"]["power"],color="orange",label="功率")
    ax2.legend(loc="lower left")
    ax2_2.legend(loc="lower right")
    ax2.set_title("raw data:风速/功率-时间")
    ax2.grid()
    
    if result['analysis_data'] and result['analysis_data']["online_x"]:
        thresholds = [np.min(result["analysis_data"]["online_x"])]+result["analysis_data"]["threshold"] \
        + [np.max(result["analysis_data"]["online_x"])]
        y_lim = [np.min(result["analysis_data"]["online_y"]),np.max(result["analysis_data"]["online_y"])]
        ax3.fill_betweenx(y=y_lim,x1=thresholds[0],x2=thresholds[1],color="orange")
        ax3.fill_betweenx(y=y_lim,x1=thresholds[6],x2=thresholds[7],color="orange")
        ax3.fill_betweenx(y=y_lim,x1=thresholds[1],x2=thresholds[2],color="peachpuff")
        ax3.fill_betweenx(y=y_lim,x1=thresholds[5],x2=thresholds[6],color="peachpuff")
        ax3.fill_betweenx(y=y_lim,x1=thresholds[2],x2=thresholds[3],color="yellow")
        ax3.fill_betweenx(y=y_lim,x1=thresholds[4],x2=thresholds[5],color="yellow")
        ax3.fill_betweenx(y=y_lim,x1=thresholds[3],x2=thresholds[4],color="lightgreen")
        
        ax3.scatter(result["analysis_data"]["online_x"],result["analysis_data"]["online_y"],s=3)
        ax3.plot(result["analysis_data"]["curve_x"],result["analysis_data"]["curve_y"],c="r")
        ax3.axvline(result["analysis_data"]["vline"],linestyle = "--",color="r")
        ax3.set_title(f"偏差角度：{result['max_wdir']},预警等级：{result['alarm']},健康值：{result['distance']},结果可靠等级：{result['rank']}")
        ax3.set_ylabel("风能利用指数")
    ax0.plot(res_index["time"],res_index["result"],marker="D",markerfacecolor="cyan",label="偏差角度")
    ax0_1 = ax0.twinx()
    ax0_1.plot(res_index["time"],res_index["distance"],marker="o",markerfacecolor="lime",label="健康值",c="orange")
    ax0.set_ylabel("偏差角度")
    ax0.legend(loc="upper left")
    ax0_1.set_ylabel("健康值")
    ax0_1.legend(loc="upper right")
    ax0.grid()
    plt.show()

# def yaw_misalignment_main(data):
#     global ws
#     global gp
#     global yaw
#     global time
#     global yaw_left_edge
#     global yaw_right_edge
#     global min_data_need
#     global alarm_threshold
#     global ba
#     global wt_status
#     global wtstatus_n
#     global has_wtstatus
#     global cut_speed
#     global rated_speed
#
#     return yaw_misalignment_wrapper(data, ws, gp, yaw, time, yaw_left_edge, yaw_right_edge, min_data_need, ba,
#                                     wtstatus_n, has_wtstatus, alarm_threshold)





if __name__ == '__main__':

#     offline = True    
#     plotflag = True

#     folder = "../../data" if isJFWT else "../../data/MY"

#     for i in ["W085"]:#"W056","W070","W085"  MY: "W019","W031","W010"
#         print("*"*50,i,"*"*50)
#         data = pd.read_csv(os.path.join(folder,i+".csv"),nrows=200000)
#         data["time"] = pd.to_datetime(data.time) + datetime.timedelta(hours=8)
#         data.set_index("time",drop=False,inplace=True)
#         columns = [time, ws, yaw, ba, gp, wt_status] if has_wtstatus == "True" else [time, ws, yaw, ba, gp]  
#         data = data[columns]
        
#         periods = pd.date_range(start = data["time"].min(),end=data.time.max(),freq = '1D',normalize=True)#
        
#         res_index ={"time":[],"result":[],"distance":[]}
#         for j in range( len(periods)-2 ):#       144,145   len(periods)-28
#             data_ = data[periods[j]:periods[j+2]] 
#             print("*"*10,j,pd.to_datetime(periods[j+2]),end="    ")#
#             res = yaw_misalignment_main(data_)
#             res_index["time"].append(pd.to_datetime(periods[j+2]))
#             res_index["result"].append(res["max_wdir"])
#             res_index["distance"].append(res["distance"])
#             print(res["start_time"])
#         if plotflag:
#             result_plot(res,res_index)
    
    # # 在线数据测试
    columns = [ts, ws, yaw, ba, gp, wt_status]
    # import json
    # with open(r"D:\Code\Work\风电\大唐EPS\偏航对风不正\YawMisalignment-85-data.json","r") as f:
    #     data  = json.load(f)
    # res = pd.DataFrame(np.array(data),columns=["info"])
    # for col in columns:
    #     res[col] = res["info"].apply(lambda x:x.get(col,None))
    # result = yaw_misalignment_main(res)
    # print(result[-5:])
    # print(to_alarm(3, alarm_threshold))

    # mongo在线数据测试
    import pymongo
    conn = pymongo.MongoClient(host="192.168.199.243", port=27017)
    # conn.admin.authenticate()
    db = conn['datang_offshore']
    collection = db['windinsight_eps_5s']
    yaw_res = []
    month = 9
    i = 19
    tb=85
    st = f"2022-0{month}-{'%02d'%i} 00:00:03"
    et = f"2022-0{month}-{'%02d'%(i+1)} 23:59:58"
    wt = f'{tb}'
    res = collection.find({"dataTimeStamp": {"$gt": st, "$lt": et}, "assetName": wt},)

    raw_data = pd.DataFrame(list(res))
    data_wt = raw_data[columns]
    result = yaw_misalignment_main(data_wt)
    print(result[-5:])

    # for tb in [56,70,85]:
    #     try:
    #         for month in [7,8,9]:
    #             for i in range(1,12):
    #             # for i in [13]:
    #                 st = f"2022-0{month}-{'%02d'%i} 00:00:03"
    #                 et = f"2022-0{month}-{'%02d'%(i+1)} 23:59:58"
    #                 wt = f'{tb}'
    #                 res = collection.find({"dataTimeStamp": {"$gt": st, "$lt": et}, "assetName": wt},)
    #                 # conn.close()
    #                 raw_data = pd.DataFrame(list(res))
    #                 data_wt = raw_data[columns]
    #                 print("-"*20)
    #                 print(st,et,wt)
    #                 print(data_wt.shape)
    #                 print(data_wt.head())
    #                 print(data_wt.tail())
    #                 result = yaw_misalignment_main(data_wt)
    #                 # print(result)
    #                 print(result[-5:])
    #                 yaw_res.append({'st':st, 'et':et, 'wt':wt,
    #                             'status_code':result[-5],
    #                             'max_wdir':result[-4],
    #                             'alarm':result[-3],
    #                             'distance':result[-2],
    #                            'rank':result[-1]})
    #     except Exception as e:
    #         print(e)
    # df = pd.DataFrame(yaw_res)
    # df.to_csv('./yaw_res.csv')