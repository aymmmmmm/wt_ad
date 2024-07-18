"""
变桨电机温度异常
"""

import numpy as np
# from numpy.lib.financial import rate
import pandas as pd
from PublicUtils.util_datadiscretization import data_discretization
from PublicUtils.util_tfidf import tfidf_by_window, make_dist_df
import pickle
import os
import math
import datetime
import statsmodels.api as sm
from scipy.interpolate import interp1d
from decimal import *
import warnings

warnings.filterwarnings('ignore')

isJFWT = False  # 是否是金风风机
online = True  # 在线运行

window = 1440
k = [0, 3, 4, 5, 6]

# 在线
pbt1 = 'pitch_motor_temp_1'
pbt2 = 'pitch_motor_temp_2'
pbt3 = 'pitch_motor_temp_3'
time = 'time'  # 时间变量名
wind_speed = 'wind_speed'
active_power = 'power'
main_status = 'main_status'
ba1_speed = "pitch_speed_1"
ba2_speed = "pitch_speed_2"
ba3_speed = "pitch_speed_3"


temp_threshold1 = 100  # 实际没有用到
temp_threshold = 10
diff_upper_limit = 10
diff_lower_limit = -10
bin_size = 0.5
has_wtstatus = "False"
wtstatus_n = [1, 4, 5] if isJFWT else [8, 9, 12, 13, 14, 15, 32]  # TODO 有变桨动作的状态,不再需要，用变桨速度选择


# rated_speed = 10 if isJFWT else 9.3

def data_preprocess(data):  # TODO 新增预处理函数
    # for v in [pbt1,pbt2,pbt3,wind_speed,main_status]:
    data[[pbt1, pbt2, pbt3, wind_speed]].interpolate(method='linear', limit=1500, axis=0, inplace=True)
    data.fillna(method="ffill", inplace=True, limit=1000)
    data.fillna(method="bfill", inplace=True, limit=1000)
    return data.dropna()


def import_data_check(data):
    '''
    数据质量检查：
    '''
    status_code= '000'
    if data is None:
        # raise BaseException('input data is None')  # 判断输入数据是否为None
        status_code = '300'
    elif len(data) == 0:
        # raise BaseException('input data is empty')  # 判断输入数据是否为空
        status_code = '300'
    elif data.isnull().all().any():
        # raise BaseException('column of input data is empty')  # 判断数据某一列是否为空
        status_code = '300'

    # if  data.dropna().shape[0] < 1440:
    if data.notna().sum().min() < 1440:  # TODO 避免不同列数据缺失不同步，可能导致dropna后数据太少
        # raise BaseException('the length of data is not enough')  # 输入数据不足
        status_code = '301'

    if (data[[pbt1, pbt2, pbt3, wind_speed]].std() == 0).any():
        # raise BaseException('column of input data has repeated value')  # 判断数据某列数值是否全部重复
        status_code = '302'

    return data, status_code


def distance_transform(value, x, y):
    # print(value, x, y)
    if len(x) == len(y):
        lenx = len(x)
        if value < x[0]:
            value_t = 0
        elif (value >= x[0]) & (value < x[lenx - 1]):  # u到u+6sigma之间
            itpl = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            value_t = float(itpl(value))
        else:
            value_t = 100
        return 100 - value_t  # 所以是数值越高越健康
    else:
        raise Exception


def distance_to_alarm(distance):
    if 0 <= distance <= 40:
        alarm = 3
    elif 40 < distance <= 60:
        alarm = 2
    elif 60 < distance <= 80:
        alarm = 1
    else:
        alarm = 0
    return alarm


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


def alarm_integrate(alarm1, alarm2, alarm3):
    alarm = 0

    if alarm1 > 0:
        if alarm2 > 0 or alarm3 > 0:
            alarm = int(np.nanmax([alarm1, alarm2, alarm3]))
    else:
        if alarm2 > 0 and alarm3 > 0:
            alarm = int(np.nanmax([alarm1, alarm2, alarm3]))

    return alarm


def pitch_motor_tempunnormal(
        data,
        compvec,
        idf,
        window,
        pbt1,
        pbt2,
        pbt3,
        time,
        wind_speed,
        active_power,
        threshold,
        bin_size,
        diff_upper_limit,
        diff_lower_limit,
        base_plot):
    """

    :param data:
    :param compvec:
    :param idf:
    :param window:
    :param pbt1:
    :param pbt2:
    :param pbt3:
    :param time:
    :param threshold:
    :param bin_size:
    :param diff_upper_limit:
    :param diff_lower_limit:
    :param base_plot:
    :return:
    """
    lstime = str(max(data[time]))
    stime = str(min(data[time]))
    result = dict()

    raw_d = data
    raw_data = dict()
    raw_data['datetime'] = list(
        map(str, pd.to_datetime(raw_d[time].values.tolist())))
    raw_data['pbt1'] = list(map(lambda x: float(
        Decimal(x).quantize(Decimal('0.0000'))), raw_d[pbt1].values.tolist()))
    raw_data['pbt2'] = list(map(lambda x: float(
        Decimal(x).quantize(Decimal('0.0000'))), raw_d[pbt2].values.tolist()))
    raw_data['pbt3'] = list(map(lambda x: float(
        Decimal(x).quantize(Decimal('0.0000'))), raw_d[pbt3].values.tolist()))
    raw_data['ws'] = list(map(lambda x: float(Decimal(x).quantize(
        Decimal('0.0000'))), raw_d[wind_speed].values.tolist()))
    raw_data['gp'] = list(map(lambda x: float(Decimal(x).quantize(
        Decimal('0.0000'))), raw_d[active_power].values.tolist()))

    # data = data.loc[(data[main_status].isin(wtstatus_n)) & (data[wind_speed] > (rated_speed-1))]
    data = data.loc[
        (data[ba1_speed].abs() > 0) | (data[ba2_speed].abs() > 0) | (data[ba3_speed].abs() > 0)]  # TODO 选择数据条件改变

    if len(data) > 0:
        status_code = '000'
        # 去除变桨电机温度中的噪音点
        data[time] = pd.to_datetime(data[time])
        data = data.loc[(data[pbt1] > temp_threshold) & (
                data[pbt2] > temp_threshold) & (data[pbt3] > temp_threshold)]
        # 计算变桨电机温度之差
        data['temp_diff12'] = data[pbt1] - data[pbt2]
        data['temp_diff13'] = data[pbt1] - data[pbt3]
        data['temp_diff23'] = data[pbt2] - data[pbt3]

        for col in ['temp_diff12', 'temp_diff13', 'temp_diff23']:
            data.loc[(data[col] > diff_upper_limit), col] = diff_upper_limit
            data.loc[(data[col] < diff_lower_limit), col] = diff_lower_limit
        # data[time] = pd.to_datetime(data[time])
        if len(data) > int(len(raw_d) * 0.05):
            data['temp_diff_bin12'] = data_discretization(
                data.temp_diff12,
                mode='type2',
                minnum=diff_lower_limit - bin_size,
                maxnum=diff_upper_limit + bin_size,
                binsize=bin_size)
            data['temp_diff_bin13'] = data_discretization(
                data.temp_diff13,
                mode='type2',
                minnum=diff_lower_limit - bin_size,
                maxnum=diff_upper_limit + bin_size,
                binsize=bin_size)
            data['temp_diff_bin23'] = data_discretization(
                data.temp_diff23,
                mode='type2',
                minnum=diff_lower_limit - bin_size,
                maxnum=diff_upper_limit + bin_size,
                binsize=bin_size)
            tfidf12 = tfidf_by_window(
                data, 'temp_diff_bin12', time, idf, window)
            tfidf13 = tfidf_by_window(
                data, 'temp_diff_bin13', time, idf, window)
            tfidf23 = tfidf_by_window(
                data, 'temp_diff_bin23', time, idf, window)
            tfidf_distance12 = make_dist_df(
                tfidf12, compvec, 'DateTime', 'TF-IDF', mode='B')
            tfidf_distance13 = make_dist_df(
                tfidf13, compvec, 'DateTime', 'TF-IDF', mode='B')
            tfidf_distance23 = make_dist_df(
                tfidf23, compvec, 'DateTime', 'TF-IDF', mode='B')

            kde12 = sm.nonparametric.KDEUnivariate(data['temp_diff12'].values)
            kde12.fit(bw=1)
            kde13 = sm.nonparametric.KDEUnivariate(data['temp_diff13'].values)
            kde13.fit(bw=1)
            kde23 = sm.nonparametric.KDEUnivariate(data['temp_diff23'].values)
            kde23.fit(bw=1)
            analysis_data = dict()
            analysis_data['density_x12'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), kde12.support))
            analysis_data['density_y12'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), kde12.density))
            analysis_data['density_x13'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), kde13.support))
            analysis_data['density_y13'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), kde13.density))
            analysis_data['density_x23'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), kde23.support))
            analysis_data['density_y23'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), kde23.density))
            analysis_data['base_x'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), base_plot['support']))
            analysis_data['base_y'] = list(map(lambda x: float(
                Decimal(x).quantize(Decimal('0.0000'))), base_plot['density']))
            result['start_time'] = str(stime)
            result['end_time'] = str(lstime)
            result['result1'] = float(
                Decimal(
                    distance_transform(
                        tfidf_distance12.distance.mean(), threshold, [0, 20, 40, 60, 100])).quantize(
                    Decimal('0.0000')))  # TODO distance[0]改为distance.mean(),当实际数据不是一个window时取均值
            result['result2'] = float(
                Decimal(
                    distance_transform(
                        tfidf_distance13.distance.mean(), threshold, [0, 20, 40, 60, 100])).quantize(
                    Decimal('0.0000')))
            result['result3'] = float(
                Decimal(
                    distance_transform(
                        tfidf_distance23.distance.mean(), threshold, [0, 20, 40, 60, 100])).quantize(
                    Decimal('0.0000')))

            print(result["result1"], result['result2'], result["result3"])
            a, b, c = result["result1"], result['result2'], result["result3"]
            result["result1"] = np.mean([a, b])
            result["result2"] = np.mean([a, c])
            result["result3"] = np.mean([b, c])

            result['alarm1'] = distance_to_alarm(result['result1'])
            result['alarm2'] = distance_to_alarm(result['result2'])
            result['alarm3'] = distance_to_alarm(result['result3'])
            result['alarm'] = alarm_integrate(
                result['alarm1'], result['alarm2'], result['alarm3'])

            result['status_code'] = status_code
            result['raw_data'] = raw_data
            result['analysis_data'] = analysis_data
        else:
            status_code = '300'  # 输入数据经过过滤后数据量太少
            result['start_time'] = str(stime)
            result['end_time'] = str(lstime)
            result['result1'] = 100
            result['result2'] = 100
            result['result3'] = 100
            result['status_code'] = status_code
            result['raw_data'] = raw_data
            result['analysis_data'] = None
            result['alarm1'] = 0
            result['alarm2'] = 0
            result['alarm3'] = 0
            result['alarm'] = 0
    else:
        status_code = '200'
        result['start_time'] = str(stime)
        result['end_time'] = str(lstime)
        result['result1'] = None
        result['result2'] = None
        result['result3'] = None
        result['status_code'] = status_code
        result['raw_data'] = raw_data
        result['analysis_data'] = None
        result['alarm1'] = None
        result['alarm2'] = None
        result['alarm3'] = None
        result['alarm'] = None




    return result


def pitch_motor_temp_abnormal_main(data, model_compvec, model_idf, model_tf_idf, model_kde_plot):
    global window
    global k
    global pbt1
    global pbt2
    global pbt3
    global time
    global wind_speed
    global active_power
    global diff_lower_limit
    global diff_upper_limit
    global bin_size
    global main_status
    global temp_threshold
    global wtstatus_n
    global has_wtstatus
    global rated_speed

    if isinstance(k, str):
        k = eval(k)

    columns = [pbt1, pbt2, pbt3, active_power, wind_speed, time]

    wtid= data['turbineName'][0]

    data, status_code = import_data_check(data)
    data = data_preprocess(data)


    try:

        compvec = model_compvec
        idf = model_idf
        tfidf_distance = model_tf_idf
        base_plot = model_kde_plot



        idf['Word'] = list(map(str, idf['Word'].values))
        mean_train = np.mean(tfidf_distance['distance'].values)
        std_train = np.std(tfidf_distance['distance'].values)
        threshold = []
        for i in k:
            threshold.append(mean_train + i * std_train)

    except BaseException:
        raise Exception('No Such File or Directory')

    data[time] = pd.to_datetime(data[time])
    data = resample_bytime_mean(data, 1, time)
    data[time] = data.index

    result = pitch_motor_tempunnormal(
        data,
        compvec,
        idf,
        window,
        pbt1,
        pbt2,
        pbt3,
        time,
        wind_speed,
        active_power,
        threshold,
        bin_size,
        diff_upper_limit,
        diff_lower_limit,
        base_plot)
    print(result['start_time'], result['result1'], result['result2'], result['result3'])



    if result['alarm']>0:

        plt.plot(result["analysis_data"]["base_x"], result["analysis_data"]["base_y"], c="r", label="base")
        plt.plot(result["analysis_data"]["density_x12"], result["analysis_data"]["density_y12"], "--b",                 label="diff12")
        plt.plot(result["analysis_data"]["density_x13"], result["analysis_data"]["density_y13"], "--y",                 label="diff13")
        plt.plot(result["analysis_data"]["density_x23"], result["analysis_data"]["density_y23"], "--g",                 label="diff23")
        plt.ylabel("温差特征值")
        plt.legend()
        plt.title('dis1_'+str(result['result1'])+ 'dis2_'+str(result['result2'])+'dis3_'+str(result['result3']))

        # plt.show()
        savePath = '../Result/pitch_motor_temp_abnormal/fault/'

        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(savePath + wtid + '_' + str(result['start_time'])[0:10] + 'pitchMotor_temp.jpg', format='jpg',
                    dpi=plt.gcf().dpi, bbox_inches='tight')
        plt.clf()


    return (    result["raw_data"], result["analysis_data"], result["status_code"], result["start_time"], result["end_time"],
    result["alarm"], result["alarm1"], result["alarm2"], result["alarm3"], result["result1"],
    result["result2"], result["result3"])


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def result_plot(result, res_index):
    plt.rcParams['font.sans-serif'] = ['simhei']  # 解决中文显示问题-设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    ax2_2 = ax2.twinx()

    raw_time = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), result["raw_data"]["datetime"]))
    ax1.plot(raw_time, result["raw_data"]['pbt1'], label="变桨电机1")
    ax1.plot(raw_time, result["raw_data"]['pbt2'], label="变桨电机2")
    ax1.plot(raw_time, result["raw_data"]['pbt3'], label="变桨电机3")
    ax1.set_title("raw data：变桨电机温度")
    ax1.set_xlabel(result['status_code'])
    ax1.grid()
    ax1.legend()

    ax2.plot(raw_time, result["raw_data"]["ws"], label="风速")
    ax2_2.plot(raw_time, result["raw_data"]["gp"], color="orange", label="功率")
    ax2.legend(loc="lower left")
    ax2_2.legend(loc="lower right")
    ax2.set_title("raw data:风速/功率-时间")
    ax2.grid()

    if result['analysis_data'] and result['analysis_data']["density_x12"]:
        ax3.plot(result["analysis_data"]["base_x"], result["analysis_data"]["base_y"], c="r", label="base")
        ax3.plot(result["analysis_data"]["density_x12"], result["analysis_data"]["density_y12"], "--b", label="diff12")
        ax3.plot(result["analysis_data"]["density_x13"], result["analysis_data"]["density_y13"], "--y", label="diff13")
        ax3.plot(result["analysis_data"]["density_x23"], result["analysis_data"]["density_y23"], "--g", label="diff23")
        ax3.set_title(
            f"健康值 1/2/3: {res_index['result'][-1][0]}/{res_index['result'][-1][1]}/{res_index['result'][-1][2]},   预警等级1/2/3: {res_index['alarm'][-1][0]}/{res_index['alarm'][-1][1]}/{res_index['alarm'][-1][2]}")
        ax3.set_ylabel("温差特征值")
        ax3.legend()
    ax0.plot(res_index["time"], np.array(res_index["result"])[:, 0], marker="D", markerfacecolor="cyan", label="变桨电机1")
    ax0.plot(res_index["time"], np.array(res_index["result"])[:, 1], marker="D", markerfacecolor="blue", label="变桨电机2")
    ax0.plot(res_index["time"], np.array(res_index["result"])[:, 2], marker="D", markerfacecolor="red", label="变桨电机3")
    ax0.set_ylabel("偏差角度")
    ax0.set_ylim([0, 110])
    ax0.legend(loc="upper left")
    ax0.grid()
    plt.show()

if __name__ == "__main__":
#     plotflag = True
#     online = False #离线运行

#     folder = "../../data" if isJFWT else "../../data/MY"

#     for i in ["W019","W031","W010"]:#"W056","W070","W085"  MY: "W019","W031","W010"
#         with open("./Model/pitch_motor_temp_abnormal"+".pkl","rb") as f:
#             model_all = pickle.load(f)
#         print("*"*50,i,"*"*50)
#         model = model_all[i]
#         data = pd.read_csv(os.path.join(folder,i+".csv"))
#         data["time"] = pd.to_datetime(data.time) + datetime.timedelta(hours=8)
#         data.set_index("time",drop=False,inplace=True)
#         if i == "W010":
#             # has_wtstatus == "False"
#             data[main_status] = 14
#         columns = [time, wind_speed, active_power, pbt1, pbt2, pbt3,main_status,ba1_speed,ba2_speed,ba3_speed] 
#         data = data[columns]["2021-6-1":]

#         periods = pd.date_range(start = data["time"].min(),end=data.time.max(),freq = '1D',normalize=True)#

#         res_index ={"time":[],"result":[],"alarm":[]}
#         for j in range(len(periods)-28):#  62,67    45,50   len(periods)-1   len(periods)-28
#             data_ = data[periods[j]:periods[j+1]] 
#             print("*"*10,j,pd.to_datetime(periods[j+1]),end="    ")#
#             if len(data_) > 0:
#                 data_ = data_preprocess(data_)
#                 # if i == "W010":# 10没有运行状态变量
#                 #     data_[main_status] = 14
#                 res = pitch_motor_temp_abnormal_main(data_,model)
#                 res_index["time"].append(pd.to_datetime(periods[j+1]))
#                 res_index["result"].append([res["result1"],res["result2"],res["result3"]])
#                 res_index["alarm"].append([res["alarm1"],res["alarm2"],res["alarm3"]])
#         if plotflag:
#             result_plot(res,res_index)

    # 在线数据测试
    import json

    with open("../Model/pitch_motor_temp_abnormal"+".pkl", "rb") as f:
        model_all = pickle.load(f)
        # model = model_all["W019"]
    with open("../data/PitchMotorTemperatureAbnormalDetection-19-data.json", "r") as f:
        data = json.load(f)
    res = pd.DataFrame(np.array(data), columns=["info"])
    for col in ['pitch_temp_1', 'pitch_temp_2', 'pitch_temp_3', 'active_power', 'wind_speed', 'main_status',
                'dataTimeStamp', 'assetName', 'pitch_speed_1', 'pitch_speed_2', 'pitch_speed_3']:
        res[col] = res["info"].apply(lambda x: x.get(col, None))
    pitch_motor_temp_abnormal_main(res, model_all)
